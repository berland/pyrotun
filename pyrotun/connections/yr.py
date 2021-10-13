import asyncio
import datetime
import os

import aiohttp
import defusedxml.minidom
import dotenv
import pandas as pd

import pyrotun

logger = pyrotun.getLogger()

dotenv.load_dotenv()
LONGITUDE = os.getenv("LOCAL_LONGITUDE")
LATITUDE = os.getenv("LOCAL_LATITUDE")
MET_CLIENT_ID = os.getenv("FROST_CLIENT_ID")


class YrConnection:
    def __init__(self):
        self.symbolsframe = None
        self.old_id_to_symbols = None
        self.symbolcodedict = None
        self.websession = None

    async def ainit(self, websession=None):
        logger.info("yr.ainit()")
        self.websession = websession
        if self.websession is None:
            self.websession = aiohttp.ClientSession()

        symbols_result = await self._symbols()

        self.symbolsframe = symbols_result["dataframe"]
        self.old_id_to_symbols = symbols_result["id_to_symbol"]
        self.symbolcodedict = symbols_result["symbolcodedict"]

    async def nowcast_precipitation_rates(self):
        """Returns a Pandas series with the 5 minute precipitation forecast"""
        async with self.websession.get(
            "https://api.met.no/weatherapi/nowcast/2.0/complete",
            params=dict(lon=LONGITUDE, lat=LATITUDE),
            auth=aiohttp.BasicAuth(MET_CLIENT_ID, ""),
            headers={"User-Agent": "Custom smarthouse using OpenHAB"},
        ) as response:
            result = await response.json()
            dframe = pd.DataFrame(
                [
                    (x["time"], x["data"]["instant"]["details"]["precipitation_rate"])
                    for x in result["properties"]["timeseries"]
                ],
                columns=["datetime", "precipitation_rate"],
            ).set_index("datetime")
            # unit is mm/h
            dframe.index = pd.to_datetime(dframe.index).tz_convert(
                os.getenv("TIMEZONE")
            )
            return dframe["precipitation_rate"]

    async def get_svg_meteogram(self, met_location_id):
        async with self.websession.get(
            "https://www.yr.no/nb/innhold/" + met_location_id + "/meteogram.svg"
        ) as response:
            assert response.status == 200
            svg = await response.text()
            print(svg[0:100])
            return svg

    async def _symbols(self):
        """Return weather symbol information, as a dict with three views of the data"""
        async with self.websession.get(
            "https://api.met.no/weatherapi/weathericon/2.0/legends",
            auth=aiohttp.BasicAuth(MET_CLIENT_ID, ""),
            headers={"User-Agent": "Custom smarthouse using OpenHAB"},
        ) as response:
            result = await response.json()
            dframe = pd.DataFrame([{"symbol_code": x, **result[x]} for x in result])
            return {
                "symbolcodedict": result,
                "dataframe": dframe,
                "id_to_symbol": dframe[["old_id", "symbol_code"]]
                .set_index("old_id")
                .to_dict(),
            }

    async def forecast(self):
        """Returns a dataframe with the current forecast"""
        async with self.websession.get(
            "https://api.met.no/weatherapi/locationforecast/2.0/complete",
            params=dict(lon=LONGITUDE, lat=LATITUDE),
            auth=aiohttp.BasicAuth(MET_CLIENT_ID, ""),
            headers={"User-Agent": "Custom smarthouse using OpenHAB"},
        ) as response:
            result = await response.json()
            time_index = [x["time"] for x in result["properties"]["timeseries"]]
            dframe = pd.DataFrame(
                index=time_index,
                data=[
                    x["data"]["instant"]["details"]
                    for x in result["properties"]["timeseries"]
                ],
            )
            # unit is mm/h
            dframe.index = pd.to_datetime(dframe.index).tz_convert(
                os.getenv("TIMEZONE")
            )

            dframe["symbol_code"] = [
                x["data"]["next_1_hours"]["summary"]["symbol_code"]
                if "next_1_hours" in x["data"]
                else ""
                for x in result["properties"]["timeseries"]
            ]
            prec_frame = pd.DataFrame(
                index=dframe.index,
                data=[
                    x["data"]["next_1_hours"]["details"]
                    if "next_1_hours" in x["data"]
                    else {}
                    for x in result["properties"]["timeseries"]
                ],
            )
            dframe = pd.concat([dframe, prec_frame], axis=1)

            return dframe

    async def get_historical_cloud_fraction(
        self, startdate="2017-01-01", enddate=datetime.datetime.now().date().isoformat()
    ):
        """Returns a series with historical cloud fraction

        Values for every third hour. 0.0 means clear sky, 1.0 means
        overcast.
        """
        async with self.websession.get(
            "https://frost.met.no/observations/v0.jsonld",
            params=dict(
                sources="SN50540",
                referencetime=str(startdate) + "/" + str(enddate),
                elements="cloud_area_fraction",
            ),
            auth=aiohttp.BasicAuth(MET_CLIENT_ID, ""),
            headers={"User-Agent": "Custom smarthouse using OpenHAB"},
        ) as response:
            result = await response.json()
            time_index = [x["referenceTime"] for x in result["data"]]
            cloud_fractions = [x["observations"][0]["value"] for x in result["data"]]
            series = pd.Series(
                cloud_fractions,
                index=pd.to_datetime(time_index),
                name="cloud_area_fraction",
            )
            # the series is of type "octa", where 0 is clear sky, and
            # 8 is totally overcast. Translate this to a fraction beteween
            # 0 (clear sky) and 1.0 (overcast)
            series = series / 8.0
            return series


def crop_svg_meteogram(svg) -> str:
    """Removes logos and stuff from yr's meteogram"""
    doc = defusedxml.minidom.parseString(svg)

    root = doc.childNodes[0]
    # Remove all whitespace nodes:
    for node in root.childNodes:
        if node.toxml().strip() == "":
            node.unlink()

    # node 3 should contain <rect> which fill the background with white.
    root.childNodes[3].setAttribute("height", "280")

    # Remove blue yr-logo:
    assert "circle fill" in root.childNodes[5].toxml()
    root.childNodes[5].unlink()

    assert "En tjeneste fra" in root.childNodes[7].toxml()
    root.childNodes[7].unlink()

    # NRK-logo:
    root.childNodes[9].unlink()

    # Meteorologisk institutt-logo:
    root.childNodes[11].unlink()

    assert "Værvarsel for " in root.childNodes[13].toxml()
    root.childNodes[13].unlink()

    # Node 15 er selve meteogrammet, inkludert dato.
    meteo = root.childNodes[15]
    assert "Vindkast" in meteo.toxml()

    assert meteo.getAttribute("transform") == "translate(0, 84.86)"
    meteo.setAttribute("transform", "translate(0, 0)")

    assert "Vindkast" in meteo.childNodes[11].toxml()
    meteo.childNodes[11].unlink()

    return doc.toxml()


async def main():
    yr = YrConnection()
    await yr.ainit()
    res = await yr.get_historical_cloud_fraction()
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
