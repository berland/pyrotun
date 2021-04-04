import os

import dotenv
import aiohttp
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
