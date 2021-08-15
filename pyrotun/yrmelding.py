import os
import asyncio

import dotenv

import astral
from astral.sun import sun  # noqa
import datetime

import pyrotun
from pyrotun import persist
import pyrotun.connections.openhab
import pyrotun.connections.yr

logger = pyrotun.getLogger()


dotenv.load_dotenv()
LONGITUDE = os.getenv("LOCAL_LONGITUDE")
LATITUDE = os.getenv("LOCAL_LATITUDE")
MET_CLIENT_ID = os.getenv("FROST_CLIENT_ID")


async def main(pers=None):

    close_pers_here = False
    if pers is None:
        pers = persist.PyrotunPersistence()
        close_pers_here = True
    await pers.ainit(requested=["openhab", "yr", "powermodels", "influxdb"])

    # print(pers.yr.symbolcodedict)
    forecast_df = await pers.yr.forecast()

    # Predict sunheight:
    city = astral.LocationInfo(
        "HOME", os.getenv("LOCAL_CITY"), os.getenv("TIMEZOME"), LATITUDE, LONGITUDE
    )
    forecast_df["sunheight"] = [
        astral.sun.elevation(city.observer, timepoint + datetime.timedelta(minutes=30))
        for timepoint in forecast_df.index
    ]

    irradiation_product = (
        forecast_df["sunheight"].clip(0, 90)
        * (1 - forecast_df["cloud_area_fraction"] / 100)
    ) / 100
    await pers.openhab.set_item("Yr_irradiation_now", irradiation_product[0], log=True)
    await pers.openhab.set_item(
        "Yr_irradiation_next12", irradiation_product[0:12].sum(), log=True
    )
    await pers.openhab.set_item(
        "Yr_irradiation_next24", irradiation_product[0:24].sum(), log=True
    )
    # Calculate how much the sun will increase the house temperature:
    if pers.powermodels is not None:
        soloppvarming = pers.powermodels.sunheatingmodel.predict(
            [[irradiation_product[0:12].sum()]]
        ) - pers.powermodels.sunheatingmodel.predict([[0]])
        await pers.openhab.set_item(
            "Estimert_soloppvarming",
            round(min(max(float(soloppvarming), 0), 5), 1),
            log=True,
        )

    # Each column in the forecast dataframe has its own item in OpenHAB:
    for yr_item in forecast_df.columns:
        if "percentile" in yr_item:
            continue
        await pers.openhab.set_item(
            "Yr_" + yr_item, forecast_df.iloc[0][yr_item], log=True
        )

    # YrmeldingNaa (legacy item):
    await pers.openhab.set_item(
        "YrmeldingNaa",
        pers.yr.symbolcodedict[forecast_df.iloc[0]["symbol_code"].split("_")[0]][
            "old_id"
        ],
        log=True,
    )

    # YrMaksTempNeste6timer
    await pers.openhab.set_item(
        "YrMaksTempNeste6timer", forecast_df.head(6)["air_temperature"].max()
    )

    if close_pers_here:
        await pers.aclose()


if __name__ == "__main__":
    asyncio.run(main())
