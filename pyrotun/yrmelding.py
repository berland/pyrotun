import os
import asyncio

import dotenv

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
    await pers.ainit(requested=["openhab", "yr"])

    # print(pers.yr.symbolcodedict)
    forecast_df = await pers.yr.forecast()

    # Each column in the forecast dataframe has its own item in OpenHAB:
    for yr_item in forecast_df.columns:
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
