import asyncio
import pandas as pd

import pyrotun

logger = pyrotun.getLogger(__name__)

NETTLEIE = 0.4154


async def main(pers):

    prices_df = await pers.tibber.get_prices()
    logger.info("Tibber prices %s", str(prices_df))

    daily_consumption = pers.smappee.get_daily_df()
    # logger.info("Daily consumption %s", str(daily_consumption))

    dframe = (
        pd.concat([daily_consumption, prices_df], axis=1, sort=True)
        .sort_index()
        .dropna()
    )
    dframe["gridrental"] = NETTLEIE
    dframe["cost"] = (
        dframe["consumption"] / 1000 * (dframe["NOK/KWh"] + dframe["gridrental"])
    )

    todayscost = round(dframe["cost"].sum(), 1)  # i kroner
    logger.info("Strømkostnad til nå i dag: %s NOK", todayscost)
    await pers.openhab.set_item("Tibber_day_cumcost", todayscost)

    logger.info(
        "Gjennomsnittspris til nå i dag: %s øre",
        str(round(dframe["NOK/KWh"].mean()*100, 2)),
    )

    nupris = await pers.tibber.get_currentprice()
    logger.info("Pris nå: %s øre", nupris)
    await pers.openhab.set_item("Tibber_current_price", nupris)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
