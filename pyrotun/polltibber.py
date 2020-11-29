import asyncio
import pandas as pd
import dotenv

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

NETTLEIE = 0.4154

PERS = None


async def main(pers=PERS):
    dotenv.load_dotenv()
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence(readonly=True)
    if pers.tibber is None:
        await pers.ainit(["tibber", "smappee", "openhab"])
    prices_df = await pers.tibber.get_prices()
    logger.info("Got Tibber prices")  # , str(prices_df))

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
        str(round(dframe["NOK/KWh"].mean() * 100, 2)),
    )

    nupris, priceorder, relpriceorder = await pers.tibber.get_currentprice()
    logger.info("Pris nå: %s øre", nupris)
    await pers.openhab.set_item("Tibber_current_price", nupris)
    await pers.openhab.set_item("PowerPriceOrder", priceorder, log=True)
    await pers.openhab.set_item("RelativePowerPriceOrder", relpriceorder, log=True)


if __name__ == "__main__":
    PERS = pyrotun.persist.PyrotunPersistence()
    asyncio.get_event_loop().run_until_complete(main())
