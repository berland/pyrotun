import asyncio
import pandas as pd

import pyrotun
from pyrotun.connections import tibber, mqtt, smappee, openhab

logger = pyrotun.getLogger(__name__)

NETTLEIE = 0.4154


async def main(connections=None):
    if connections is None:
        connections = {
            "tibber": tibber.TibberConnection(),
            "mqtt": mqtt.MqttConnection(),
            "smappee": smappee.SmappeeConnection(),
            "openhab": openhab.OpenHABConnection(),
        }
        await connections["tibber"].ainit()
    if connections["tibber"] is None:
        connections["tibber"] = tibber.TibberConnection()
        await connections["tibber"].ainit()

    prices_df = await connections["tibber"].get_prices()
    # logger.info("Tibber prices %s", str(prices_df))

    daily_consumption = connections["smappee"].get_daily_df()
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
    connections["openhab"].set_item("Tibber_day_cumcost", todayscost)

    logger.info(
        "Gjennomsnittspris til nå i dag: %s øre",
        str(round(dframe["NOK/KWh"].mean()*100, 2)),
    )

    nupris = await connections["tibber"].get_currentprice()
    logger.info("Pris nå: %s øre", nupris)
    connections["openhab"].set_item("Tibber_current_price", nupris)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
