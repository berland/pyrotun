import pyrotun
from pyrotun.connections import tibber, mqtt

logger = pyrotun.getLogger(__name__)



async def main(connections=None):
    if connections is None:
        connections = await {
            "tibber": tibber.TibberConnection(),
            "mqtt": mqtt.MqttConnection(),
        }
        await connections["tibber"].ainit()
    if connections["tibber"] is None:
        connections["tibber"] = tibber.TibberConnection()
        await connections["tibber"].ainit()

    prices_df = await connections["tibber"].get_prices()
    logger.info("Tibber prices %s", str(prices_df))


if __name__ == "__main__":
    main()
