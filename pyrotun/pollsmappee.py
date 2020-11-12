import pyrotun
from pyrotun.connections import smappee, mqtt

logger = pyrotun.getLogger(__name__)


def main(connections=None):
    if connections is None:
        connections = {
            "smappee": smappee.SmappeeConnection(),
            "mqtt": mqtt.MqttConnection(),
        }

    wattage = connections["smappee"].avg_watt_5min()

    logger.info("Last 5 min wattage is %s", str(wattage))
    connections["mqtt"].publish(
        "smappee/total/5min", connections["smappee"].avg_watt_5min()
    )

    daily_cum = connections["smappee"].get_daily_cum()
    logger.info("Daily cumulative power usage is %s", str(daily_cum))
    connections["mqtt"].publish("smappee/total/daycum", daily_cum)


if __name__ == "__main__":
    main()
