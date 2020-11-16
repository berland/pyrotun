import pyrotun
from pyrotun.connections import smappee, mqtt, openhab

logger = pyrotun.getLogger(__name__)


def main(connections=None):
    if connections is None:
        connections = {
            "smappee": smappee.SmappeeConnection(),
            "mqtt": mqtt.MqttConnection(),
            "openhab": openhab.OpenHABConnection(),
        }

    wattage = connections["smappee"].avg_watt_5min()

    logger.info("Last 5 min wattage is %s W", str(wattage))
    # connections["mqtt"].publish("smappee/total/5min", wattage)
    connections["openhab"].set_item("Smappee_avgW_5min", wattage)

    daily_cum = connections["smappee"].get_daily_cum()
    logger.info("Daily cumulative power usage is %s KWh", str(daily_cum))
    # connections["mqtt"].publish("smappee/total/daycum", daily_cum)
    connections["openhab"].set_item("Smappee_day_cumulative", daily_cum)


if __name__ == "__main__":
    main()
