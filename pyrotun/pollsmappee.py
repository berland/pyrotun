import logging

from pyrotun.connections import smappee, mqtt


def main():
    mysmappee = smappee.SmappeeConnection()
    mqttconnection = mqtt.MqttConnection()

    wattage = mysmappee.avg_watt_5min()

    logging.info("Last 5 min wattage is %s", str(wattage))
    mqttconnection.publish("smappee/total/5min", mysmappee.avg_watt_5min())

    daily_cum = mysmappee.get_daily_cum()
    logging.info("Daily cumulative power usage is %s", str(daily_cum))
    mqttconnection.publish("smappee/total/daycum", daycum)

if __name__ == "__main__":
    main()
