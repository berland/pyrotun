import pyrotun

logger = pyrotun.getLogger(__name__)


async def main(pers):

    # Oops, blocking call..
    wattage = pers.smappee.avg_watt_5min()
    if wattage is not None:
        logger.info("Last 5 min wattage is %s W", str(round(wattage, 1)))
        await pers.openhab.set_item("Smappee_avgW_5min", wattage)

    daily_cum = pers.smappee.get_daily_cum()
    if daily_cum is not None:
        logger.info("Daily cumulative power usage is %s KWh", str(daily_cum))
        await pers.openhab.set_item("Smappee_day_cumulative", daily_cum)


if __name__ == "__main__":
    main()
