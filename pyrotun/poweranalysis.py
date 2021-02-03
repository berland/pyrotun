async def make_heatingmodel(influx, target, ambient, powermeasure):
    """Make a linear heating model, for how  much wattage is needed
    to obtain a target temperature

    Args:
        influx:  A pyrotun.connection object to InfluxDB
        target: item-name
        ambient: item-name
        powermeasure: item-name, unit=Watt
    """
    # Resampling in InfluxDB over Pandas for speed reasons.
    target_series = await influx.get_series_grouped(target, time="1h")
    ambient_series = await influx.get_series_grouped(ambient, time="1h")
    power_series = await influx.get_series_grouped(powermeasure, time="1h")

    # Substract waterheater from power_series.


async def non_heating_powerusage(influx):
    """Return a series with hour sampling  for power usage that is not
    used in heating, typically waterheater and electrical car"""

    cum_usage = await influx.get_series("Varmtvannsbereder_kwh_sum")
    # The cumulative series is perhaps regularly reset to zero.



