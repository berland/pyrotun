"""Module for controlling/truncating house power usage

To be run()'ed every minute.

"""
import asyncio
import datetime
from pyrotun import persist
import pandas as pd

CURRENT_POWER_ITEM = "AMSpower"
HOURUSAGE_ESTIMATE_ITEM = "WattHourEstimate"
MAXHOURWATT_LASTMONTH_ITEM = "MaxHourwatt_lastmonth"


def run(pers, dry=True):

    temp_plan = pers.powermodels["temperatureplan"]
    assert isinstance(temp_plan, pd.DataFrame)


async def estimate_currenthourusage(pers):
    lasthour = datetime.datetime.utcnow().replace(second=0, minute=0, microsecond=0)
    lastminute = datetime.datetime.utcnow() - datetime.timedelta(minutes=1)

    query = f"SELECT * FROM {CURRENT_POWER_ITEM} WHERE time > '{lasthour}'"

    lasthour_df = await pers.influxdb.dframe_query(query)

    # Use last minute for extrapolation:
    lastminute = await pers.influxdb.dframe_query(
        f"SELECT mean(*) FROM {CURRENT_POWER_ITEM} WHERE time > '{lastminute}'"
    )
    return _estimate_currenthourusage(lasthour_df["value"], lastminute.values[0][0])


def _estimate_currenthourusage(lasthour_series: pd.Series, lastminute_value: float):
    if lasthour_series.empty:
        return round(lastminute_value)
    time_min = lasthour_series.index.min()
    time_max = time_min + datetime.timedelta(hours=1)
    lasthour_s = lasthour_series.resample("s").mean().fillna(method="ffill")
    remainder_hour = pd.Series(
        index=pd.date_range(
            start=lasthour_s.index[-1] + datetime.timedelta(seconds=1),
            end=time_max - datetime.timedelta(seconds=1),  # end at :59:59
            freq="s",
        ),
        data=lastminute_value,
    )
    full_hour = pd.concat([lasthour_s, remainder_hour])
    return round(full_hour.mean())

async def main():
    pers = persist.PyrotunPersistence()
    await pers.ainit(["influxdb"])
    est = await estimate_currenthourusage(pers)
    print(f"Estimated power usage for current hour is: {est} Wh")
    await pers.aclose()

if __name__ == "__main__":
    asyncio.run(main())
