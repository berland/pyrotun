import os
import argparse
import asyncio
import pytz
import datetime
from pathlib import Path

from matplotlib import pyplot
import pandas as pd

import dotenv

import pyrotun
from pyrotun import persist  # noqa


async def make_heatingmodel(influx, target, ambient, powermeasure):
    """Make a linear heating model, for how  much wattage is needed
    to obtain a target temperature

    Args:
        influx:  A pyrotun.connection object to InfluxDB
        target: item-name
        ambient: item-name
        powermeasure: item-name, unit=Watt
    """
    raise NotImplementedError
    # Resampling in InfluxDB over Pandas for speed reasons.
    # target_series = await influx.get_series_grouped(target, time="1h")
    # ambient_series = await influx.get_series_grouped(ambient, time="1h")
    # power_series = await influx.get_series_grouped(powermeasure, time="1h")
    # Substract waterheater from power_series.


async def non_heating_powerusage(influx):
    """Return a series with hour sampling  for power usage that is not
    used in heating, typically waterheater and electrical car"""
    raise NotImplementedError
    # cum_usage = await influx.get_series("Varmtvannsbereder_kwh_sum")
    # The cumulative series is perhaps regularly reset to zero.


async def estimate_savings_yesterday(pers, dryrun):
    yesterday = datetime.date.today() - datetime.timedelta(hours=24)

    results = await estimate_savings(pers, 2, get_daily_profile(), plot=False)
    if not dryrun:
        await pers.openhab.set_item(
            "PowercostSavingsYesterday",
            float(results.loc[yesterday]["savings"]),
            log=True,
        )


async def estimate_savings(pers, daycount, norway_daily_profile, plot=False):
    """Calculate the savings from pre-heating house from Dijkstra
    optimization compared to a an average Norwegian 24h power usage profile"""
    tz = pytz.timezone(os.getenv("TIMEZONE"))
    prof = norway_daily_profile

    prices = await pers.influxdb.dframe_query(
        f"SELECT mean(value) FROM Tibber_current_price "
        f"where time > now() - {daycount}d group by time(1h)"
    )

    cons = (
        await pers.influxdb.dframe_query(
            f"SELECT mean(value) FROM AMSpower "
            f"where time > now() - {daycount}d "
            f"group by time(1h) fill(previous)"
        )
        / 1000
    )

    dframe = pd.concat([prices, cons], axis="columns").dropna(axis="rows")
    dframe.columns = ["price", "usage"]
    dframe.index = dframe.index.tz_convert(tz)
    dframe["hour"] = dframe.index.hour
    dframe["date"] = dframe.index.date
    dframe = pd.merge(dframe, prof, on="hour")

    results = []
    for date, df_date in dframe.groupby("date"):
        if date.weekday() < 6:
            df_date["scaled_profile"] = (
                df_date["usage"].sum() / df_date["kwh-day"].sum() * df_date["kwh-day"]
            )
        else:
            df_date["scaled_profile"] = (
                df_date["usage"].sum()
                / df_date["kwh-weekend"].sum()
                * df_date["kwh-weekend"]
            )

        profcost = (df_date["scaled_profile"] * df_date["price"] / 100).sum()
        cost = (df_date["usage"] * df_date["price"] / 100).sum()
        savings = profcost - cost
        results.append(
            {"date": date, "cost": cost, "profcost": profcost, "savings": savings}
        )
    res = pd.DataFrame(results).set_index("date")
    print(res)
    print(res.index.dtype)
    print("Total savings: " + str(res["savings"].sum()))
    if plot:
        res.plot(y="savings")
        pyplot.show()
    return res


async def main(pers=None):
    closepers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["influxdb", "openhab"])
        closepers = True

    # await estimate_savings_yesterday(pers, dryrun=False)
    await estimate_savings(pers, 30, get_daily_profile())

    if closepers:
        await pers.aclose()


def get_daily_profile():
    return pd.read_csv(Path(__file__).parent / "daypowerprofile.csv", comment="#")


def get_parser():
    parser = argparse.ArgumentParser()
    return parser


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = get_parser()
    args = parser.parse_args()
    asyncio.run(
        main(
            pers=None,
        )
    )
