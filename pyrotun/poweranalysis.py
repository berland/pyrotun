import os
import argparse
import asyncio
import pytz
import datetime
from pathlib import Path

import sklearn
from matplotlib import pyplot
import pandas as pd

import dotenv

import pyrotun
from pyrotun import persist  # noqa

logger = pyrotun.getLogger(__name__)

async def make_heatingmodel(
    pers,
    target="Sensor_fraluft_temperatur",
    ambient="Netatmo_ute_temperatur",  # "UteTemperatur",
    powermeasure="Smappee_avgW_5min",
):
    """Make heating and/or power models.

    Args:
        influx:  A pyrotun.connection object to InfluxDB
        target: item-name
        ambient: item-name
        powermeasure: item-name, unit=Watt
    """
    target = "Sensor_fraluft_temperatur"
    # Resampling in InfluxDB over Pandas for speed reasons.
    # BUG: get_series_grouped returns dataframe..
    target_series = (await pers.influxdb.get_series_grouped(target, time="1h"))[target]
    ambient_series = (await pers.influxdb.get_series_grouped(ambient, time="1h"))[
        ambient
    ]
    power_series = (await pers.influxdb.get_series_grouped(powermeasure, time="1h"))[
        powermeasure
    ]

    # Sun altitude in degrees:
    sunheight = (await pers.influxdb.get_series_grouped("Solhoyde", time="1h"))[
        "Solhoyde"
    ].clip(lower=0)

    # No contribution from low sun (terrain)
    sunheight[sunheight < 10] = 0

    # Cloud cover (sort of. Number 12 is chosen as it gives the highest explanation in regression)
    yrmelding = 12- (
        (await pers.influxdb.get_series_grouped("YrmeldingNaa", time="1h"))[
            "YrmeldingNaa"
        ]
        .ffill()
        .clip(lower=1, upper=12)
    )

    irradiation_proxy = (sunheight * yrmelding) / 100
    irradiation_proxy.name = "IrradiationProxy"
    irradiation_proxy.plot()

    substract = await non_heating_powerusage(pers)
    heating_power = power_series / 1000 - substract
    heating_power.name = "HeatingPower"
    heating_power.clip(lower=0, inplace=True)

    dataset = pd.concat(
        [target_series, ambient_series, heating_power, irradiation_proxy], axis=1,
    ).dropna()

    dataset["indoorvsoutdoor"] = dataset[target] - dataset[ambient]
    dataset["indoorderivative"] = dataset[target].diff()
    dataset.dropna(inplace=True)  # Drop egde NaN due to diff()
    dataset.plot(y="HeatingPower")

    lm = sklearn.linear_model.LinearRegression()
    modelparameters = ["indoorderivative", "indoorvsoutdoor", "IrradiationProxy"]
    X = dataset[modelparameters]
    y = dataset[["HeatingPower"]]

    powermodel = lm.fit(X, y)
    print("How much can we explain? %.2f" % powermodel.score(X, y))
    print("Coefficients %s" % str(powermodel.coef_))

    print(" - in variables: %s" % str(modelparameters))
    print("Preheating requirement: %f" % (powermodel.coef_[0][1] / powermodel.coef_[0][0] + 1))

    p2 = ["HeatingPower", "indoorvsoutdoor" ,"IrradiationProxy"]
    y2 = dataset["indoorderivative"]
    lm2 = sklearn.linear_model.LinearRegression()
    tempmodel = lm2.fit(dataset[p2], y2)
    print("How much can we explain? %.3f" % tempmodel.score(dataset[p2], y2))
    print("Coefficients %s" % str(tempmodel.coef_))
    print(" - in variables: %s" % str(p2))

    return {"powerneed": powermodel, "tempincrease": tempmodel}

async def non_heating_powerusage(pers):
    """Return a series with hour sampling  for power usage that is not
    used in heating, typically waterheater and electrical car.

    Returns time-series with unit kwh.
    """
    cum_item = "Varmtvannsbereder_kwh_sum"
    cum_usage = (await pers.influxdb.get_series(cum_item))[cum_item]
    cum_usage_rawdiff = cum_usage.diff()
    cum_usage = cum_usage[(0 < cum_usage_rawdiff) & (cum_usage_rawdiff < 1000)].dropna()

    cum_usage_hourly = (
        cum_usage.resample("1h").mean().interpolate(method="linear").diff().shift(-1)
    )
    return cum_usage_hourly[
        (0 < cum_usage_hourly) & (cum_usage_hourly < 3.0)
    ]  # .clip(lower=0, upper=3.0)

    # cum_usage = await influx.get_series("Varmtvannsbereder_kwh_sum")
    # The cumulative series is perhaps regularly reset to zero.

async def sunheating_model(pers, plot=False):
    """Make a model of how much sun and cloud-cover affects maximal
    indoor temperature during a day.

    For prediction from weather forecast, ensure that the equation for making
    the irradiation proxy is the same.
    """
    # Indoor temp:
    indoor = (await pers.influxdb.get_series_grouped("InneTemperatur", time="1h"))[
        "InneTemperatur"
    ]

    # Sun altitude in degrees:
    sunheight = (await pers.influxdb.get_series_grouped("Solhoyde", time="1h"))[
        "Solhoyde"
    ].clip(lower=0)

    # No contribution from low sun (terrain)
    sunheight[sunheight < 10] = 0

    # Cloud cover
    cloudcover = (
        await pers.influxdb.get_series_grouped("Yr_cloud_area_fraction", time="1h")
    )["Yr_cloud_area_fraction"]

    irradiation_proxy = sunheight * (100 - cloudcover) / 10000
    irradiation_proxy.name = "IrradiationProxy"

    dataset = (
        pd.concat([indoor, sunheight, cloudcover, irradiation_proxy], axis=1)
        .dropna()
        .resample("1d")
        .agg({"InneTemperatur": "max", "IrradiationProxy": "sum"})
        .dropna()
    )

    lm = sklearn.linear_model.LinearRegression()
    modelparameters = ["IrradiationProxy"]
    X = dataset[modelparameters]
    y = dataset[["InneTemperatur"]]

    powermodel = lm.fit(X, y)

    if plot:
        dataset.plot.scatter(x="IrradiationProxy", y="InneTemperatur")
        pyplot.plot(dataset["IrradiationProxy"], lm.predict(dataset["IrradiationProxy"].values.reshape(-1,1)), color='blue', linewidth=3)
        pyplot.show()

    print("How much can we explain? %.2f" % powermodel.score(X, y))
    print("Coefficients %s" % str(powermodel.coef_))

    print(" - in variables: %s" % str(modelparameters))

    return powermodel


async def estimate_savings_yesterday(pers, dryrun):
    yesterday = datetime.date.today() - datetime.timedelta(hours=24)

    results = await estimate_savings(pers, 2, get_daily_profile(), plot=False)
    savings = results.loc[yesterday]["savings"]
    if not dryrun:
        await pers.openhab.set_item(
            "PowercostSavingsYesterday", float(savings), log=True,
        )
    else:
        logger.info("(dryrun) Power savings yesterday %s", str(savings))


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

        minmaxdiff = df_date["price"].max() - df_date["price"].min()
        profcost = (df_date["scaled_profile"] * df_date["price"] / 100).sum()
        cost = (df_date["usage"] * df_date["price"] / 100).sum()
        savings = profcost - cost
        results.append(
            {
                "date": date,
                "cost": cost,
                "profcost": profcost,
                "savings": savings,
                "minmaxdiff": minmaxdiff,
            }
        )
    res = pd.DataFrame(results).set_index("date")
    print(res)
    print(res.index.dtype)
    print("Total savings: " + str(res["savings"].sum()))
    if plot:
        res.plot(y="savings")
        pyplot.show()

        res.plot.scatter(x="minmaxdiff", y="savings")
        pyplot.show()
    return res


async def main(pers=None, days=30, plot=False, yesterday=False):
    closepers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["influxdb", "openhab"])
        closepers = True

    res = await make_heatingmodel(pers)

    if yesterday:
        await estimate_savings_yesterday(pers, dryrun=False)
    else:
        res = await estimate_savings(pers, days, get_daily_profile(), plot=plot)
        print(res)
        print("Total savings: " + str(res["savings"].sum()))


    sunmodel = await sunheating_model(pers, plot)

    if closepers:
        await pers.aclose()


def get_daily_profile():
    return pd.read_csv(Path(__file__).parent / "daypowerprofile.csv", comment="#")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--yesterday", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = get_parser()
    args = parser.parse_args()
    asyncio.run(
        main(pers=None, yesterday=args.yesterday, days=args.days, plot=args.plot)
    )
