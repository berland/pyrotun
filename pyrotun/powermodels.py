import argparse
import asyncio

import dotenv
import pandas as pd
import sklearn
from matplotlib import pyplot

import pyrotun
from pyrotun import persist  # noqa

logger = pyrotun.getLogger(__name__)


class Powermodels:
    """Container class for various statistical models
    related to power/heating:
    
    * powermodel: How many watts are needed to raise house temperature
      at a given outdoor difference
    * tempmodel: How much the indoor temperature reacts to power usage
    * sunheatingmodel: How much sun and cloud-cover affects maximal
      indoor temperature during a day.

    """

    def __init__(self):
        self.powermodel = None
        self.tempmodel = None
        self.sunheatingmodel = None

        self.pers = None

    async def ainit(self, pers):
        logger.info("PowerHeatingModels.ainit()")
        self.pers = pers

        models = await make_heatingmodel(pers)
        self.powermodel = models["powermodel"]
        self.tempmodel = models["tempmodel"]
        self.sunheatingmodel = await sunheating_model(pers)


async def sunheating_model(pers, plot=False):
    """Make a model of how much sun and cloud-cover affects maximal
    indoor temperature during a day.

    For prediction from weather forecast, ensure that the equation for making
    the irradiation proxy is the same.

    After this algorithm was applied, the training data is destroyed, so
    we need to limit the training data set to a small set of days in april 2021..
    """
    # Indoor temp:
    indoor = (await pers.influxdb.get_series_grouped("InneTemperatur", time="1h"))[
        "InneTemperatur"
    ]
    # After this algorithm was applied, the training data is destroyed, so
    # we need to limit the training data set to a small set of days in april 2021..
    indoor = indoor[indoor.index < pd.to_datetime("2021-05-01 00:00:00+00:00")]

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

    print(sunheight)
    print(sunheight.max())
    print(sunheight.min())
    print(cloudcover)
    print(cloudcover.max())
    print(cloudcover.min())
    irradiation_proxy = sunheight * (1 - cloudcover / 100) / 100
    # 1/100 is just a scaling factor to get number in the ballpark 0 - 10.
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
        pyplot.plot(
            dataset["IrradiationProxy"],
            lm.predict(dataset["IrradiationProxy"].values.reshape(-1, 1)),
            color="blue",
            linewidth=3,
        )
        pyplot.show()

    logger.info(
        "Innetemp_maxtemp vs clouds: How much can we explain? %.2f"
        % powermodel.score(X, y)
    )
    logger.info("Coefficients %s" % str(powermodel.coef_))

    return powermodel


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

    # Todo: delete days where max(sunheight) == 0

    # No contribution from low sun (terrain)
    sunheight[sunheight < 5] = 0

    cloud_area_fraction = await pers.yr.get_historical_cloud_fraction()
    sun_cloud = (
        pd.concat([cloud_area_fraction, sunheight], axis=1)
        .sort_index()
        .fillna(method="ffill")
        .dropna()
    )
    sun_cloud["Irradiation"] = (1 - sun_cloud["cloud_area_fraction"]) * sun_cloud[
        "Solhoyde"
    ]
    # Cleanse data:
    sun_cloud["date"] = sun_cloud.index.date.astype(str)
    max_sun_cloud_date = sun_cloud.groupby("date").max()
    erroneous_dates = max_sun_cloud_date[max_sun_cloud_date["Solhoyde"] < 0.1].index
    for error_date in erroneous_dates:
        sun_cloud = sun_cloud[sun_cloud["date"] != error_date]

    substract = await non_heating_powerusage(pers)
    heating_power = power_series / 1000 - substract
    heating_power.name = "HeatingPower"
    heating_power.clip(lower=0, inplace=True)

    dataset = pd.concat(
        [target_series, ambient_series, heating_power, sun_cloud["Irradiation"]],
        axis=1,
    ).dropna()

    dataset["indoorvsoutdoor"] = dataset[target] - dataset[ambient]
    dataset["indoorderivative"] = dataset[target].diff()
    dataset.dropna(inplace=True)  # Drop egde NaN due to diff()
    dataset.plot(y="HeatingPower")

    lm = sklearn.linear_model.LinearRegression()
    modelparameters = ["indoorderivative", "indoorvsoutdoor"]  # , "Irradiation"]
    X = dataset[modelparameters]
    y = dataset[["HeatingPower"]]

    powermodel = lm.fit(X, y)
    logger.info("Powerusage explanation %.3f", powermodel.score(X, y))
    logger.info("Coefficients %s", str(powermodel.coef_))
    logger.info(" - in variables: %s", str(modelparameters))

    logger.info(
        "Preheating requirement: %f",
        (powermodel.coef_[0][1] / powermodel.coef_[0][0] + 1),
    )

    # "explanation factor" increases  by only one percent point by including
    # Irradiation.. Predictive power???
    p2 = ["HeatingPower", "indoorvsoutdoor", "Irradiation"]
    y2 = dataset["indoorderivative"]
    lm2 = sklearn.linear_model.LinearRegression()
    tempmodel = lm2.fit(dataset[p2], y2)
    logger.info("Temp increase explanation: %.3f", tempmodel.score(dataset[p2], y2))
    logger.info("Coefficients %s", str(tempmodel.coef_))
    logger.info(" - in variables: %s", str(p2))

    return {"powermodel": powermodel, "tempmodel": tempmodel}

 
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
    return cum_usage_hourly[(0 < cum_usage_hourly) & (cum_usage_hourly < 3.0)]


async def main(pers=None):
    closepers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["influxdb", "openhab", "yr"])
        closepers = True

    res = await make_heatingmodel(pers)
    print(res)

    print("Sun heating model")
    sun = await sunheating_model(pers)
    print(sun)

    if closepers:
        await pers.aclose()


def get_parser():
    parser = argparse.ArgumentParser()
    return parser


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = get_parser()
    args = parser.parse_args()
    asyncio.run(main(pers=None))
