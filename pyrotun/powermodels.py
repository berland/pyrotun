import argparse
import asyncio
from contextlib import suppress

import dotenv
import pandas as pd
import sklearn
from matplotlib import pyplot

import pyrotun
from pyrotun import persist  # noqa

logger = pyrotun.getLogger(__name__)


class Powermodels:
    """Container class for various statistical models
    related to power/heating"""

    def __init__(self):
        self.powermodel = None
        self.tempmodel = None
        self.sunheatingmodel = None

        self.pers = None

    async def ainit(self, pers):
        logger.info("PowerHeatingModels.ainit()")
        self.pers = pers
        await self.update_heatingmodel()

    async def update_heatingmodel(self):
        with suppress(Exception):
            models = await make_heatingmodel(self.pers)
            self.powermodel = models["powermodel"]
            self.tempmodel = models["tempmodel"]
            await asyncio.sleep(0.01)
            self.sunheatingmodel = await sunheating_model(self.pers)


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
        # pyplot.show()

    logger.info(
        "Innetemp_maxtemp vs clouds: How much can we explain? %.2f"
        % powermodel.score(X, y)
    )
    logger.info("Coefficients %s" % str(powermodel.coef_))

    return powermodel


async def make_heatingmodel(
    pers,
    target: str = "Sensor_fraluft_temperatur",
    ambient: str = "Netatmo_ute_temperatur",  # "UteTemperatur",
    powermeasure: list | str = "Smappee_avgW_5min",
    include_sun: bool = True,
) -> dict:
    """Make heating and/or power models.

    Args:
        influx:  A pyrotun.connection object to InfluxDB
        target: item-name
        ambient: item-name
        powermeasure: item-name, unit=Watt
    """
    # Resampling in InfluxDB over Pandas for speed reasons.
    # BUG: get_series_grouped returns dataframe..
    target_series = (await pers.influxdb.get_series_grouped(target, time="1h"))[target]
    await asyncio.sleep(0.01)
    ambient_series = (await pers.influxdb.get_series_grouped(ambient, time="1h"))[
        ambient
    ]
    await asyncio.sleep(0.01)
    if not isinstance(powermeasure, list):
        powermeasures = [powermeasure]
    else:
        powermeasures = powermeasure

    powerdata = {}
    for measure in powermeasures:
        await asyncio.sleep(0.01)
        print(measure)
        powerdata[measure] = (
            # Må være på 10sek eller lavere for god forklaringsevne.
            # 1 sekund-oppløsning krever 0.5Mb overføring
            (await pers.influxdb.get_series_grouped(measure, time="10s"))[measure]
            .ffill()
            .resample("1h")
            .mean()
        )
    print("done")

    power_series = (
        pd.concat([powerdata[x] for x in powerdata], axis=1).fillna(value=0).sum(axis=1)
    )

    # Sun altitude in degrees:
    if include_sun:
        sunheight = (await pers.influxdb.get_series_grouped("Solhoyde", time="1h"))[
            "Solhoyde"
        ].clip(lower=0)

        # Todo: delete days where max(sunheight) == 0

        # No contribution from low sun (terrain)
        sunheight[sunheight < 5] = 0

        cloud_area_fraction = await pers.yr.get_historical_cloud_fraction()
        if cloud_area_fraction is None:
            logger.error("Not able to get cloud fraction from yr")
            return {"powermodel": None, "tempmodel": None}
        sun_cloud = (
            pd.concat([cloud_area_fraction, sunheight], axis=1)
            .sort_index()
            .ffill()
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
        irradiation = sun_cloud["Irradiation"]
    else:
        irradiation = pd.Series()

    if "Smappee" in powermeasure:
        # Trekk fra vvb:
        substract = await non_heating_powerusage(pers)
        heating_power = power_series / 1000 - substract
    else:
        heating_power = power_series / 1000

    heating_power.name = "HeatingPower"
    heating_power.clip(lower=0, inplace=True)

    if include_sun:
        cols = [target_series, ambient_series, heating_power, irradiation]
    else:
        cols = [target_series, ambient_series, heating_power]
    dataset = pd.concat(
        cols,
        join="inner",
        axis=1,
    ).dropna()

    if "Kjoleskap" in target:
        # Må trekke fra når vi er tilstede..
        setpoint = await pers.influxdb.get_series_grouped(
            "Namronovn_Gang_600w_setpoint", time="1h"
        )
        setpoint.ffill(inplace=True)
        target_series = target_series[(setpoint < 13).index]

    dataset["indoorvsoutdoor"] = dataset[target] - dataset[ambient]
    dataset["indoorderivative"] = dataset[target].diff()

    # Remove impossible indoor derivatives:
    dataset = dataset[dataset["indoorderivative"].abs() < 1]

    dataset.dropna(inplace=True)  # Drop egde NaN due to diff()

    dataset.plot.scatter(
        x="indoorderivative",
        y="HeatingPower",
        c="indoorvsoutdoor",
        grid=True,
        # positiv indoorvsoutdoor betyr varmere inne enn ute
    )
    pyplot.show()

    lm = sklearn.linear_model.LinearRegression(fit_intercept=False)
    modelparameters = ["indoorderivative", "indoorvsoutdoor"]  # , "Irradiation"]
    X = dataset[modelparameters].values
    y = dataset[["HeatingPower"]].values

    powermodel = lm.fit(X, y)
    logger.info("Powerusage explanation %.3f", powermodel.score(X, y))
    logger.info(f"Coefficients {powermodel.coef_}")
    logger.info(f"Intercept {powermodel.intercept_}")
    logger.info(" - in variables: %s", str(modelparameters))

    logger.info(
        "Preheating requirement: %f",
        (powermodel.coef_[0][1] / powermodel.coef_[0][0] + 1),
    )

    if include_sun:
        # "explanation factor" increases  by only one percent point by including
        # Irradiation.. Predictive power???
        p2 = ["HeatingPower", "indoorvsoutdoor", "Irradiation"]
        y2 = dataset["indoorderivative"]
        lm2 = sklearn.linear_model.LinearRegression()
        tempmodel = lm2.fit(dataset[p2], y2)
        logger.info("Temp increase explanation: %.3f", tempmodel.score(dataset[p2], y2))
        logger.info("Coefficients %s", str(tempmodel.coef_))
        logger.info(" - in variables: %s", str(p2))
    else:
        tempmodel = None

    return {"powermodel": powermodel, "tempmodel": tempmodel}


async def non_heating_powerusage(pers):
    """Return a series with hour sampling  for power usage that is not
    used in heating, typically waterheater and electrical car.

    Returns time-series with unit kwh.
    """
    cum_item = "Varmtvannsbereder_kwh_sum"
    cum_usage = (await pers.influxdb.get_series(cum_item))[cum_item]
    cum_usage_rawdiff = cum_usage.diff()
    cum_usage = cum_usage[(cum_usage_rawdiff > 0) & (cum_usage_rawdiff < 1000)].dropna()

    cum_usage_hourly = (
        cum_usage.resample("1h").mean().interpolate(method="linear").diff().shift(-1)
    )
    return cum_usage_hourly[(cum_usage_hourly > 0) & (cum_usage_hourly < 3.0)]


async def elva_main(pers=None):
    closepers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["influxdb", "openhab", "yr"])
        closepers = True

    res = await make_heatingmodel(
        pers,
        target="Sensor_Kjoleskap_temperatur",
        ambient="YrtemperaturMjolfjell",
        powermeasure=[
            "Namronovn_Stue_800w_effekt",
            "Namronovn_Bad_400w_effekt",
            "Namronovn_Gang_600w_effekt",
        ],
        include_sun=False,
    )

    print(res)
    if closepers:
        await pers.aclose()


async def main(pers=None):
    closepers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["influxdb", "openhab", "yr"])
        closepers = True

    res = await make_heatingmodel(
        pers,
        include_sun=False,
    )
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
    asyncio.run(elva_main(pers=None))
