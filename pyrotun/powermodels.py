import argparse
import asyncio
import datetime
import math
import os
from contextlib import suppress

import astral
import astral.sun
import dotenv
import numpy as np
import pandas as pd
import pvlib
import seaborn as sns
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
        self.solarpanel_heatmapdata: pd.DataFrame | None = None

        self.pers = None

    async def ainit(self, pers):
        logger.info("PowerHeatingModels.ainit() (as background task)")
        self.pers = pers
        self._update_heatingmodel_task = asyncio.create_task(self.update_heatingmodel())

    async def update_heatingmodel(self):
        with suppress(Exception):
            models = await make_heatingmodel(self.pers)
            self.powermodel = models["powermodel"]
            self.tempmodel = models["tempmodel"]
            await asyncio.sleep(0.01)
            self.sunheatingmodel = await sunheating_model(self.pers)

            logger.info("Getting averaged SolcelleWatt for all history")
            df = await self.pers.influxdb.get_series_grouped(
                "SolcelleWatt", time="15m", offset="-7m30s"
            )
            logger.info("Making solarwatt heatmap")
            self.solarpanel_heatmapdata = await make_solarwatt_heatmap(df)
            logger.info("Solarwatt heatmap data available")

    def predict_solarwatt_by_timestamp(
        self, timestamp: datetime.datetime | None = None
    ) -> float | None:
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.UTC)
        if self.solarpanel_heatmapdata is None:
            logger.warning("Heatmap data for solar not ready (yet?)")
            return None
        stencil = [-10, -5, 0, 5, 10]
        stencil_prediction = [
            predict_solarwatt(
                self.solarpanel_heatmapdata,
                timestamp + datetime.timedelta(minutes=stencil_element),
            )
            for stencil_element in stencil
        ]
        return float(
            np.mean([value for value in stencil_prediction if value is not None])
        )


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

        # Should delete days where max(sunheight) == 0

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


def recalculate_sun_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate solar elevation and azimuth using pvlib.
    (pvlib is faster than astral as it supports vectorized operations)
    """
    latitude = float(os.getenv("LOCAL_LATITUDE", "0"))
    longitude = float(os.getenv("LOCAL_LONGITUDE", "0"))

    sun_positions = pvlib.solarposition.get_solarposition(
        time=df.index,
        latitude=latitude,
        longitude=longitude,
        altitude=58,
    )

    df = df.copy()
    df["Solhoyde"] = sun_positions["apparent_elevation"].to_numpy()
    df["Solposisjon"] = sun_positions["azimuth"].to_numpy()
    return df


async def make_solarwatt_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Recalculating sun positions")
    df = await asyncio.to_thread(recalculate_sun_positions, df)
    logger.info("Recalculating sun positions - done!")
    df["hour"] = df.index.hour  # UTC!! (kan ikke bruke lokal tid pga sommertid)

    az_min, az_max = (
        math.floor(df["Solposisjon"].min()),
        math.ceil(df["Solposisjon"].max()),
    )
    el_min, el_max = 0, math.ceil(df["Solhoyde"].max())

    h_bins = np.arange(el_min, el_max + 2, 1)
    a_bins = np.arange(az_min, az_max + 5, 5)
    df["h_bin"] = pd.cut(df["Solhoyde"], bins=h_bins)
    df["a_bin"] = pd.cut(df["Solposisjon"], bins=a_bins)
    heatmap_data = (
        df.groupby(["h_bin", "a_bin"], observed=False)["SolcelleWatt"].max().unstack()
    )
    heatmap_data.sort_index(ascending=True)
    heatmap_data = heatmap_data.cummax(axis=0)
    return heatmap_data


def plot_solarpanel_heatmapdata(heatmap_data: pd.DataFrame) -> None:
    pyplot.figure(figsize=(14, 7))
    sns.heatmap(
        heatmap_data.astype(float),
        cmap="viridis",
        cbar_kws={"label": "Max wattage"},
    )
    pyplot.title("Maximum solar output by sun position")

    pyplot.xlabel("Sun azimuth (Degrees)")
    pyplot.ylabel("Sun elevation (Degrees)")

    # Invert y-axis so 90° (zenith) is at the top
    pyplot.gca().invert_yaxis()
    pyplot.show()


def predict_solarwatt(
    heatmap_data: pd.DataFrame, timestamp: datetime.datetime | None = None
) -> float | None:
    observer = astral.Observer(
        latitude=float(os.getenv("LOCAL_LATITUDE", "0")),
        longitude=float(os.getenv("LOCAL_LONGITUDE", "0")),
        elevation=58,
    )
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.UTC)
    sun_height = astral.sun.elevation(observer, timestamp)
    sun_azimuth = astral.sun.azimuth(observer, timestamp)

    height_indexes = [idx for idx in heatmap_data.index if sun_height in idx]
    if not height_indexes:
        return None
    else:
        height_index = height_indexes[0]
    azimuth_indexes = [idx for idx in heatmap_data.columns if sun_azimuth in idx]
    if not azimuth_indexes:
        return None
    else:
        azimuth_index = azimuth_indexes[0]

    assert height_index is not None
    assert azimuth_index is not None
    return float(heatmap_data.loc[height_index, azimuth_index])


async def solarpanelmodel(pers=None):
    closepers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["influxdb"])
        closepers = True

    # Apply 15 minute average to avoid unnatural spikes occuring
    # when clouds are passing by (roughly one datapoint pr 5 minute)
    df = await pers.influxdb.get_series_grouped(
        "SolcelleWatt", time="15m", offset="-7m30s"
    )

    heatmap_data = make_solarwatt_heatmap(df)
    predict_solarwatt(heatmap_data)
    plot_solarpanel_heatmapdata(heatmap_data)

    if closepers:
        await pers.aclose()


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
    # asyncio.run(elva_main(pers=None))
    asyncio.run(solarpanelmodel(pers=None))
