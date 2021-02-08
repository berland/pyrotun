#!/bin/env python
import asyncio
import dotenv
import fnmatch
import argparse

import pandas as pd

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

TRUNCATORS = {
    "Termostat_*_SetpointHeating": {"min": 3, "max": 40},
    "Termostat_*_SensorGulv": {"min": 0, "max": 50},
    "InneTemperatur": {"min": 10, "max": 40},
}


async def main(pers=None, readonly=True, hours=48):
    if pers is None:
        dotenv.load_dotenv()
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["influxdb"])

    await truncate(pers, TRUNCATORS, readonly=readonly, hours=hours)
    await remove_spikes(pers, mindev=7, stddevs=3, readonly=readonly, hours=hours)


async def truncate(pers, truncators, readonly=True, hours=48):
    measurements = await pers.influxdb.get_measurements()
    for measurement in measurements:
        for truncator in truncators:
            if fnmatch.fnmatch(measurement, truncator):
                if "min" in truncators[truncator]:
                    query = (
                        f"SELECT value FROM {measurement} "
                        f"where time > now() - {hours}h "
                        f"and value < {truncators[truncator]['min']}"
                    )
                    to_truncate = await pers.influxdb.dframe_query(query)
                    if isinstance(to_truncate, pd.DataFrame):
                        for point in to_truncate.iterrows():
                            await delete_point(
                                pers, measurement, point[0], readonly=readonly
                            )
                if "max" in truncators[truncator]:
                    query = (
                        f"SELECT value FROM {measurement} "
                        f"where time > now() - {hours}h "
                        f"and value > {truncators[truncator]['max']}"
                    )
                    to_truncate = await pers.influxdb.dframe_query(query)
                    if isinstance(to_truncate, pd.DataFrame):
                        for point in to_truncate.iterrows():
                            await delete_point(
                                pers, measurement, point[0], readonly=readonly
                            )


async def remove_spikes(pers, mindev=7, stddevs=3, readonly=True, hours=48):
    """

    Args:
        mindev (int): Minimal deviation from 5-point rolling mean and
            value for a spike (in degrees/RH)
        stddevs (int):  Number of standard devs required for a spike

    """

    measurements = await pers.influxdb.get_measurements()
    measurements = filter_measurements(measurements)

    for measurement in measurements:
        meas_data = await pers.influxdb.dframe_query(
            f"SELECT * FROM {measurement} where time > now() - {hours}h"
        )
        # The dframe has 1 colummn named 'value', and
        # indexed by datetime in UTC
        if isinstance(meas_data, dict):  # this means empty..
            continue
        series = meas_data
        # Find the deviation from a 5-point rolling mean
        deviation = abs(series - series.rolling(5).mean())
        stddev = deviation.dropna().std(axis=0).value

        # print stddev
        if deviation.max(axis=0).value < mindev:
            continue
        # Add to dataframe
        series["deviation"] = deviation
        series.dropna(
            inplace=True
        )  # remove end effects where rolling mean has no value

        if len(series) == 0:  # spikes was at ends, now dropped by statement above
            continue

        spike = series.sort_values("deviation").tail(1)

        # print(deviation)
        # print(deviation[deviation.value > stddev * STDDEVS])
        # Now that we have a spike, delete the
        # rows with highest deviation, above stddev requirement:
        big_deviators = deviation.value > stddev * stddevs

        if not sum(big_deviators):
            continue
        # timestamptodelete = (
        #    deviation[big_deviators]
        #    .sort_values("value")
        #    .dropna()
        #    .tail()
        #    .index.astype(np.int64)[-1]
        # )
        # logger.info("Suggesting to delete from " + measurement + " " + str(spike))

        iloc = series.index.get_loc(spike.index[0])
        logger.info(series.iloc[iloc - 1 : iloc + 2])
        await delete_point(pers, measurement, spike.index[0], readonly=readonly)


async def delete_point(pers, measurement, timestamp, readonly=False):
    deletestatement = (
        "DELETE FROM  "
        + measurement
        + " WHERE time='"
        + str(timestamp).replace("+00:00", "")
        + "'"
    )
    logger.info(deletestatement)

    if not readonly:
        await pers.influxdb.dframe_query(deletestatement)


def filter_measurements(measurements):
    """Loop over a list and filter to only those
    measurements we want to check for spikes"""

    skiplist = [
        "Mjolfjell",
        "Kjoleskap",
        "akselerasjon",
        "lysstyrke",
        "CPU",
        "Bakside_lys",
        "fraluft",
        "avkast",
        "tilluft",
        "inntak",
    ]

    # Process the skiplist:
    measurements = [
        meas
        for meas in measurements
        if not any([skipstr in meas for skipstr in skiplist])
    ]

    req_startswith_list = [
        # "Sno",
        "Sensor_",
        "Solhoyde",
        "InneTemperatur",
        "Termostat_",
        "Netatmo_ute",
    ]
    # Filter to only those that starts with the above.
    measurements = [
        meas
        for meas in measurements
        if any([meas.startswith(req) for req in req_startswith_list])
    ]

    # Special casing for Termostat_*, filter to only Termostat_*Sensor*:
    measurements = [
        meas
        for meas in measurements
        if not meas.startswith("Termostat_")
        or meas.startswith("Termostat_")
        and "Sensor" in meas
    ]
    return measurements


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remove",
        action="store_true",
        help="If set, will send removal. If not, read-only-mode",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=48,
        help="How many hours to look back for, default=48",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(
        main(readonly=not args.remove, hours=args.hours)
    )
