import asyncio
import dotenv

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)


async def main(pers=None, readonly=True):
    if pers is None:
        dotenv.load_dotenv()
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["influxdb"])

    await remove_spikes(pers, mindev=7, stddevs=3, readonly=readonly)


async def remove_spikes(pers, mindev=7, stddevs=3, readonly=True):
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
            f"SELECT * FROM {measurement} where time > now() - 24h"
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
        deletestatement = (
            "Delete from "
            + measurement
            + " where time='"
            + str(spike.index[0]).replace("+00:00", "")
            + "'"
        )
        logger.info(deletestatement)
        iloc = series.index.get_loc(spike.index[0])
        logger.info(series.iloc[iloc - 1 : iloc + 2])

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

    measurements = [
        meas
        for meas in measurements
        if not any([skipstr in meas for skipstr in skiplist])
    ]

    req_startswith_list = [
        "Sensor_",
        "Solhoyde",
        "InneTemperatur",
        "Termostat_",
        "Netatmo_ute",
    ]

    measurements = [
        meas
        for meas in measurements
        if any([meas.startswith(req) for req in req_startswith_list])
    ]

    measurements = [
        meas
        for meas in measurements
        if not meas.startswith("Termostat_")
        or meas.startswith("Termostat_")
        and meas.endswith("SensorTemperature")
    ]
    return measurements


if __name__ == "__main__":

    asyncio.get_event_loop().run_until_complete(main())
