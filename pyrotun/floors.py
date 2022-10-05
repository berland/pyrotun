import argparse
import asyncio
import datetime
import os
import random
from functools import partial
from pathlib import Path
from typing import Optional, Union

import dotenv
import networkx
import numpy as np
import pandas as pd
import pytz
import yaml
from matplotlib import pyplot

import pyrotun
from pyrotun import heatreservoir, persist  # noqa
from pyrotun.connections import localpowerprice

logger = pyrotun.getLogger(__name__)

# Positive number means colder house:
COLDER_FOR_POWERSAVING = 1


FLOORSFILE = "floors.yml"
# If the above file exists, it will overwrite the FLOORS variable.
FLOORS = {
    # This serves as an example:
    "Bad": {
        "sensor_item": "Termostat_Bad_SensorGulv",
        "setpoint_item": "Termostat_Bad_SetpointHeating",
        "setpoint_base": "temperature",  # or "target"
        "delta": 0,
        "heating_rate": 5,
        "cooling_rate": -0.1,
        "cooling_rate_winter": -0.6,
        "setpoint_force": 1,
        "wattage": 600,
        "maxtemp": 33,
        "backup_setpoint": 24,
    }
}

TIMEDELTA_MINUTES = 60  # minimum is 8 minutes!!
PD_TIMEDELTA = str(TIMEDELTA_MINUTES) + "min"
VACATION_ITEM = "Ferie"
WEATHER_COMPENSATION_ITEM = "Estimert_soloppvarming"
WEATHER_COMPENSATION_MULTIPLIER = 2
BACKUPSETPOINT = 22


def interpolate_summer_winter(
    day: Union[int, datetime.date], summer_value: float, winter_value: float
) -> float:
    if isinstance(day, datetime.date):
        day = (day - day.replace(month=1, day=1)).days
    assert 0 < day < 366
    interpolant = abs(day - 365 / 2) / (365 / 2)
    return (1 - interpolant) * summer_value + interpolant * winter_value


async def analyze_history(pers, selected_floors):
    daysago: int = 12 * 30
    skipbackwardsdays = 0 * 30
    minutegrouping = 10
    if (Path(__file__).parent / FLOORSFILE).exists():
        logger.info("Loading floor configuration from %s", FLOORSFILE)
        FLOORS = yaml.safe_load((Path(__file__).parent / FLOORSFILE).read_text())
    else:
        logger.warning("Did not find yml file with floor configuration, dummy run")
    for floor in selected_floors:
        logger.info("Analyzing heating and cooling rates for floor %s", floor)
        query = (
            f"SELECT difference(mean(value)) "
            f"FROM {FLOORS[floor]['sensor_item']} "
            f"where time > now() - {daysago}d "
            f"and time < now() - {skipbackwardsdays}d "
            f"group by time({minutegrouping}m) "
            "fill(linear) "
        )
        logger.info(query)
        resp = await pers.influxdb.dframe_query(query) * (60 / minutegrouping)

        # Crop away impossible derivatives:
        resp = resp[resp < 6]
        resp = resp[resp > -2]
        resp.hist(bins=30)
        pyplot.title(floor)
        pyplot.axvline(x=FLOORS[floor]["heating_rate"], color="red", linewidth=2)
        pyplot.axvline(x=0, color="black", linewidth=2)
        pyplot.axvline(x=FLOORS[floor]["cooling_rate"], color="red", linewidth=2)
        pyplot.axvline(
            x=FLOORS[floor]["cooling_rate_winter"], color="purple", linewidth=2
        )
        pyplot.show()


async def amain(
    pers=None,
    dryrun=False,
    plot=False,
    plotnodes=False,
    floors=None,
    freq="10min",
    vacation="auto",
    hoursago=0,
    minutesago=0,
    analyze=False,
):
    """Called from service or interactive"""
    assert hoursago >= 0
    assert minutesago >= 0

    floorsfile = Path(__file__).parent / FLOORSFILE
    if floorsfile.exists():
        logger.info("Loading floor configuration from %s", str(floorsfile))
        FLOORS = yaml.safe_load(floorsfile.read_text())
    else:
        # If pyrotun is installed into site_packags, floors.yml might
        # not have followed. Try looking relative to cwd (set by systemd)
        floorsfile = Path("pyrotun") / FLOORSFILE
        if floorsfile.exists():
            logger.info("Loading floor configuration from %s", str(floorsfile))
            FLOORS = yaml.safe_load(floorsfile.read_text())
        else:
            logger.error(
                f"Did not find yml file {str(floorsfile)} "
                "with floor configuration, dummy run"
            )

    closepers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["tibber", "influxdb", "openhab"])
        closepers = True

    if floors is None:
        selected_floors = FLOORS.keys()
    else:
        selected_floors = floors

    if hoursago > 0 or minutesago > 0:
        assert dryrun is True, "Only dryrun when jumping back in history"

    if analyze:
        await analyze_history(pers, selected_floors)
        if closepers:
            await pers.aclose()
        return

    prices_df = await pers.tibber.get_prices()

    # Grid rental is time dependent:
    prices_df = prices_df.copy()
    prices_df["NOK/KWh"] += localpowerprice.get_gridrental(prices_df.index)

    if not vacation or vacation == "auto":
        vacation = await pers.openhab.get_item(VACATION_ITEM, datatype=bool)
    else:
        if vacation.lower() == "on":
            vacation = True
            logger.info("Vacation forced to on")
        else:
            logger.info("Vacation forced to off")
            vacation = False

    # Summer mode; truncate maxtemp to two degrees above setpoint:
    if 4 < datetime.datetime.now().month < 10:
        for floor in FLOORS:
            # Bad coding: overwriting a "constant" variable.
            FLOORS[floor]["maxtemp"] = 25 + 2 + FLOORS[floor].get("delta", 0)

    for floor in selected_floors:
        logger.info("Starting optimization for floor %s", floor)
        if hoursago > 0:
            logger.info("* Jumping back %s hours", str(hoursago))
        if minutesago > 0:
            logger.info("* Jumping back %s minutes", str(minutesago))
        currenttemp = await pers.influxdb.get_item(
            FLOORS[floor]["sensor_item"],
            datatype=float,
            ago=hoursago * 60 + minutesago,
            unit="m",
        )
        logger.debug(f"Current temperature is {currenttemp}")
        if (
            currenttemp is None
            or str(currenttemp) == "UNDEF"
            or not (12 < currenttemp < 40)
        ):
            logger.error(
                "Currenttemp was %s, can't be right, giving up" % str(currenttemp)
            )
            # Backup temperature:
            if not dryrun:
                await pers.openhab.set_item(
                    FLOORS[floor]["setpoint_item"],
                    str(FLOORS[floor]["backup_setpoint"]),
                    log=True,
                )
            continue

        if "delta" in FLOORS[floor]:
            delta = FLOORS[floor]["delta"]
        else:
            delta = 0

        # Deduct more in to correct for good weather:
        weathercompensation = (
            await pers.openhab.get_item(WEATHER_COMPENSATION_ITEM, datatype=float)
            * WEATHER_COMPENSATION_MULTIPLIER
        )
        delta = delta - round(weathercompensation * 2.0) / 2.0 - COLDER_FOR_POWERSAVING

        if currenttemp > FLOORS[floor]["maxtemp"]:
            logger.info("Floor is above allowed maxtemp, turning OFF")
            if not dryrun:
                min_temp = temp_requirement(
                    datetime.datetime.now(), vacation=vacation, prices=None, delta=delta
                )
                await pers.openhab.set_item(
                    FLOORS[floor]["setpoint_item"],
                    str(max(10, min_temp - FLOORS[floor]["setpoint_force"])),
                    log=True,
                )
            continue

        tz = pytz.timezone(os.getenv("TIMEZONE"))
        starttime = (
            datetime.datetime.now()
            - datetime.timedelta(hours=hoursago)
            - datetime.timedelta(minutes=minutesago)
        ).astimezone(tz)

        if "cooling_rate_winter" in FLOORS[floor]:
            cooling_rate_interpolated = interpolate_summer_winter(
                datetime.datetime.today(),
                FLOORS[floor]["cooling_rate"],
                FLOORS[floor]["cooling_rate_winter"],
            )
        else:
            cooling_rate_interpolated = FLOORS[floor]["cooling_rate"]

        result = heatreservoir.optimize(
            starttime=starttime,
            starttemp=currenttemp,
            prices=prices_df["NOK/KWh"],
            min_temp=10,  # temp_requirement??
            max_temp=FLOORS[floor]["maxtemp"],
            temp_predictor=partial(
                floortemp_predictor,
                wattage=FLOORS[floor]["wattage"],
                heating_rate=FLOORS[floor]["heating_rate"],
                cooling_rate=cooling_rate_interpolated,
            ),
            freq=freq,
        )

        assert result
        if not graph:
            logger.warning(
                f"Temperature ({currenttemp}) below minimum, should force on"
            )
            if not dryrun:
                min_temp = temp_requirement(
                    datetime.datetime.now(), vacation=vacation, prices=None, delta=delta
                )
                await pers.openhab.set_item(
                    FLOORS[floor]["setpoint_item"],
                    str(min_temp + FLOORS[floor]["setpoint_force"]),
                    log=True,
                )
            continue

        if plotnodes:
            plot_graph(graph, ax=None, show=True)  # Plots all nodes in graph.

        logger.debug("Finding shortest path in the graph for %s", floor)
        opt_results = analyze_graph(
            graph, starttemp=currenttemp, endtemp=0, starttime=starttime
        )
        logger.info(
            f"Cost for floor {floor} is {opt_results['opt_cost']:.2f}, "
            f"KWh is {opt_results['kwh']:.2f}"
        )

        onoff = path_onoff(opt_results["opt_path"])
        thermostat_values = path_thermostat_values(opt_results["opt_path"])

        # Calculate new setpoint
        if FLOORS[floor]["setpoint_base"] == "temperature":
            setpoint_base = thermostat_values.values[0]
        elif FLOORS[floor]["setpoint_base"] == "target":
            setpoint_base = temp_requirement(
                starttime,
                vacation=vacation,
                prices=None,
                delta=delta,
            )
        else:
            logger.error("Wrong configuration for floor %s", floor)
            setpoint_base = 20

        if onoff[0]:
            setpoint = setpoint_base + FLOORS[floor]["setpoint_force"]
            logger.info("Turning floor %s ON now", floor)
        else:
            setpoint = setpoint_base - FLOORS[floor]["setpoint_force"]
            logger.info("Turning floor %s OFF now", floor)

        setpoint = max(11, int(setpoint))

        if not dryrun:
            await pers.openhab.set_item(
                FLOORS[floor]["setpoint_item"],
                str(setpoint),
                log=True,
            )
        try:
            first_on_timestamp = onoff[onoff == 1].head(1).index.values[0]
            logger.info(
                "Will turn floor %s on at %s",
                floor,
                np.datetime_as_string(first_on_timestamp, unit="m", timezone=tz),
            )
        except IndexError:
            logger.info(
                "Will not turn floor %s on in price-future",
                floor,
            )

        if plot:
            fig, ax = pyplot.subplots()
            plot_path(opt_results["opt_path"], ax=ax, show=False)
            pyplot.title(floor)

            # Yesterdays temperatures shifted forward by 24h:
            hist_temps = await pers.influxdb.get_series(
                FLOORS[floor]["sensor_item"],
                since=datetime.datetime.now() - datetime.timedelta(hours=48),
            )
            # Smoothen curve:
            hist_temps = hist_temps.resample("15min").mean().interpolate(method="time")
            hist_temps.index = hist_temps.index.tz_convert(
                "Europe/Oslo"
            ) + datetime.timedelta(hours=24)
            ax.plot(
                hist_temps.index,
                hist_temps[FLOORS[floor]["sensor_item"]],
                color="green",
                label="direct",
                alpha=0.7,
            )

            prices_df["mintemp"] = (
                prices_df.reset_index()["index"]
                .apply(
                    temp_requirement, vacation=vacation, prices=prices_df, delta=delta
                )
                .values
            )
            ax.step(
                prices_df.index,
                prices_df["mintemp"],
                where="post",
                color="blue",
                alpha=0.4,
            )

            # Prices on a secondary y-axis:
            ax2 = ax.twinx()
            ax2.step(prices_df.index, prices_df["NOK/KWh"], where="post", alpha=0.2)

            ax.set_xlim(left=prices_df.index.min(), right=prices_df.index.max())
            pyplot.show()

    if closepers:
        await pers.aclose()


def remove_this____is_now_in_heaterservoir___optimize_heatreservoir(
    starttime=None,
    starttemp=20,
    prices=None,  # pd.Series
    min_temp=None,  # pd.Series
    max_temp=None,  # pd.Series
    maxhours=36,
    temp_predictor=None,  # function handle
    freq="10min",  # pd.date_range frequency
    temp_resolution=10000,
):
    """Build a networkx Directed 2D ~lattice Graph, with
    datetime on the x-axis and temperatures on the y-axis.

    Edges from nodes determined by (time, temp) has an associated
    cost in NOK and energy need in kwh

    Returns a dict.
         result = {"now": True,
                   "next_on": datetime.datetime, if now is True, then this could be in the past.
                   "cost": 2.1,  # NOK
                   "kwh":  3.3 # KWh
                   "path": path # Cheapest path to lowest and latest temperature.
                   "on_in_minutes":  float
                   "off_in_Minutes": float
          }
    """
    # Get a series with prices at the datetimes we want to optimize at:
    pred_dframe = prediction_dframe(
        starttime, prices, min_temp, max_temp, freq, maxhours
    )

    logger.info("Building graph for future floor temperatures")

    # Build Graph, starting with current temperature
    graph = networkx.DiGraph()
    temps = {}  # timestamps are keys, values are lists of scaled integer temps.

    # Initial point/node:
    temps[pred_dframe.index[0]] = [int_temp(starttemp)]

    # We allow only constant timesteps:
    assert (
        len(set(pd.Series(pred_dframe.index).diff()[1:])) == 1
    ), "Only constant timestemps allowed"
    t_delta_hours = (
        float((pred_dframe.index[1] - pred_dframe.index[0]).value) / 1e9 / 60 / 60
    )  # convert from nanoseconds to hours.

    # Loop over time:
    for tstamp, next_tstamp in zip(pred_dframe.index, pred_dframe.index[1:]):
        # print(tstamp)
        temps[next_tstamp] = []

        min_temp = 19
        max_temp = 27
        # Loop over available temperature nodes until now:
        for temp in temps[tstamp]:
            # print(f"At {tstamp} with temp {temp}")
            # print(temp_predictor(float_temp(temp), tstamp, t_delta_hours))
            for pred in temp_predictor(float_temp(temp), tstamp, t_delta_hours):
                if min_temp < pred["temp"] < max_temp:
                    graph.add_edge(
                        (tstamp, temp),
                        (next_tstamp, int_temp(pred["temp"])),
                        cost=pred["kwh"] * pred_dframe.loc[tstamp, "NOK/KWh"],
                        kwh=pred["kwh"],
                    )
                    temps[next_tstamp].append(int_temp(pred["temp"]))
            # Collapse next temperatures according to resolution
            temps[next_tstamp] = list(set(temps[next_tstamp]))
            # print(f"Next temperatures are {temps[next_tstamp]}")

    # Build result dictionary:
    result = {
        "graph": graph,
        "path": cheapest_path(graph, pred_dframe.index[0]),
    }
    result["cost"] = path_costs(graph, result["path"])
    result["onoff"] = path_onoff(result["path"])
    if result["onoff"].max() > 0:
        result["on_at"] = result["onoff"][result["onoff"] == 1].index.values[0]
    else:
        result["on_at"] = None
    result["on_now"] = result["onoff"].values[0] == 1
    return result


def cheapest_path(graph, starttime):
    startnode = find_node(graph, starttime, 0)
    endnode = find_node(graph, starttime + pd.Timedelta(hours=72), 0)
    return networkx.shortest_path(
        graph, source=startnode, target=endnode, weight="cost"
    )


async def floor_specific_function_heatreservoir_temp_cost_graph(
    starttemp=60,
    prices_df=None,
    mintemp=10,
    maxtemp=35,
    wattage=1000,
    heating_rate=2,
    cooling_rate=0.1,
    vacation=False,
    maxhours=36,
    starttime=None,
    delta=0,
    freq="10min",
):
    """Build the networkx Directed 2D ~lattice  Graph, with
    datetime on the x-axis and water-temperatur on the y-axis.

    Edges from nodes determined by (time, temp) has an associated
    cost in NOK.
    """
    logger.info("Building graph for future floor temperatures")
    if starttime is None:
        tz = pytz.timezone(os.getenv("TIMEZONE"))
        starttime = datetime.datetime.now().astimezone(tz)
    starttime_wholehour = starttime.replace(minute=0, second=0, microsecond=0)
    datetimes = pd.date_range(
        starttime_wholehour,
        prices_df.index.max() + pd.Timedelta(1, unit="hour"),
        freq=freq,
        tz=prices_df.index.tz,
    )

    # Only timestamps after starttime is up for prediction:
    datetimes = datetimes[datetimes > starttime]

    # Delete timestamps mentioned in prices, for correct merging:
    duplicates = []
    for tstamp in datetimes:
        if tstamp in prices_df.index:
            duplicates.append(tstamp)
    datetimes = datetimes.drop(duplicates)
    # Merge prices into the requested datetime:
    dframe = pd.concat(
        [
            pd.DataFrame(index=pd.DatetimeIndex([starttime])),
            prices_df,
            pd.DataFrame(index=datetimes),
        ],
        axis="index",
    )
    dframe = dframe.sort_index()

    dframe = dframe[dframe.index < starttime + pd.Timedelta(maxhours, unit="hour")]
    dframe = dframe.ffill().bfill()
    dframe = dframe[dframe.index > starttime]
    logger.debug(dframe.head())

    # Yield..
    await asyncio.sleep(0.001)

    # Build Graph, starting with current temperature
    graph = networkx.DiGraph()

    # Temperatures in the graph are always integers, and for that reason scaled up.

    # dict of timestamp to list of reachable temperatures:
    temps = {}
    temps[dframe.index[0]] = [starttemp]
    first_tstamp = dframe.index[0]
    logger.debug(f"First timestamp in graph is {first_tstamp}")
    # Loop over all datetimes, and inject nodes and possible edges
    for tstamp, next_tstamp in zip(dframe.index, dframe.index[1:]):
        # Yield..
        await asyncio.sleep(0.001)
        temps[next_tstamp] = []

        # Collapse similar temperatures
        temps[tstamp] = list(set(temps[tstamp]))

        powerprice = dframe.loc[tstamp]["NOK/KWh"]
        t_delta = next_tstamp - tstamp
        t_delta_hours = float(t_delta.value) / 1e9 / 60 / 60  # from nanoseconds

        logger.debug(
            " (graph building at %s, %d temps)", str(tstamp), len(temps[tstamp])
        )
        for temp in temps[tstamp]:

            # This is Explicit Euler solution of the underlying
            # differential equation, predicting future temperature:

            #  future_temps_callback(temp, t_delta_hours, powerprice,
            #           wattage, cooling_rate, heating_rate)
            # Returns list of dict:
            # [  {"temp": 23.1, "cost": 0, "kwh": 0},
            #    {"temp": 25.1, "cost": 2, "kwh": 1}]
            # This function can also be called for a list of future temperatures,
            # then it will compute the cost of going there.

            assert cooling_rate < 0
            no_heater_temp = float_temp(int_temp(temp + cooling_rate * t_delta_hours))
            min_temp = max(
                temp_requirement(
                    tstamp, vacation=vacation, prices=prices_df, delta=delta
                ),
                mintemp,
            )
            if no_heater_temp > min_temp:
                # Add an edge for the no-heater-scenario:
                temps[next_tstamp].append(no_heater_temp)
                graph.add_edge(
                    (tstamp, int_temp(temp)),
                    (next_tstamp, int_temp(no_heater_temp)),
                    cost=0,
                    kwh=0,
                    tempdeviation=abs(no_heater_temp - starttemp),
                )
                if tstamp == first_tstamp:
                    logger.debug(f"Adding edge for no-heater to temp {no_heater_temp}")
            heater_on_temp = float_temp(int_temp(temp + heating_rate * t_delta_hours))
            if min_temp < heater_on_temp < maxtemp:
                kwh = wattage / 1000 * t_delta_hours
                cost = kwh * powerprice + hightemp_penalty(
                    heater_on_temp, powerprice, t_delta_hours
                )  # Unit NOK

                # Add edge for heater-on:
                temps[next_tstamp].append(heater_on_temp)
                graph.add_edge(
                    (tstamp, int_temp(temp)),
                    (next_tstamp, int_temp(heater_on_temp)),
                    cost=cost,
                    kwh=kwh,
                    tempdeviation=abs(heater_on_temp - starttemp),
                )
                if tstamp == first_tstamp:
                    logger.debug(
                        f"Adding edge for heater-on at cost {cost} to "
                        f"temp {heater_on_temp}"
                    )

        # Yield..
        await asyncio.sleep(0.001)
        # For every temperature node, we need to link up with other nodes
        # between its corresponding no-heater and heater-temp, as we
        # should regard these as "reachable" for algorithm stability.
        for temp in temps[tstamp]:
            no_heater_temp = float_temp(int_temp(temp + cooling_rate * t_delta_hours))
            heater_on_temp = float_temp(int_temp(temp + heating_rate * t_delta_hours))

            inter_temps = [
                t for t in temps[next_tstamp] if no_heater_temp < t < heater_on_temp
            ]
            if not inter_temps:
                continue
            if len(inter_temps) > 7:
                # When we are "far" out in the graph, we don't need
                # every possible choice. It is the first three or
                # four steps that matter for stability
                inter_temps = random.sample(inter_temps, 7)
            full_kwh = wattage / 1000 * t_delta_hours
            for inter_temp in inter_temps:
                rel_temp_inc = (inter_temp - no_heater_temp) / (
                    heater_on_temp - no_heater_temp
                )
                assert 0 < rel_temp_inc < 1
                rel_kwh = rel_temp_inc * full_kwh
                cost = rel_kwh * powerprice + hightemp_penalty(
                    inter_temp, powerprice, t_delta_hours
                )
                # logger.info(
                #    f"Adding extra edge {temp} to {inter_temp} at cost {cost}, "
                #    "full cost is {full_kwh*powerprice}"
                # )
                graph.add_edge(
                    (tstamp, int_temp(temp)),
                    (next_tstamp, int_temp(inter_temp)),
                    cost=cost,
                    kwh=kwh,
                    tempdeviation=abs(inter_temp - temp),
                )

    logger.info(
        f"Built graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
    )

    # If we currently below accepted temperatures, graph is empty
    if not graph:
        return graph

    # After graph is built, add an extra (dummy) node (at zero cost) collecting all
    # nodes. This is necessary as we don't know exactly which end-node we should
    # find the path to, because 23.34 degrees at endpoint might be cheaper to
    # go to than 23.31, due to the discrete nature of the optimization.
    min_temp = min(temps[next_tstamp])  # Make this the endpoint node
    for temp in temps[next_tstamp]:
        graph.add_edge(
            (next_tstamp, temp),
            (next_tstamp + t_delta, min_temp),
            cost=0,
            kwh=0,
            tempdeviation=0,
        )

    return graph


def hightemp_penalty(temp, powerprice, t_delta_hours):
    # Add cost of 160W pr degree, spread on all heaters, say 10, so 16 watt pr degree.
    # Positive values for all values > 15 degrees.

    # Multiply the extra watts by this, as to some degree we are eroding
    # the job to be done by the heat pump which has a high COP
    extra_factor = 5

    # Important to not return a positive number to have a stable algoritm.
    overshoot_temp = max(temp - 15, 0.001)
    extra_kwh = overshoot_temp * 16 * extra_factor / 1000
    return powerprice * extra_kwh * t_delta_hours


def floortemp_predictor(
    temp, tstamp, t_delta_hours, wattage=0, heating_rate=0, cooling_rate=0
):
    return [
        {"temp": temp + heating_rate * t_delta_hours, "kwh": wattage * t_delta_hours},
        {"temp": temp - cooling_rate * t_delta_hours, "kwh": 0},
    ]


def temp_requirement(
    timestamp: pd.Timestamp,
    vacation: bool = False,
    master_correction: Optional[pd.Series] = None,
    delta: float = 0,
) -> float:
    """
    Args:
        timestamp
        vacation
        master_correction: A time-dependent correction added
            to the master temperature
        delta: Room-dependent (constant in time) correction added
            to master temperature. Positive value means warmer.
    """
    hour = timestamp.hour
    weekday = timestamp.weekday()  # Monday = 0, Sunday = 6
    friday = 4
    corona = False
    if vacation:
        # Ferie
        return 15 + delta
    if hour < 6 or hour > 21:
        # Natt:
        return 18 + delta
    if corona is False and (hour > 7 and hour < 13 and weekday <= friday):
        # Dagsenking:
        return 18 + delta
    if hour > 16 and hour < 22:
        # Ettermiddag
        return 22 + delta
    # Morgen og middagstid, 25 er egentlig komfort her!!
    return 23 + delta


def path_onoff(path):
    """Compute a pandas series with 1 or 0 whether the heater
    should be on or off along a path (computed assuming
    that if temperature increases, heater must be on)"""
    timestamps = [node[0] for node in path][:-1]  # skip the last one
    temps = [node[1] for node in path]
    onoff = pd.Series(temps).diff().shift(-1).dropna()
    onoff.index = timestamps
    return np.maximum(0, np.sign(onoff))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set", action="store_true", help="Actually submit setpoint to OpenHAB"
    )
    parser.add_argument("--plot", action="store_true", help="Make plots")
    parser.add_argument("--plotnodes", action="store_true", help="Make node plots")
    parser.add_argument("--floors", nargs="*", help="Floornames")
    parser.add_argument(
        "--freq", type=str, help="Time frequency, default 10min", default="10min"
    )
    parser.add_argument("--vacation", type=str, help="ON or OFF or auto", default="")
    parser.add_argument("--analyze", action="store_true", help="Analyze mode")
    parser.add_argument("--hoursago", type=int, help="Step back some hours", default=0)
    parser.add_argument(
        "--minutesago", type=int, help="Step back some minutes", default=0
    )

    return parser


if __name__ == "__main__":
    # Interactive testing:
    dotenv.load_dotenv()

    parser = get_parser()
    args = parser.parse_args()

    asyncio.run(
        amain(
            pers=None,
            dryrun=not args.set,
            plot=args.plot,
            plotnodes=args.plotnodes,
            floors=args.floors,
            freq=args.freq,
            vacation=args.vacation,
            hoursago=args.hoursago,
            minutesago=args.minutesago,
            analyze=args.analyze,
        )
    )
