import os
import asyncio
import networkx
import datetime
import random
import yaml
from pathlib import Path

import argparse
import pytz
import numpy as np
import pandas as pd
from matplotlib import pyplot
import dotenv

import pyrotun

from pyrotun import persist  # noqa

logger = pyrotun.getLogger(__name__)

TEMPERATURE_RESOLUTION = 10000
"""If the temperature resolution is too low, it will make the decisions
unstable for short timespans. It is tempting to keep it low to allow
some sort of graph collapse."""


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
        "cooling_rate": -0.4,
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
BACKUPSETPOINT = 22


async def analyze_history(pers, selected_floors):
    daysago = 7
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
        pyplot.show()


async def main(
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
        weathercompensation = await pers.openhab.get_item(
            WEATHER_COMPENSATION_ITEM, datatype=float
        )
        delta = delta - round(weathercompensation * 2.0) / 2.0

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

        graph = await heatreservoir_temp_cost_graph(
            starttime=starttime,
            starttemp=currenttemp,
            prices_df=prices_df,
            mintemp=10,
            maxtemp=FLOORS[floor]["maxtemp"],
            wattage=FLOORS[floor]["wattage"],
            heating_rate=FLOORS[floor]["heating_rate"],
            cooling_rate=FLOORS[floor]["cooling_rate"],
            vacation=vacation,
            freq=freq,
            delta=delta,
        )

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
            prices_df["mintemp"] = (
                prices_df.reset_index()["index"]
                .apply(
                    temp_requirement, vacation=vacation, prices=prices_df, delta=delta
                )
                .values
            )
            prices_df.plot(
                drawstyle="steps-post", ax=ax, y="mintemp", color="blue", alpha=0.4
            )

            # Prices on a secondary y-axis:
            ax2 = ax.twinx()
            prices_df.plot(drawstyle="steps-post", y="NOK/KWh", ax=ax2, alpha=0.2)
            pyplot.show()

    if closepers:
        await pers.aclose()


def int_temp(temp):
    return int(temp * TEMPERATURE_RESOLUTION)


def float_temp(int_temp):
    return float(int_temp / float(TEMPERATURE_RESOLUTION))


async def heatreservoir_temp_cost_graph(
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

    # Temperatures in the graph is always integers, and multiplied up!

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
            # then it will compute andthe cost of going there.

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
            (next_tstamp, int_temp(temp)),
            (next_tstamp + t_delta, int_temp(min_temp)),
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


def plot_graph(graph, ax=None, show=False):
    if ax is None:
        fig, ax = pyplot.subplots()

    logger.info("Plotting some graph edges, wait for it..")
    counter = 0
    maxnodes = 400
    for edge_0, edge_1, data in graph.edges(data=True):
        counter += 1
        pd.DataFrame(
            [
                {"index": edge_0[0], "temp": edge_0[1]},
                {"index": edge_1[0], "temp": edge_1[1]},
            ]
        ).plot(x="index", y="temp", ax=ax, legend=False)
        mid_time = edge_0[0] + (edge_1[0] - edge_0[0]) / 2
        pyplot.text(mid_time, (edge_0[1] + edge_1[1]) / 2, str(round(data["cost"], 5)))
        if counter > maxnodes:
            break
    nodes_df = pd.DataFrame(data=graph.nodes, columns=["index", "temp"]).head(maxnodes)

    logger.info("Plotting all graph nodes..")
    nodes_df.plot.scatter(x="index", y="temp", ax=ax)

    if show:
        pyplot.show()


def plot_path(path, ax=None, show=False, linewidth=2, color="red"):
    if ax is None:
        fig, ax = pyplot.subplots()

    path_dframe = pd.DataFrame(path, columns=["index", "temp"]).set_index("index")
    path_dframe["temp"] = [float_temp(temp) for temp in path_dframe["temp"]]

    # pyplot.plot(
    #    path_dframe.index[0],
    #    path_dframe["temp"].values[0],
    #    marker="o",
    #    markersize=3,
    #    color="red",
    # )

    # BUG: Circumvent unresolved plotting bug that messes up index in plot:
    path_dframe = path_dframe.iloc[1:]

    path_dframe.plot(y="temp", linewidth=linewidth, color=color, ax=ax)
    if show:
        pyplot.show()


def find_node(graph, when, temp):
    nodes_df = pd.DataFrame(data=graph.nodes, columns=["index", "temp"])
    when = pd.Timestamp(when.astimezone(nodes_df["index"].dt.tz))
    closest_time_idx = (
        nodes_df.iloc[(nodes_df["index"] - when).abs().argsort()]
        .head(1)
        .index.values[0]
    )
    temp_df = nodes_df[nodes_df["index"] == nodes_df.iloc[closest_time_idx]["index"]]
    closest_temp_idx = (
        temp_df.iloc[(temp_df["temp"] - temp).abs().argsort()].head(1).index.values[0]
    )
    row = nodes_df.iloc[closest_temp_idx]
    return (row["index"], row["temp"])


def temp_requirement(
    timestamp, vacation=False, prices=None, master_correction=None, delta=0
):
    """
    Args:
        timestamp (pd.Timestamp)
        vacation (bool)
        prices (pd.DataFrame): not used
        master_correction (pd.Series): A time-dependent correction added
            to the master temperature
        delta (float): Room-dependent (constant in time) correction added
            to master temperature

    Return:
        float
    """
    hour = timestamp.hour
    weekday = timestamp.weekday()  # Monday = 0, Sunday = 6
    friday = 4
    if vacation:
        # Ferie
        return 15 + delta
    if hour < 6 or hour > 21:
        # Natt:
        return 18 + delta
    if hour > 7 and hour < 13 and weekday <= friday:
        # Dagsenking:
        return 18 + delta
    if hour > 16 and hour < 22:
        # Ettermiddag
        return 23 + delta
    # Morgen og middagstid
    return 25 + delta


def path_costs(graph, path):
    """Compute a list of the cost along a temperature path, in NOK"""
    return [graph.edges[path[i], path[i + 1]]["cost"] for i in range(len(path) - 1)]


def path_onoff(path):
    """Compute a pandas series with 1 or 0 whether the heater
    should be on or off along a path (computed assuming
    that if temperature increases, heater must be on)"""
    timestamps = [node[0] for node in path][:-1]  # skip the last one
    temps = [node[1] for node in path]
    onoff = pd.Series(temps).diff().shift(-1).dropna()
    onoff.index = timestamps
    return np.maximum(0, np.sign(onoff))


def path_thermostat_values(path):
    """Extract a Pandas series of thermostat values (integers) from a
    path with temperatures"""
    timestamps = [node[0] for node in path][:-1]  # skip the last one
    tempvalues = [float_temp(node[1]) for node in path]
    # Perturb temperatures, +1 when it should be on, and -1 when off:
    onoff = pd.Series(tempvalues).diff().shift(-1).dropna().apply(np.sign)
    onoff.index = timestamps
    inttemp_s = pd.Series(tempvalues).astype(int)
    inttemp_s.index = timestamps + [np.nan]
    return (inttemp_s + onoff).dropna()


def path_kwh(graph, path):
    return [graph.edges[path[i], path[i + 1]]["kwh"] for i in range(len(path) - 1)]


def analyze_graph(graph, starttemp=60, endtemp=60, starttime=None):
    """Find shortest path, and do some extra calculations for estimating
    savings. The savings must be interpreted carefully, and is
    probably only correct if start and endtemp is equal"""

    if starttime is None:
        starttime = datetime.datetime.now()

    startnode = find_node(graph, starttime, starttemp)
    logger.debug(f"startnode is {startnode}")
    endnode = find_node(graph, starttime + pd.Timedelta("48h"), endtemp)
    logger.debug(f"endnode is {endnode}")
    path = networkx.shortest_path(
        graph, source=startnode, target=endnode, weight="cost"
    )
    opt_cost = sum(path_costs(graph, path))
    kwh = sum(path_kwh(graph, path))
    # timespan = (endnode[0] - startnode[0]).value / 1e9 / 60 / 60  # from nanoseconds
    return {
        "opt_cost": opt_cost,
        "kwh": kwh,
        "opt_path": path,
    }


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
        main(
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
