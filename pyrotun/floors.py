import os
import asyncio
import networkx
import datetime
import itertools

import argparse
import pytz
import numpy as np
import pandas as pd
from matplotlib import pyplot
import dotenv

import pyrotun

from pyrotun import persist  # noqa

logger = pyrotun.getLogger(__name__)

FLOORS = {
    "Andre": {
        "sensor_item": "Sensor_Andretak_temperatur",  # "Termostat_Andre_SensorGulv",
        "setpoint_item": "Termostat_Andre_SetpointHeating",
        "setpoint_base": "target",  # means not relative to current temp
        "delta": -1,
        "heating_rate": 0.9,  # degrees/hour
        "cooling_rate": -0.6,  # degrees/hour
        "setpoint_force": 8,
        "wattage": 600,
        "maxtemp": 27,
        "backup_setpoint": 20,
    },
    "Leane": {
        "sensor_item": "Sensor_Leanetak_temperatur",
        "setpoint_item": "Termostat_Leane_SetpointHeating",
        "delta": -1,
        "setpoint_base": "target",
        "heating_rate": 1,  # degrees/hour
        "cooling_rate": -0.55,  # degrees/hour
        "setpoint_force": 8,
        "wattage": 600,
        "maxtemp": 27,
        "backup_setpoint": 20,
    },
    "Bad_Oppe": {
        "sensor_item": "Termostat_Bad_Oppe_SensorGulv",
        "setpoint_item": "Termostat_Bad_Oppe_SetpointHeating",
        "setpoint_base": "temperature",
        "heating_rate": 5,
        "cooling_rate": -0.4,
        "setpoint_force": 1,
        "wattage": 600,
        "maxtemp": 31,
        "backup_setpoint": 24,
    },
    "Bad_Kjeller": {
        "sensor_item": "Termostat_Bad_Kjeller_SensorGulv",
        "setpoint_item": "Termostat_Bad_Kjeller_SetpointHeating",
        "setpoint_base": "temperature",
        "heating_rate": 5,
        "cooling_rate": -0.3,
        "setpoint_force": 1,
        "wattage": 1000,
        "maxtemp": 31,
        "backup_setpoint": 24,
    },
    "Sofastue": {
        "sensor_item": "Termostat_Sofastue_SensorGulv",
        "setpoint_item": "Termostat_Sofastue_SetpointHeating",
        "setpoint_base": "temperature",
        "delta": -2,
        "heating_rate": 1,
        "cooling_rate": -0.3,
        "setpoint_force": 2,
        "wattage": 1700,
        "maxtemp": 27,
        "backup_setpoint": 20,
    },
    "Tvstue": {
        "sensor_item": "Termostat_Tvstue_SensorGulv",
        "setpoint_item": "Termostat_Tvstue_SetpointHeating",
        "delta": -2,  # relative to master-temp at 25, adapt to sensor and wish.
        "setpoint_base": "temperature",
        "heating_rate": 1,
        "cooling_rate": -0.3,
        "setpoint_force": 3,
        "wattage": 600,
        "maxtemp": 27,
        "backup_setpoint": 20,
    },
    "Gangoppe": {
        "sensor_item": "Termostat_Gangoppe_SensorGulv",
        "setpoint_item": "Termostat_Gangoppe_SetpointHeating",
        "delta": -5,  # relative to master-temp at 25, adapt to sensor and wish.
        "setpoint_base": "temperature",
        "heating_rate": 0.4,
        "cooling_rate": -0.2,
        "setpoint_force": 3,
        "wattage": 600,
        "maxtemp": 27,
        "backup_setpoint": 20,
    },
    "Bakgang": {
        "sensor_item": "Termostat_Bakgang_SensorGulv",
        "setpoint_item": "Termostat_Bakgang_SetpointHeating",
        "delta": -5,  # relative to master-temp at 25, adapt to sensor and wish.
        "setpoint_base": "temperature",
        "heating_rate": 2,
        "cooling_rate": -0.1,
        "setpoint_force": 2,
        "wattage": 800,
        "maxtemp": 30,
        "backup_setpoint": 15,
    },
    "Vaskegang": {
        "sensor_item": "Termostat_Vaskegang_SensorTemperature",
        "setpoint_item": "Termostat_Vaskegang_SetpointHeating",
        "delta": -3,  # relative to master-temp at 25, adapt to sensor and wish.
        "setpoint_base": "temperature",
        "heating_rate": 3,
        "cooling_rate": -0.2,
        "setpoint_force": 2,
        "wattage": 800,
        "maxtemp": 30,
        "backup_setpoint": 15,
    },
    "Inngang": {
        "sensor_item": "Termostat_Inngang_SensorGulv",
        "setpoint_item": [
            "Termostat_Inngang_SetpointHeating",
            "Termostat_Langgang_Nede_SetpointHeating",
        ],
        "delta": 0,  # relative to master-temp at 25, adapt to sensor and wish.
        "setpoint_base": "temperature",
        "heating_rate": 0.4,
        "cooling_rate": -0.6,
        "setpoint_force": 5,
        "wattage": 600,
        "maxtemp": 32,
        "backup_setpoint": 23,
    },
}
TIMEDELTA_MINUTES = 10  # minimum is 8 minutes!!
ROUND = 1
PD_TIMEDELTA = str(TIMEDELTA_MINUTES) + "min"
VACATION_ITEM = "Ferie"

BACKUPSETPOINT = 22


async def main(
    pers=None,
    dryrun=False,
    plot=False,
    floors=None,
    freq="10min",
    vacation="auto",
    hoursago=0,
    analyze=False,
):
    """Called from service or interactive"""
    assert hoursago >= 0

    closepers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["tibber", "influxdb", "openhab"])
        closepers = True

    if floors is None:
        selected_floors = FLOORS.keys()
    else:
        selected_floors = floors

    if analyze:
        daysago = 30
        minutegrouping = 10
        for floor in selected_floors:
            query = (
                f"SELECT difference(mean(value)) "
                f"FROM {FLOORS[floor]['sensor_item']} "
                f"where time > now() - {daysago}d "
                f"group by time({minutegrouping}m)"
            )
            logger.info(query)
            resp = await pers.influxdb.dframe_query(query) * (60 / minutegrouping)

            # Crop away impossible derivatives:
            resp = resp[resp < 6]
            resp = resp[resp > -2]
            resp.hist(bins=30)
            pyplot.title(floor)
            pyplot.show()
        return

    if hoursago > 0:
        assert dryrun is True, "Only dryrun when jumping back in history"

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

    for floor in selected_floors:
        logger.info("Starting optimization for floor %s", floor)
        if hoursago > 0:
            logger.info("* Jumping back %s hours", str(hoursago))
        currenttemp = await pers.influxdb.get_item(
            FLOORS[floor]["sensor_item"], datatype=float, ago=hoursago, unit="h"
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

        starttime = datetime.datetime.now() - datetime.timedelta(hours=hoursago)
        tz = pytz.timezone(os.getenv("TIMEZONE"))
        starttime = starttime.astimezone(tz)
        graph = heatreservoir_temp_cost_graph(
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
                    FLOORS[floor]["setpoint_item"], str(min_temp), log=True
                )
            continue

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
        else:
            setpoint = setpoint_base - FLOORS[floor]["setpoint_force"]

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
                # first_on_timestamp,
                np.datetime_as_string(first_on_timestamp, unit="m", timezone=tz)
                # pd.Timestamp(first_on_timestamp).tz_localize(tz),
            )
        except IndexError:
            logger.info(
                "Will not turn floor %s on in price-future",
                floor,
            )

        if plot:
            fig, ax = pyplot.subplots()
            # plot_graph(graph, ax=ax, show=True)  # Plots all nodes in graph.
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
    return int(temp * 100)


def float_temp(int_temp):
    return float(int_temp / 100.0)


def heatreservoir_temp_cost_graph(
    starttemp=60,
    prices_df=None,
    mintemp=10,
    maxtemp=35,
    wattage=1000,
    heating_rate=2,
    cooling_rate=0.1,
    vacation=False,
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
    dframe = dframe.ffill().bfill()
    dframe = dframe[dframe.index >= starttime]

    # Build Graph, starting with current temperature
    graph = networkx.DiGraph()

    # Temperatures in the graph is always integers, and multiplied up!

    # dict of timestamp to list of reachable temperatures:
    temps = {}
    temps[dframe.index[0]] = [starttemp]
    # Loop over all datetimes, and inject nodes and possible edges
    for tstamp, next_tstamp in zip(dframe.index, dframe.index[1:]):
        temps[next_tstamp] = []
        temps[tstamp] = list(set(temps[tstamp]))
        temps[tstamp].sort()
        powerprice = dframe.loc[tstamp]["NOK/KWh"]
        t_delta = next_tstamp - tstamp
        t_delta_hours = float(t_delta.value) / 1e9 / 60 / 60  # from nanoseconds
        for temp in temps[tstamp]:
            # This is Explicit Euler solution of the underlying
            # differential equation, predicting future temperature:
            assert cooling_rate < 0
            no_heater_temp = temp + cooling_rate * t_delta_hours
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
            heater_on_temp = round(
                temp + heating_rate * t_delta_hours,
                ROUND,
            )
            if min_temp < heater_on_temp < maxtemp:
                kwh = wattage / 1000 * t_delta_hours
                hightemp_penalty = (heater_on_temp - 15) / 20 / 100
                hightemp_penalty = 1
                # Small penalty for high temps:
                cost = kwh * hightemp_penalty * powerprice

                # Add edge for heater-on:
                temps[next_tstamp].append(heater_on_temp)
                graph.add_edge(
                    (tstamp, int_temp(temp)),
                    (next_tstamp, int_temp(heater_on_temp)),
                    cost=cost,
                    kwh=kwh,
                    tempdeviation=abs(heater_on_temp - starttemp),
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


def plot_graph(graph, ax=None, show=False):
    if ax is None:
        fig, ax = pyplot.subplots()

    if len(graph.edges) < 1000:
        logger.info("Plotting all graph edges, wait for it..")
        for edge_0, edge_1, data in graph.edges(data=True):
            pd.DataFrame(
                [
                    {"index": edge_0[0], "temp": edge_0[1]},
                    {"index": edge_1[0], "temp": edge_1[1]},
                ]
            ).plot(x="index", y="temp", ax=ax, legend=False)
    nodes_df = pd.DataFrame(data=graph.nodes, columns=["index", "temp"])

    logger.info("Plotting all graph nodes..")
    nodes_df.plot.scatter(x="index", y="temp", ax=ax)

    if show:
        pyplot.show()


def plot_path(path, ax=None, show=False, linewidth=2, color="red"):
    if ax is None:
        fig, ax = pyplot.subplots()

    path_dframe = pd.DataFrame(path, columns=["index", "temp"]).set_index("index")
    path_dframe["temp"] = [float_temp(temp) for temp in path_dframe["temp"]]

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


def temp_requirement(timestamp, vacation=False, prices=None, delta=0):
    hour = timestamp.hour

    if vacation:
        return 15 + delta
    if hour < 6 or hour > 22:
        return 18 + delta
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


def shortest_paths(graph, k=5, starttemp=60, endtemp=60, now=datetime.datetime.now()):
    """Return the k shortest paths. Runtime is K*N**3, too much
    for practical usage"""
    startnode = find_node(graph, now - pd.Timedelta("1h"), int_temp(starttemp))
    endnode = find_node(graph, now + pd.Timedelta("48h"), int_temp(endtemp))
    # Path generator:
    return list(
        itertools.islice(
            networkx.shortest_simple_paths(
                graph, source=startnode, target=endnode, weight="cost"
            ),
            k,
        )
    )


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
    no_opt_path = networkx.shortest_path(
        graph, source=startnode, target=endnode, weight="tempdeviation"
    )
    opt_cost = sum(path_costs(graph, path))
    no_opt_cost = sum(path_costs(graph, no_opt_path))
    kwh = sum(path_kwh(graph, path))
    savings = no_opt_cost - opt_cost
    timespan = (endnode[0] - startnode[0]).value / 1e9 / 60 / 60  # from nanoseconds
    savings24h = savings / float(timespan) * 24.0
    return {
        "opt_cost": opt_cost,
        "no_opt_cost": no_opt_cost,
        "savings24h": savings24h,
        "kwh": kwh,
        "opt_path": path,
        "no_opt_path": no_opt_path,
    }


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set", action="store_true", help="Actually submit setpoint to OpenHAB"
    )
    parser.add_argument("--plot", action="store_true", help="Make plots")
    parser.add_argument("--floors", nargs="*", help="Floornames")
    parser.add_argument(
        "--freq", type=str, help="Time frequency, default 10min", default="10min"
    )
    parser.add_argument("--vacation", type=str, help="ON or OFF or auto", default="")
    parser.add_argument("--analyze", action="store_true", help="Analyze mode")
    parser.add_argument("--hoursago", type=int, help="Step back some hours", default=0)

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
            floors=args.floors,
            freq=args.freq,
            vacation=args.vacation,
            hoursago=args.hoursago,
            analyze=args.analyze,
        )
    )
