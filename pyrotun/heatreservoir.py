import argparse
import asyncio
import datetime

import dotenv
import networkx
import numpy as np
import pandas as pd
import pytz
from matplotlib import pyplot

import pyrotun
from pyrotun import persist  # noqa

logger = pyrotun.getLogger(__name__)

# Celsius temperatures are multiplied with this number and then
# made into an int:
TEMPERATURE_RESOLUTION = 100
# If the temperature resolution is too low, it will make the decisions
# unstable for short timespans. It is tempting to keep it low to allow
# numerically nearby nodes to collapse.


def prediction_dframe(
    starttime, prices, min_temp=None, max_temp=None, freq="10min", maxhours=36
):
    """Spread prices on a time-series of the requested frequency into
    the future"""
    starttime_wholehour = starttime.replace(minute=0, second=0, microsecond=0)
    datetimes = pd.date_range(
        starttime_wholehour,
        prices.index.max() + pd.Timedelta(1, unit="hour"),
        freq=freq,
        tz=prices.index.tz,
    )

    # Only timestamps after starttime is up for prediction:
    datetimes = datetimes[datetimes >= starttime - pd.Timedelta(freq)]
    print(datetimes)
    # Delete timestamps mentioned in prices, for correct merging:
    duplicates = []
    for tstamp in datetimes:
        if tstamp in prices.index:
            duplicates.append(tstamp)
    print(f"Duplicates removed: {duplicates}")
    datetimes = datetimes.drop(duplicates)

    # Merge prices into the requested datetime:
    dframe = pd.concat(
        [
            prices,
            pd.DataFrame(index=datetimes),
        ],
        axis="index",
    )
    dframe.columns = ["NOK/KWh"]
    dframe = dframe.sort_index()
    if min_temp is not None:
        dframe["min_temp"] = min_temp
    if max_temp is not None:
        dframe["max_temp"] = max_temp
    dframe = dframe[dframe.index < starttime + pd.Timedelta(maxhours, unit="hour")]
    # Constant extrapolation of prices:
    dframe = dframe.ffill().bfill()
    print(dframe)
    # Is is a rounding error we solve by the one second delta?
    dframe = dframe[dframe.index > starttime - pd.Timedelta(freq)]
    logger.debug("prediction_dframe")
    print(dframe.head())
    return dframe


def optimize(
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
         result = {"on_now": True,
                   "on_at": datetime.datetime
                   "cost": 2.1,  # NOK
                   "kwh":  3.3 # KWh
                   "path": path # Cheapest path to lowest and latest temperature.
                   "graph":
                   "on_in_minutes":  float
                   "off_in_Minutes": float
          }
    """
    # Get a series with prices at the datetimes we want to optimize at:
    pred_dframe = prediction_dframe(
        starttime, prices, min_temp, max_temp, freq, maxhours
    )

    logger.info("Building graph for future temperatures")
    # print(pred_dframe)

    # Build Graph, starting with current temperature
    graph = networkx.DiGraph()
    temps = {}  # timestamps are keys, values are lists of scaled integer temps.

    # Initial point/node:
    temps[pred_dframe.index[0]] = [int_temp(starttemp)]

    # We allow only constant timesteps:
    timesteps = pd.Series(pred_dframe.index).diff()[1:]
    # print(pred_dframe.index)
    assert (
        len(set(timesteps)) == 1
    ), f"Only constant timestemps allowed, got {set(timesteps)}"
    t_delta_hours = (
        float((pred_dframe.index[1] - pred_dframe.index[0]).value) / 1e9 / 60 / 60
    )  # convert from nanoseconds to hours.

    # Loop over time:
    for tstamp, next_tstamp in zip(pred_dframe.index, pred_dframe.index[1:]):
        # print(tstamp)
        next_temps_raw = []
        # min_temp
        # max_temp = 27  # ???? FIXME
        # Loop over available temperature nodes until now:
        for temp in temps[tstamp]:
            # print(f"At {tstamp} with temp {temp}")
            # print(temp_predictor(float_temp(temp), tstamp, t_delta_hours))

            for pred in temp_predictor(float_temp(temp), tstamp, t_delta_hours):
                if (
                    pred_dframe.loc[next_tstamp, "min_temp"]
                    <= pred["temp"]
                    <= pred_dframe.loc[next_tstamp, "max_temp"]
                ):
                    # A tiny number linear in temperature so
                    # the algorithm will favour lower temperatures
                    # when all else is equal.
                    hours_since_start = (next_tstamp - starttime).value / 1e9 / 60 / 60
                    num_stability_cost = (pred["temp"] + hours_since_start) / 100000000
                    # This is rounded away when cost is summed!

                    graph.add_edge(
                        (tstamp, temp),
                        (next_tstamp - pd.Timedelta(seconds=0), int_temp(pred["temp"])),
                        cost=pred["kwh"] * pred_dframe.loc[tstamp, "NOK/KWh"]
                        + num_stability_cost,
                        kwh=pred["kwh"],
                    )
                    next_temps_raw.append(int_temp(pred["temp"]))
        # Collapse next temperatures according to resolution
        next_temps_raw = sorted(list(set(next_temps_raw)))
        print(f"next_temps:  {next_temps_raw}")
        # Group
        if len(next_temps_raw) > 10:
            grouped_next_temps = sorted(
                list(
                    set(
                        [next_temps_raw[0]]
                        + list(grouper(next_temps_raw, 0.5 * TEMPERATURE_RESOLUTION))
                    )
                )
            )
            print(f"Grouped: {grouped_next_temps}")
            # Add fake edges for all collapsed nodes:
            for temp in next_temps_raw:
                if temp not in grouped_next_temps:
                    v = np.array(grouped_next_temps)
                    higher_grouped_temp = v[v > temp][0]
                    kwh = cost_from_predictor(
                        temp_predictor, temp, higher_grouped_temp, t_delta_hours, tstamp
                    )
                    # kwh is None if the higher_grouped_temp is not reachable from temp.
                    if kwh is not None:
                        print(f"ADDING FAKE EDGE {temp} {higher_grouped_temp}  {kwh}")
                        graph.add_edge(
                            (next_tstamp, temp),
                            (next_tstamp, higher_grouped_temp),
                            cost=kwh * pred_dframe.loc[tstamp, "NOK/KWh"],
                            kwh=kwh,
                        )
                    else:
                        print(f"NOT REACHABLE {temp} {higher_grouped_temp}  {kwh}")

            temps[next_tstamp] = grouped_next_temps
        else:
            temps[next_tstamp] = next_temps_raw
        print(f"Next temperatures are {temps[next_tstamp]}")
    # print(temps)
    # Build result dictionary:
    result = {
        "graph": graph,
        "path": cheapest_path(graph, pred_dframe.index[0]),
        "pred_frame": pred_dframe,
    }
    result["cost"] = round(sum(path_costs(graph, result["path"])), 5)
    result["onoff"] = path_onoff(result["path"])
    now_dict = {
        "on_at": starttime,
        "on_in": pd.Timedelta(0),
        "on_in_minutes": 0,
        "on_now": True,
    }
    never_dict = {"on_at": None, "on_in": None, "on_in_minutes": None, "on_now": False}
    if result["onoff"].empty or len(result["path"]) < len(pred_dframe):
        # If there is no graph or if it is not reached the very end,
        # we have either started too cold or too hot.
        if starttemp > pred_dframe.loc[starttime, "max_temp"]:
            result.update(never_dict)
        else:
            result.update(now_dict)
    elif result["onoff"].max() > 0:
        result["on_at"] = result["onoff"][result["onoff"] == 1].index[0]
        result["on_in"] = pd.to_datetime(result["on_at"]) - starttime
        result["on_in_minutes"] = result["on_in"].value / 1e9 / 60
        result["on_now"] = result["onoff"].values[0] == 1
        result["temperatures"] = path_values(result["path"])
        result["max_temp"] = result["temperatures"].max()
    else:
        # There is a path stepping down in temperature at each timestep:
        result["temperatures"] = path_values(result["path"])
        result["max_temp"] = result["temperatures"].max()
        result.update(never_dict)

    return result


def cost_from_predictor(
    predictor, temp1, temp2, t_delta_hours, tstamp=datetime.datetime.now()
):
    """Given a temperature predictor, predict the cost of going
    to a specific temperature within the predicted span"""
    prediction = predictor(temp1, tstamp, t_delta_hours)
    temps = [pred["temp"] for pred in prediction]
    kwhs = [pred["kwh"] for pred in prediction]

    t_point = (temp2 - min(temps)) / (max(temps) - min(temps))
    if -0.2 <= t_point <= 1.2:  # allow 20% extrapolation
        # Linear interpolation for kwh cost:
        return t_point * (max(kwhs) - min(kwhs))
    return None


def cheapest_path(graph, starttime):
    if not graph:
        return []
    startnode = find_node(graph, starttime, 0)
    endnode = find_node(graph, starttime + pd.Timedelta(hours=72), 0)
    return networkx.shortest_path(
        graph, source=startnode, target=endnode, weight="cost"
    )


def int_temp(temp):
    return int(round(temp * TEMPERATURE_RESOLUTION, 0))


def float_temp(int_temp):
    return float(int_temp / float(TEMPERATURE_RESOLUTION))


def plot_graph(graph, path=None, ax=None, show=False, maxnodes=2000):
    if ax is None:
        fig, ax = pyplot.subplots()
    cmap = pyplot.get_cmap("viridis")

    if path is not None:
        path_dframe = pd.DataFrame(path, columns=["index", "temp"]).set_index("index")
        path_dframe["temp"] = [float_temp(temp) for temp in path_dframe["temp"]]
        path_dframe.reset_index().plot(
            x="index",
            y="temp",
            linewidth=7,
            color=cmap(0.5),
            alpha=0.4,
            ax=ax,
            legend=False,
        )

    logger.info("Plotting some graph edges, wait for it..")
    counter = 0
    for edge_0, edge_1, data in graph.edges(data=True):
        counter += 1
        edge_frame = pd.DataFrame(
            [
                {"index": edge_0[0], "temp": float_temp(edge_0[1])},
                {"index": edge_1[0], "temp": float_temp(edge_1[1])},
            ]
        )

        edge_frame.plot(x="index", y="temp", ax=ax, legend=False)
        # print(edge_1)
        mid_time = edge_0[0] + (edge_1[0] - edge_0[0]) / 2
        pyplot.text(
            mid_time,
            float_temp((edge_0[1] + edge_1[1])) / 2,
            str(round(data["cost"], 5)),
        )
        pyplot.gcf().autofmt_xdate()

        if counter > maxnodes:
            break
    nodes_df = pd.DataFrame(data=graph.nodes, columns=["index", "temp"]).head(maxnodes)

    logger.info("Plotting all graph nodes..")
    nodes_df["temp"] = nodes_df["temp"].apply(float_temp)
    # print(nodes_df)
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
    if nodes_df.empty:
        return (None, None)
    # when = pd.Timestamp(when).astimezone(nodes_df["index"].dt.tz))
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


def path_values(path):
    timestamps = [node[0] for node in path]
    temps = [float_temp(node[1]) for node in path]
    return pd.Series(index=timestamps, data=temps)


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
    timespan = (endnode[0] - startnode[0]).value / 1e9 / 60 / 60  # from nanoseconds
    return {
        "opt_cost": opt_cost,
        "kwh": kwh,
        "opt_path": path,
    }


def grouper(iterable, distance=1):
    """Group a list of numbers into buckets with a certain minimal internal
    distance, and return the maximum of the numbers in a group"""
    prev = None
    group = []
    for item in iterable:
        if not prev:
            group.append(item)
        else:
            assert item >= prev
            if item - group[0] < distance:
                group.append(item)
            else:
                yield max(group)
                group = [item]
        prev = item
    if group:
        yield max(group)


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
