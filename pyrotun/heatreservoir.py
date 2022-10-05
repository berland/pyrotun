import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import networkx
import numpy as np
import pandas as pd
from matplotlib import pyplot

import pyrotun

logger = pyrotun.getLogger(__name__)


def prediction_dframe(
    starttime: datetime.datetime,
    prices: pd.Series,
    min_temp: Optional[Union[pd.Series, float]] = None,
    max_temp: Optional[Union[pd.Series, float]] = None,
    freq: str = "10min",
    maxhours: int = 36,
) -> pd.DataFrame:
    """Spread prices on a time-series of the requested frequency into
    the future.

    Returned dataframe has a datetime with timezone as index, at requested
    frequency. Columns are:

    * maxtemp (max temperature requiremend, valid forward in time)
    """
    starttime_wholehour = starttime.replace(minute=0, second=0, microsecond=0)
    datetimes = pd.date_range(
        starttime_wholehour,
        prices.index.max() + pd.Timedelta(1, unit="hour"),
        freq=freq,
        tz=prices.index.tz,
    )

    # Only timestamps after starttime is up for prediction:
    datetimes = datetimes[datetimes >= starttime - pd.Timedelta(freq)]

    # Delete timestamps mentioned in prices, for correct merging:
    duplicates = []
    for tstamp in datetimes:
        if tstamp in prices.index:
            duplicates.append(tstamp)
    datetimes = datetimes.drop(duplicates)

    # Merge prices into the requested datetime:
    dframe = pd.concat(
        [
            pd.DataFrame(prices),
            pd.DataFrame(index=datetimes),
        ],
        axis="index",
    )
    dframe.columns = ["NOK/KWh"]
    dframe = dframe.sort_index()

    if min_temp is not None:
        if isinstance(min_temp, pd.Series):
            # TODO: INTERPOLATE IN TEMPERATURES
            dframe["min_temp"] = min_temp
        else:
            dframe["min_temp"] = min_temp
    else:
        dframe["min_temp"] = -100

    if max_temp is not None:
        if isinstance(max_temp, pd.Series):
            # TODO: INTERPOLATE IN TEMPERATURES
            dframe["max_temp"] = max_temp
        else:
            dframe["max_temp"] = max_temp
    else:
        dframe["max_temp"] = 999

    dframe = dframe[dframe.index < starttime + pd.Timedelta(maxhours, unit="hour")]
    # Constant extrapolation of prices:
    dframe = dframe.ffill().bfill()
    # Is is a rounding error we solve by the one second delta?
    dframe = dframe[dframe.index > starttime - pd.Timedelta(freq)]
    print(dframe)
    return dframe


def interpolate_predictor(
    predictor_result: List[Dict[str, float]], temperature: float
) -> float:
    # pylint: disable=invalid-name
    def extrap(x, xp, yp):
        """np.interp function with linear extrapolation"""
        y = np.interp(x, xp, yp)
        y = np.where(
            x < xp[0], yp[0] + (x - xp[0]) * (yp[0] - yp[1]) / (xp[0] - xp[1]), y
        )
        y = np.where(
            x > xp[-1], yp[-1] + (x - xp[-1]) * (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]), y
        )
        return y

    assert set(["temp", "kwh"]).issubset(set(predictor_result[0].keys()))
    dframe = pd.DataFrame(predictor_result).set_index("temp").sort_index()
    return extrap(temperature, dframe.index.values, dframe["kwh"].values)


def optimize(
    starttime: datetime.datetime,
    starttemp: float = 20,
    prices: pd.Series = None,  #
    min_temp: Optional[Union[pd.Series, float]] = None,
    max_temp: Optional[Union[pd.Series, float]] = None,
    maxhours: int = 36,
    temp_predictor: Callable = None,
    freq: str = "10min",  # pd.date_range frequency
) -> Dict[str, Any]:
    """Build a networkx Directed 2D ~lattice Graph, with
    datetime on the x-axis and temperatures on the y-axis.

    Only at time zero, the emanating edges are guaranteed to point
    to nodes that are attainable with a pure on/off for the relevant
    timedelta. Afterwards, nodes are approximately attainable, and the
    cost is scaled linearly, this is in order to reduce the number
    of nodes in the lattice.

    Edges from nodes determined by (time, temp) has an associated
    cost in NOK and energy need in kwh

    The callable `temp_predictor` is a function that returns List[Dict[str,
    float]], one element for each predicted temperature, and the dict must
    contain the keys `temp` and `kwh`.

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
    assert temp_predictor is not None
    # Get a series with prices at the datetimes we want to optimize at:
    pred_dframe = prediction_dframe(
        starttime, prices, min_temp, max_temp, freq, maxhours
    )

    logger.info("Building graph for future temperatures")

    # Build Graph, starting with current temperature
    graph = networkx.DiGraph()
    temps = {}  # timestamps are keys, values are lists of scaled integer temps.

    # Initial point/node:
    temps[pred_dframe.index[0]] = [starttemp]

    # We allow only constant timesteps:
    timesteps = pd.Series(pred_dframe.index).diff()[1:]
    assert (
        len(set(timesteps)) == 1
    ), f"Only constant timestemps allowed, got {set(timesteps)}"

    t_delta_hours = (
        float((pred_dframe.index[1] - pred_dframe.index[0]).value) / 1e9 / 60 / 60
    )  # convert from nanoseconds to hours.

    # Loop over time to build the graph:
    for tstamp, next_tstamp in zip(pred_dframe.index, pred_dframe.index[1:]):
        next_temps: List[int] = []

        # Loop over available temperature nodes until now:
        for temp in temps[tstamp]:
            predictions = temp_predictor(temp, tstamp, t_delta_hours)

            node_separation = (
                max(pred["temp"] for pred in predictions)
                - min(pred["temp"] for pred in predictions)
            ) / 2

            for pred in predictions:
                if (
                    pred_dframe.loc[next_tstamp, "min_temp"]
                    <= pred["temp"]
                    <= pred_dframe.loc[next_tstamp, "max_temp"]
                ):

                    if next_temps:
                        signed_distances = [
                            pred["temp"] - next_temp for next_temp in next_temps
                        ]
                        min_signed_distance = min(signed_distances, key=abs)
                    if next_temps and abs(min_signed_distance) < node_separation:
                        # Use existing node
                        existing_temp_prediction = pred["temp"] - min_signed_distance
                        kwh = interpolate_predictor(
                            predictions, existing_temp_prediction
                        )

                        pred_temp = existing_temp_prediction
                    else:
                        # Add edge to a new node
                        # print(f"add edge to non-existing prediction {pred}")
                        pred_temp = pred["temp"]
                        kwh = pred["kwh"]
                        next_temps.append(pred_temp)

                    # A tiny number linear in temperature so
                    # the algorithm will favour lower temperatures
                    # when all else is equal.
                    hours_since_start = (next_tstamp - starttime).value / 1e9 / 60 / 60
                    hours_since_start = 0
                    num_stability_cost = (pred_temp + hours_since_start) / 100000000
                    # This is rounded away when cost is summed!

                    graph.add_edge(
                        (tstamp, temp),
                        (next_tstamp - pd.Timedelta(seconds=0), pred_temp),
                        cost=kwh * pred_dframe.loc[tstamp, "NOK/KWh"]
                        + num_stability_cost,
                        kwh=kwh,
                    )
        temps[next_tstamp] = list(set(next_temps))

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
        plot_graph(graph, path=result["path"], show=True)
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
    predictor: Callable,
    temp1: float,
    temp2: float,
    t_delta_hours: float,
    tstamp: datetime.datetime = datetime.datetime.now(),
) -> Optional[float]:
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


def cheapest_path(graph, starttime: datetime.datetime):
    if not graph:
        return []
    startnode = find_node(graph, starttime, 0)
    endnode = find_node(graph, starttime + pd.Timedelta(hours=72), 0)
    return networkx.shortest_path(
        graph, source=startnode, target=endnode, weight="cost"
    )


def plot_graph(
    graph, path=None, ax=None, show: bool = False, maxnodes: int = 2000
) -> None:
    if ax is None:
        _fig, ax = pyplot.subplots()
    cmap = pyplot.get_cmap("viridis")

    if path is not None:
        path_dframe = pd.DataFrame(path, columns=["index", "temp"]).set_index("index")
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
                {"index": edge_0[0], "temp": edge_0[1]},
                {"index": edge_1[0], "temp": edge_1[1]},
            ]
        )

        edge_frame.plot(x="index", y="temp", ax=ax, legend=False)
        mid_time = edge_0[0] + (edge_1[0] - edge_0[0]) / 2
        pyplot.text(
            mid_time,
            (edge_0[1] + edge_1[1]) / 2,
            str(round(data["cost"], 5)),
        )
        pyplot.gcf().autofmt_xdate()

        if counter > maxnodes:
            break
    nodes_df = pd.DataFrame(data=graph.nodes, columns=["index", "temp"]).head(maxnodes)

    logger.info("Plotting all graph nodes..")
    nodes_df.plot.scatter(x="index", y="temp", ax=ax)

    if show:
        pyplot.show()


def plot_path(
    path, ax=None, show: bool = False, linewidth: int = 2, color: str = "red"
) -> None:
    if ax is None:
        _, ax = pyplot.subplots()

    path_dframe = pd.DataFrame(path, columns=["index", "temp"]).set_index("index")
    # path_dframe["temp"] = [float_temp(temp) for temp in path_dframe["temp"]]

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


def find_node(graph, when: datetime.datetime, temp: float) -> tuple:
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


def temp_requirement(
    timestamp: datetime.datetime,
    vacation: bool = False,
    prices: Optional[pd.DataFrame] = None,
    delta: float = 0,
) -> float:
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
    onoff = pd.Series(temps, dtype=float).diff().shift(-1).dropna()
    onoff.index = timestamps
    return np.maximum(0, np.sign(onoff))


def path_values(path: Iterable[Any]) -> pd.Series:
    timestamps = [node[0] for node in path]
    temps = [node[1] for node in path]
    return pd.Series(index=timestamps, data=temps)


def path_thermostat_values(path: Iterable[Any]) -> pd.Series:
    """Extract a Pandas series of thermostat values (integers) from a
    path with temperatures"""
    timestamps = [node[0] for node in path][:-1]  # skip the last one
    tempvalues = [node[1] for node in path]
    # Perturb temperatures, +1 when it should be on, and -1 when off:
    onoff = pd.Series(tempvalues).diff().shift(-1).dropna().apply(np.sign)
    onoff.index = timestamps
    inttemp_s = pd.Series(tempvalues).astype(int)
    inttemp_s.index = timestamps + [np.nan]
    return (inttemp_s + onoff).dropna()


def path_kwh(graph, path):
    return [graph.edges[path[i], path[i + 1]]["kwh"] for i in range(len(path) - 1)]


def analyze_graph(
    graph,
    starttemp: float = 60,
    endtemp: float = 60,
    starttime: datetime.datetime = None,
) -> Dict[str, Any]:
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
    return {
        "opt_cost": opt_cost,
        "kwh": kwh,
        "opt_path": path,
    }
