import asyncio
import datetime
import os

import dotenv
import networkx
import pandas as pd
import pytz
from matplotlib import pyplot

import pyrotun
from pyrotun import powermodels
from pyrotun.connections import localpowerprice

logger = pyrotun.getLogger(__name__)

TIMEDELTA_MINUTES = 60  # minimum is 8 minutes!!
ROUND = 1
PD_TIMEDELTA = str(TIMEDELTA_MINUTES) + "min"
SENSOR_ITEM = "Sensor_Kjoleskap_temperatur"
SETPOINT_ITEM = "Setpoint_optimized"
MINIMUM_TEMPERATURE = 5


class ElvatunHeating:
    def __init__(self):
        # The global persistence object
        self.pers = None
        self.powerusagemodel = None

    async def ainit(self, pers):
        logger.info("Elvatunheating.ainit()")
        self.pers = pers
        await self.update_heatingmodel()

    async def update_heatingmodel(self):
        self.powerusagemodel: dict = await powermodels.make_heatingmodel(
            self.pers,
            target=SENSOR_ITEM,
            ambient="YrtemperaturMjolfjell",
            powermeasure=[
                "Namronovn_Stue_800w_effekt",
                "Namronovn_Bad_400w_effekt",
                "Namronovn_Gang_600w_effekt",
            ],
            include_sun=False,
        )

    async def controller(self) -> float:
        """Calculates the optimal path of setpoint temperatures for the future.

        Returns the updated optimal setpoint"""
        assert self.pers is not None
        if self.pers.openhab is None:
            await asyncio.sleep(1)
        assert self.pers.openhab is not None

        currenttemp = await self.pers.openhab.get_item(SENSOR_ITEM, datatype=float)
        currenttemp = round(currenttemp * 2) / 2

        currentsetpoint = await self.pers.openhab.get_item(
            SETPOINT_ITEM, datatype=float
        )

        if currenttemp is None:
            logger.warning(
                f"Setpoint set to {MINIMUM_TEMPERATURE}, "
                "no knowledge of current temperature"
            )
            return MINIMUM_TEMPERATURE

        prices_df = await self.pers.tibber.get_prices()
        prices_df = prices_df.copy()
        prices_df["NOK/KWh"] += localpowerprice.get_gridrental(prices_df.index)

        weather_forecast = await self.pers.yr.forecast()

        graph = self.future_temp_cost_graph(
            starttemp=currentsetpoint,
            prices_df=prices_df,
            temp_forecast=weather_forecast["air_temperature"],
            mintemp=4,
            maxtemp=12,
        )

        if not graph:
            logger.info(
                f"Temperature now {currenttemp} is below minimum",
            )
            return MINIMUM_TEMPERATURE

        opt_results = analyze_graph(
            graph, starttemp=currenttemp, endtemp=6
        )

        logger.info(f"Cost is {opt_results['opt_cost']:.3f} NOK")
        logger.info(f"KWh is {opt_results['kwh']:.2f}")
        return opt_results["opt_path"][1][1]

    def future_temp_cost_graph(
        self,
        starttemp=60,
        prices_df=None,
        temp_forecast: pd.Series = None,
        starttime=None,
        mintemp=4,  # Can be used to set higher reqs.
        maxtemp=12,
    ):
        """Build the networkx Directed 2D ~lattice Graph, with
        datetime on the x-axis and water-temperatur on the y-axis.

        Edges from nodes determined by (time, temp) has an associated
        cost in NOK.
        """
        logger.info("Building graph for future indoor temperatures")
        if starttime is None:
            tz = pytz.timezone(os.getenv("TIMEZONE", ""))
            starttime = datetime.datetime.now().astimezone(tz)
        starttime_wholehour = starttime.replace(minute=0, second=0, microsecond=0)
        datetimes = pd.date_range(
            starttime_wholehour,
            prices_df.index.max() + pd.Timedelta(1, unit="hour"),
            freq=PD_TIMEDELTA,
            tz=prices_df.index.tz,
        )
        # If we are at 19:48 and timedelta is 15 minutes, we should
        # round down to 19:45:
        datetimes = datetimes[datetimes > starttime - pd.Timedelta(PD_TIMEDELTA)]
        # Merge prices into the requested datetime:
        dframe = (
            pd.concat(
                [
                    prices_df,
                    pd.DataFrame(index=datetimes),
                ],
                axis="index",
            )
            .sort_index()
            .ffill()
            .bfill()  # (this is hardly necessary)
            .loc[datetimes]  # slicing to this means we do not compute
            # correcly around hour shifts
        )
        dframe = dframe[~dframe.index.duplicated(keep="first")]

        # Build Graph, starting with current temperature
        # temps = np.arange(40, 85, 0.1)

        graph = networkx.DiGraph()

        temps: dict[pd.Timestamp, list[float]] = {}

        temps[dframe.index[0]] = [starttemp]
        # Loop over all datetimes, and inject nodes and
        # possible edges
        for tstamp, next_tstamp in zip(dframe.index, dframe.index[1:]):
            temps[next_tstamp] = []
            temps[tstamp] = list(set(temps[tstamp]))  # Make rolled over temps unique
            temps[tstamp].sort()
            powerprice = dframe.loc[tstamp]["NOK/KWh"]
            for temp in temps[tstamp]:
                # Namronovner kan styres på halv-grader
                possible_setpoint_deltas = [-2, -1, 0, 0.25]
                for setpoint_delta in possible_setpoint_deltas:
                    if not (mintemp <= temp + setpoint_delta <= maxtemp):
                        continue
                    temps[next_tstamp].append(temp + setpoint_delta)
                    kwh = self.powerusagemodel["powermodel"].predict(
                        [
                            [
                                setpoint_delta,
                                temp + setpoint_delta - temp_forecast[tstamp],
                            ]
                        ]
                    )[0][0]
                    cost = max(kwh * powerprice, 0)
                    # print(
                    #    f"{tstamp} Heating from {temp} to {temp + setpoint_delta} at {kwh} {cost=}"
                    # )
                    graph.add_edge(
                        (tstamp, temp),
                        (next_tstamp, temp + setpoint_delta),
                        cost=cost,
                        kwh=kwh,
                        tempdeviation=abs(setpoint_delta),
                    )

        logger.info(
            f"Built graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
        )
        return graph


def plot_graph(graph, ax=None, show=False):
    if ax is None:
        fig, ax = pyplot.subplots()

    if len(graph.edges) < 2000:
        for edge_0, edge_1, _data in graph.edges(data=True):
            pd.DataFrame(
                [
                    {"index": edge_0[0], "temp": edge_0[1]},
                    {"index": edge_1[0], "temp": edge_1[1]},
                ]
            ).plot(x="index", y="temp", ax=ax, legend=False)
    nodes_df = pd.DataFrame(data=graph.nodes, columns=["index", "temp"])

    nodes_df.plot.scatter(x="index", y="temp", ax=ax)

    if show:
        pyplot.show()


def plot_path(path, ax=None, show=False, linewidth=2, color="red"):
    if ax is None:
        fig, ax = pyplot.subplots()

    path_dframe = pd.DataFrame(path, columns=["index", "temp"]).set_index("index")
    path_dframe.plot(
        y="temp", drawstyle="steps-post", linewidth=linewidth, color=color, ax=ax
    )
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


def path_costs(graph, path):
    """Compute a list of the cost along a temperature path, in NOK"""
    return [graph.edges[path[i], path[i + 1]]["cost"] for i in range(len(path) - 1)]


def path_kwh(graph, path):
    return [graph.edges[path[i], path[i + 1]]["kwh"] for i in range(len(path) - 1)]


def analyze_graph(graph, starttemp=60, endtemp=60):
    """Find shortest path, and do some extra calculations for estimating
    savings. The savings must be interpreted carefully, and is
    probably only correct if start and endtemp is equal"""

    startnode = find_node(graph, datetime.datetime.now(), starttemp)
    endnode = find_node(graph, datetime.datetime.now() + pd.Timedelta("48h"), endtemp)
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


async def main():
    # This is typically used for interactive testing.
    pers = pyrotun.persist.PyrotunPersistence()

    await pers.ainit(["tibber", "influxdb", "openhab", "yr"])
    prices_df = await pers.tibber.get_prices()

    forecast = await pers.yr.forecast()

    elvatunheating = ElvatunHeating()
    await elvatunheating.ainit(pers)

    # Grid rental is time dependent:
    prices_df["NOK/KWh"] += localpowerprice.get_gridrental(
        prices_df.index, provider="tendranett"
    )

    # currenttemp = await pers.openhab.get_item(SENSOR_ITEM, datatype=float)
    # MOCK
    currenttemp = 4

    starttemp = currenttemp
    graph = elvatunheating.future_temp_cost_graph(
        starttemp=starttemp,
        prices_df=prices_df,
        temp_forecast=forecast["air_temperature"],
        mintemp=3,
        maxtemp=12,
    )

    if not graph:
        logger.warning("Indoor temperature below minimum, should force on")
        await pers.aclose()
        return
    endtemp = 6
    opt_results = analyze_graph(graph, starttemp=starttemp, endtemp=endtemp)

    logger.info(f"Cost is {opt_results['opt_cost']:.3f} NOK")
    logger.info(f"KWh is {opt_results['kwh']:.2f}")
    logger.info(f"Setpoint path: {opt_results['opt_path']}")
    _, ax = pyplot.subplots()
    # plot_graph(graph, ax=ax, show=False)
    plot_path(opt_results["opt_path"], ax=ax, show=False)

    # pyplot.show()

    ax2 = ax.twinx()
    prices_df.plot(drawstyle="steps-post", y="NOK/KWh", ax=ax2)  # , alpha=0.9)
    (forecast["air_temperature"] / 10).plot(ax=ax2)

    ax.set_xlim(
        left=datetime.datetime.now() - datetime.timedelta(hours=2),
        right=prices_df.index.max(),
    )
    pyplot.grid()
    pyplot.show()

    await pers.aclose()


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main(), debug=False)