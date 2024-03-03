import asyncio
import datetime
import itertools
import os

import dotenv
import networkx
import numpy as np
import pandas as pd
import pytz
from matplotlib import pyplot

import pyrotun
from pyrotun import persist  # noqa

logger = pyrotun.getLogger(__name__)

TIMEDELTA_MINUTES = 8  # minimum is 8 minutes!!
ROUND = 1
PD_TIMEDELTA = str(TIMEDELTA_MINUTES) + "min"
VACATION_ITEM = "Ferie"


async def main(
    pers=None,
    dryrun=False,
    plot=False,
    sensor_item="Termostat_Bad_Oppe_SensorGulv",
    thermostat_items=None,
    wattage=600,
    maxtemp=30,
):
    """Called from service or interactive"""
    if thermostat_items is None:
        thermostat_items = (
            [
                "Termostat_Bad_Oppe_SetpointHeating",
                "Termostat_Bad_Kjeller_SetpointHeating",
            ],
        )

    closepers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["tibber", "influxdb", "openhab"])
        closepers = True

    prices_df = await pers.tibber.get_prices()
    currenttemp = await pers.openhab.get_item(sensor_item, datatype=float)
    vacation = await pers.openhab.get_item(VACATION_ITEM, datatype=bool)

    if not (12 < currenttemp < 40):
        logger.error("Currenttemp was %s, can't be right, giving up" % str(currenttemp))
        # Backup temperature:
        for thermostat_item in thermostat_items:
            await pers.openhab.set_item(thermostat_item, "24", log=True)
        return

    bathfloor = HeatReservoir(
        sensor_item,
        thermostat_items[0],
        wattage=wattage,
        inc_rate=5,  # Degrees pr.  hour
        dec_rate=-0.3,  # Degrees pr. hour
    )
    graph = bathfloor.future_temp_cost_graph(
        starttemp=currenttemp,
        prices_df=prices_df,
        mintemp=17,
        maxtemp=maxtemp,
        vacation=vacation,
    )

    if not graph:
        logger.warning("Temperature below minimum, should force on")
        for thermostat_item in thermostat_items:
            if not dryrun:
                await pers.openhab.set_item(thermostat_item, "25")
        return

    endtemp = 25
    opt_results = analyze_graph(graph, starttemp=currenttemp, endtemp=endtemp)
    logger.info(f"Bathfloor cost is {opt_results['opt_cost']:.2f}")
    logger.info(f"Bathfloor KWh is {opt_results['kwh']:.2f}")

    onoff = path_onoff(opt_results["opt_path"])
    thermostat_values = path_thermostat_values(opt_results["opt_path"])

    if not dryrun:
        for thermostat_item in thermostat_items:
            await pers.openhab.set_item(
                thermostat_item, str(thermostat_values.values[0]), log=True
            )
    first_on_timestamp = onoff[onoff == 1].head(1).index.values[0]
    logger.info("Will turn bathfloor on at %s", first_on_timestamp)

    if plot:
        fig, ax = pyplot.subplots()
        # plot_graph(graph, ax=ax, show=True)  # Plots all nodes in graph.
        plot_path(opt_results["opt_path"], ax=ax, show=False)
        ax2 = ax.twinx()
        prices_df.plot(drawstyle="steps-post", y="NOK/KWh", ax=ax2, alpha=0.2)
        prices_df["mintemp"] = (
            prices_df.reset_index()["index"]
            .apply(temp_requirement, vacation=False, prices=prices_df)
            .values
        )
        prices_df.plot(
            drawstyle="steps-post", ax=ax, y="mintemp", color="blue", alpha=0.4
        )
        pyplot.show()

    if closepers:
        await pers.aclose()


class HeatReservoir:
    def __init__(
        self,
        sensor_item,
        thermostat_item,
        wattage,
        inc_rate=5,  # Degrees pr.  hour
        dec_rate=-0.3,  # Degrees pr. hour
        maxtemp=30,
    ):
        self.sensor_item = sensor_item
        self.thermostat_item = thermostat_item
        self.wattage = wattage
        self.inc_rate = inc_rate
        self.dec_rate = dec_rate
        self.maxtemp = maxtemp
        # The global persistence object
        self.pers = None

    async def ainit(self, pers):
        logger.info("HeatReservoir.ainit()")
        self.pers = pers

    async def controller(self):
        """Control the heat reservoir in an optimized way.

        Sends ON/OFF to OpenHAB"""
        if self.pers.openhab is None:
            asyncio.sleep(1)
        assert self.pers.openhab is not None

        currenttemp = await self.pers.openhab.get_item(self.sensor_item, datatype=float)
        if currenttemp is None:
            await self.pers.openhab.set_item(self.heater_item, "ON", log=True)
            return

        vacation = await self.pers.openhab.get_item(VACATION_ITEM, datatype=bool)

        prices_df = await self.pers.tibber.get_prices()

        graph = self.future_temp_cost_graph(
            starttemp=currenttemp,
            prices_df=prices_df,
            vacation=vacation,
            starttime=None,
            maxhours=36,
            maxtemp=self.maxtemp,
        )
        # If we are below minimum temperature, we get a graph with 0 nodes.
        # Turn on if so.
        if not graph:
            logger.info(
                "Temperature now %s is below minimum temperature, forcing ON",
                str(currenttemp),
            )
            await self.pers.openhab.set_item(self.heater_item, "ON", log=True)
            return

        opt_results = analyze_graph(graph, starttemp=currenttemp, endtemp=25)

        # Turn on if temperature in the path should increase:
        if opt_results["opt_path"][1][1] > opt_results["opt_path"][0][1]:
            await self.pers.openhab.set_item(self.heater_item, "ON", log=True)
        else:
            await self.pers.openhab.set_item(self.heater_item, "OFF", log=True)

        onoff = pd.DataFrame(
            columns=["onoff"], data=path_onoff(opt_results["opt_path"])
        )
        # Log when we will turn on:
        if not onoff[onoff["onoff"] == 1].empty:
            firston = onoff[onoff["onoff"] == 1].head(1).index.values[0]
            logger.info("Will turn heater on at %s", firston)
        else:
            logger.warning("Not planning to turn on %s, hot enough?", self.heater_item)

    def future_temp_cost_graph(
        self,
        starttemp=60,
        prices_df=None,
        vacation=False,
        starttime=None,
        maxhours=36,
        mintemp=40,  # Can be used to set higher reqs.
        maxtemp=84,
    ):
        """Build the networkx Directed 2D ~lattice  Graph, with
        datetime on the x-axis and water-temperatur on the y-axis.

        Edges from nodes determined by (time, temp) has an associated
        cost in NOK.
        """
        logger.info("Building graph for future reservoir temperatures")
        if starttime is None:
            tz = pytz.timezone(os.getenv("TIMEZONE"))
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

        # dict of timestamp to list of reachable temperatures:
        temps = {}
        temps[dframe.index[0]] = [starttemp]
        # Loop over all datetimes, and inject nodes and
        # possible edges
        # dframe = dframe.head(10)
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
                assert self.dec_rate < 0
                no_heater_temp = temp + self.dec_rate * t_delta_hours
                min_temp = max(
                    temp_requirement(tstamp, vacation=vacation, prices=prices_df),
                    mintemp,
                )
                if no_heater_temp > min_temp:
                    # Add an edge for the no-heater-scenario:
                    temps[next_tstamp].append(no_heater_temp)
                    graph.add_edge(
                        (tstamp, temp),
                        (next_tstamp, no_heater_temp),
                        cost=0,
                        kwh=0,
                        tempdeviation=abs(no_heater_temp - starttemp),
                    )

                heater_on_temp = round(
                    temp + self.inc_rate * t_delta_hours,
                    ROUND,
                )
                if min_temp < heater_on_temp < maxtemp:
                    kwh = self.wattage / 1000 * t_delta_hours
                    hightemp_penalty = (heater_on_temp - 15) / 20 / 100
                    hightemp_penalty = 1
                    cost = kwh * hightemp_penalty * powerprice
                    # Small penalty for high temps:

                    # Add edge for heater-on:
                    temps[next_tstamp].append(heater_on_temp)
                    graph.add_edge(
                        (tstamp, temp),
                        (next_tstamp, heater_on_temp),
                        cost=cost,
                        kwh=kwh,
                        tempdeviation=abs(heater_on_temp - starttemp),
                    )

        logger.info(
            f"Built graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
        )
        return graph


def plot_graph(graph, ax=None, show=False):
    if ax is None:
        fig, ax = pyplot.subplots()

    if len(graph.edges) < 1000:
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


def temp_requirement(timestamp, vacation=False, prices=None):
    hour = timestamp.hour

    if vacation:
        return 15
    if hour < 6:
        return 20
    return 25


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
    tempvalues = [node[1] for node in path]
    # Perturb temperatures, +1 when it should be on, and -1 when off:
    onoff = pd.Series(tempvalues).diff().shift(-1).dropna().apply(np.sign)
    onoff.index = timestamps
    inttemp_s = pd.Series(tempvalues).astype(int)
    inttemp_s.index = timestamps + [np.nan]
    return (inttemp_s + onoff).dropna()


def path_kwh(graph, path):
    return [graph.edges[path[i], path[i + 1]]["kwh"] for i in range(len(path) - 1)]


def shortest_paths(graph, k=5, starttemp=60, endtemp=60):
    """Return the k shortest paths. Runtime is K*N**3, too much
    for practical usage"""
    startnode = find_node(graph, datetime.datetime.now(), starttemp)
    endnode = find_node(graph, datetime.datetime.now() + pd.Timedelta("48h"), endtemp)
    # Path generator:
    return list(
        itertools.islice(
            networkx.shortest_simple_paths(
                graph, source=startnode, target=endnode, weight="cost"
            ),
            k,
        )
    )


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


if __name__ == "__main__":
    # Interactive testing:
    dotenv.load_dotenv()
    asyncio.run(main(pers=None, dryrun=True, plot=True))
