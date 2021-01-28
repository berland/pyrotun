import os
import asyncio
import networkx
import datetime
import itertools
import pytz
from sklearn import linear_model
import pandas as pd
from matplotlib import pyplot
import dotenv

import pyrotun

from pyrotun import persist  # noqa

logger = pyrotun.getLogger(__name__)

TIMEDELTA_MINUTES = 8  # minimum is 8 minutes!!
ROUND = 1
PD_TIMEDELTA = str(TIMEDELTA_MINUTES) + "min"
VACATION_ITEM = "Ferie"


# Heat capacity for water, converted from 4187 J/kg oC
CV = 1.163055556  # Wh/kg oC
# (1J = 1Ws; 3600J = 3600Ws = 1Wh)


class HeatReservoir:
    def __init__(
        self, sensor_item, thermostat_item, wattage, inc_rate=5, dec_rate=-0.5, max_temp=30
    ):
        self.sensor_item = sensor_item
        self.thermostat_item = thermostat_item
        self.wattage = wattage
        self.inc_rate = inc_rate
        self.dec_rate = dec_rate
        self.max_temp = max_temp
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
            maxtemp=self.max_temp,
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
                no_heater_temp = temp - self.dec_rate * t_delta_hours
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
                    cost = kwh * powerprice
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
        for edge_0, edge_1, data in graph.edges(data=True):
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


def predict_temploss(waterusageprofile, timestamp, t_delta):
    """Looks up in waterusageprofile and finds the
    expected temperature drop. The column "waterdiff"
    contains the actual temperature drop over the time
    interval in the profiles minute-index.

    Returns positive values, positive implies the temperature is decreasing.
    """
    # assert t_delta is the same as in the profile!
    assert "waterdiff" in waterusageprofile
    day = timestamp.weekday()
    hour = timestamp.hour
    minute = timestamp.minute
    if (day, hour, minute) not in waterusageprofile.index:
        waterusageprofile = waterusageprofile.copy()
        waterusageprofile.loc[day, hour, minute] = {}
        waterusageprofile.sort_index(inplace=True)
        waterusageprofile.ffill(inplace=True)
    loss = -waterusageprofile.loc[day, hour, minute]["waterdiff"]
    if loss == 0:
        logger.warning("Predicted zero temperature loss, this is wrong")
    return loss


def make_heatloss_diffusion_model(dframe_away, plot=False):
    """Estimate how fast temperature drops when nothing but heat
    diffusion is at play.

    Returns a linear model where the coefficient reveals
    how much the temperature decreases (in the implicit time interval) for
    one degree extra water temperature.

    """
    if "waterdiff" not in dframe_away:
        dframe_away = dframe_away.copy()
        # Make the difference pr. timeinterval:
        dframe_away["waterdiff"] = dframe_away["watertemp"].diff().shift(-1)

    def linearmodel(dframe, xvecs, yvec):
        model = linear_model.LinearRegression().fit(dframe[xvecs], dframe[yvec])
        return (model.intercept_, model.coef_[0])

    # Do we have a correlation between waterdiff and watertemp?
    vetted_rows = (dframe_away["watertemp"] < 75) & (dframe_away["waterdiff"] > -1)
    (intercept, coef) = linearmodel(
        dframe_away[vetted_rows], ["watertemp"], "waterdiff"
    )

    # Difference should increase with increasing temperature..
    if plot:
        ax = dframe_away.plot.scatter(x="watertemp", y="waterdiff")
        dframe_away["linestdiff"] = intercept + dframe_away["watertemp"] * coef
        ax = dframe_away.plot.scatter(x="watertemp", y="linestdiff", color="red", ax=ax)
        pyplot.show()
    return (intercept, coef)


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
    return onoff.astype(int)


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


async def controller(pers):
    # This is the production code path
    await pers.waterheater.controller()


async def estimate_savings(pers):
    # This is the production code path
    await pers.waterheater.estimate_savings()


async def main(heater_item):
    # This is typically used for interactive testing.
    pers = pyrotun.persist.PyrotunPersistence()
    # Make the weekly water usage profile, and persist it:
    await pers.ainit(["tibber", "influxdb", "openhab"])
    prices_df = await pers.tibber.get_prices()
    currenttemp = await pers.openhab.get_item(heater_item, datatype=float)
    starttemp = currenttemp
    graph = pers.waterheater.future_temp_cost_graph(
        # starttemp=60, prices_df=prices_df, mintemp=0.75, maxtemp=84, vacation=False
        starttemp=starttemp,
        prices_df=prices_df,
        mintemp=12,
        maxtemp=84,
        vacation=False,
    )

    if not graph:
        logger.warning("Temperature below minimum, should force on")
        await pers.aclose()
        return
    endtemp = 50
    opt_results = analyze_graph(graph, starttemp=starttemp, endtemp=endtemp)
    # Use plus/minus 2 degrees and 8 minute accuracy to estimate the do-nothing
    # scenario.
    logger.info(f"Cost is {opt_results['opt_cost']:.3f} NOK")
    logger.info(f"KWh is {opt_results['kwh']:.2f}")
    onoff = path_onoff(opt_results["opt_path"])

    # utc times:
    if onoff[onoff == 1].empty:
        logger.warning("Not turning on waterheater in foreseeable future")
    else:
        first_on_timestamp = onoff[onoff == 1].head(1).index.values[0]
        logger.info("Will turn heater on at %s", first_on_timestamp)

    await pers.waterheater.estimate_savings(prices_df)
    fig, ax = pyplot.subplots()
    plot_graph(graph, ax=ax, show=False)
    plot_path(opt_results["opt_path"], ax=ax, show=False)
    ax2 = ax.twinx()
    prices_df.plot(drawstyle="steps-post", y="NOK/KWh", ax=ax2, alpha=0.2)
    prices_df["mintemp"] = (
        prices_df.reset_index()["index"]
        .apply(watertemp_requirement, vacation=False, prices=prices_df)
        .values
    )
    prices_df.plot(drawstyle="steps-post", ax=ax, y="mintemp", color="blue", alpha=0.4)
    pyplot.show()

    await pers.aclose()


if __name__ == "__main__":
    dotenv.load_dotenv()
    # loop = asyncio.get_event_loop()
    # asyncio.ensure_future(main())
    # loop.run_forever()
    asyncio.run(main())
