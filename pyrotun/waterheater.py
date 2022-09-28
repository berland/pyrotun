import asyncio
import datetime
import itertools
import os
from typing import Tuple

import dotenv
import networkx
import pandas as pd
import pytz
from matplotlib import pyplot
from sklearn import linear_model

import pyrotun
from pyrotun import persist  # noqa
from pyrotun.connections import localpowerprice

logger = pyrotun.getLogger(__name__)

TIMEDELTA_MINUTES = 8  # minimum is 8 minutes!!
ROUND = 1
PD_TIMEDELTA = str(TIMEDELTA_MINUTES) + "min"
SENSOR_ITEM = "Varmtvannsbereder_temperatur"
HEATERCONTROLLER_ITEM = "Varmtvannsbereder_bryter"
TARGETTEMP_ITEM = "Varmtvannsbereder_temperaturtarget"
SAVINGS24H_ITEM = "Varmtvannsbereder_besparelse"
VACATION_ITEM = "Ferie"

WATTAGE = 2600
LITRES = 194

# Heat capacity for water, converted from 4187 J/kg oC
CV = 1.163055556  # Wh/kg oC
# (1J = 1Ws; 3600J = 3600Ws = 1Wh)


class WaterHeater:
    def __init__(self):
        # self.weekly_water_usage_frame = None
        self.waterusageprofile = None

        # The global persistence object
        self.pers = None

        # How fast temperature drops as a function of watertemperature
        # (assuming a constant time interval)
        self.heatlossdiffusionmodel = None  # a tuple (intercept, coef)
        self.meanwatertemp = None

    async def ainit(self, pers):
        logger.info("Waterheater.ainit()")
        self.pers = pers
        (
            self.waterusageprofile,
            self.heatlossdiffusionmodel,
            self.meanwatertemp,
        ) = await make_weekly_profile(pers, plot=False)

    async def controller(self):
        """Control the waterheater in an optimized way.

        Sends ON/OFF to OpenHAB, and populates some statistical measures"""
        if self.pers.openhab is None:
            asyncio.sleep(1)
        assert self.pers.openhab is not None

        currenttemp = await self.pers.openhab.get_item(SENSOR_ITEM, datatype=float)
        if currenttemp is None:
            logger.warning("Waterheater set ON, no temperature knowledge")
            await self.pers.openhab.set_item(HEATERCONTROLLER_ITEM, "ON", log=True)
            await self.pers.openhab.set_item(TARGETTEMP_ITEM, 65, log=True)
            return

        vacation = await self.pers.openhab.get_item(VACATION_ITEM, datatype=bool)

        prices_df = await self.pers.tibber.get_prices()
        # Grid rental is time dependent:
        prices_df = prices_df.copy()
        prices_df["NOK/KWh"] += localpowerprice.get_gridrental(prices_df.index)

        graph = self.future_temp_cost_graph(
            starttemp=currenttemp,
            prices_df=prices_df,
            vacation=vacation,
            starttime=None,
            maxhours=36,
            mintemp=40,
            maxtemp=84,
        )
        # If we are below minimum temperature, we get a graph with 0 nodes.
        # Turn on waterheater if so.
        if not graph:
            logger.info(
                "Temperature now %s is below minimum water temperature, forcing ON",
                str(currenttemp),
            )
            await self.pers.openhab.set_item(HEATERCONTROLLER_ITEM, "ON", log=True)
            # High target gives high priority:
            await self.pers.openhab.set_item(
                TARGETTEMP_ITEM, currenttemp + 10, log=True
            )
            return

        opt_results = analyze_graph(graph, starttemp=currenttemp, endtemp=55)

        # Turn on if temperature in the path should increase:
        next_temp = opt_results["opt_path"][1][1]
        await self.pers.openhab.set_item(TARGETTEMP_ITEM, next_temp, log=True)
        if next_temp > opt_results["opt_path"][0][1]:
            await self.pers.openhab.set_item(HEATERCONTROLLER_ITEM, "ON", log=True)
        else:
            await self.pers.openhab.set_item(HEATERCONTROLLER_ITEM, "OFF", log=True)

        logger.info(f"Cost is {opt_results['opt_cost']:.3f} NOK")
        logger.info(f"KWh is {opt_results['kwh']:.2f}")

        # Dump CSV for legacy plotting solution, naive timestamps:
        isonowhour = datetime.datetime.now().replace(minute=0).strftime("%Y-%m-%d-%H00")
        onoff = pd.DataFrame(
            columns=["onoff"], data=path_onoff(opt_results["opt_path"])
        )
        onoff.index = onoff.index.tz_localize(None)
        onoff["timestamp"] = onoff.index
        onoff.to_csv("/home/berland/heatoptplots/waterheater-" + isonowhour + ".csv")

        # Log when we will turn on:
        if not onoff[onoff["onoff"] == 1].empty:
            firston = onoff[onoff["onoff"] == 1].head(1).index.values[0]
            logger.info("Will turn heater on at %s", firston)
        else:
            logger.warning("Not planning to turn on waterheater, hot enough?")

    async def estimate_savings(self, prices_df=None, starttemp=70):
        """Savings must be estimated without reference to current temperature
        and needs only to be run for every price change (i.e. every hour)"""
        if prices_df is None:
            prices_df = await self.pers.tibber.get_prices()
            # Grid rental is time dependent:
            prices_df = prices_df.copy()
            prices_df["NOK/KWh"] += localpowerprice.get_gridrental(prices_df.index)

        logger.info("Building graph for 24h savings estimation")
        # starttemp should be above the minimum temperature, as the
        # temperature is often higher than minimum, as we are able to
        # "cache" temperature. In situations with falling prices, this could
        # still overestimate the savings (?)
        graph = self.future_temp_cost_graph(
            starttemp=starttemp,
            prices_df=prices_df,
            vacation=False,
            starttime=None,
            maxhours=36,
            mintemp=40,
            maxtemp=84,
        )
        # There is a "tempdeviation" parameter in the graph which
        # is the deviation from the starttemp at every node. Minimizing
        # on this is the same as running with thermostat control.

        # The thermostat mode cost can also be estimated directly from
        # the usage data probably, without any reference to the graph.

        opt_results = analyze_graph(graph, starttemp=starttemp, endtemp=starttemp)

        logger.info(f"Reference optimized cost is {opt_results['opt_cost']:.3f} NOK")
        logger.info(f"Thermostat cost is {opt_results['no_opt_cost']:.3f} NOK")
        logger.info(
            f"24-hour saving from Dijkstra: {opt_results['savings24h']:.2f} NOK"
        )
        await self.pers.openhab.set_item(
            SAVINGS24H_ITEM, str(opt_results["savings24h"])
        )

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
        logger.info("Building graph for future water temperatures")
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
            temploss = predict_temploss(
                self.waterusageprofile, tstamp, t_delta
            ) - self.diffusionloss(self.meanwatertemp, t_delta)
            for temp in temps[tstamp]:
                # This is Explicit Euler solution of the underlying
                # differential equation, predicting future temperature:
                no_heater_temp = round(
                    temp - temploss - self.diffusionloss(temp, t_delta), ROUND
                )
                min_temp = max(
                    watertemp_requirement(tstamp, vacation=vacation, prices=prices_df),
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
                    temp
                    + predict_tempincrease(t_delta)
                    - self.diffusionloss(temp, t_delta),
                    ROUND,
                )
                if min_temp < heater_on_temp < maxtemp:
                    kwh = waterheaterkwh(t_delta)
                    cost = kwh * powerprice
                    # print(f"Heating from {temp} to {heater_on_temp} at cost {cost}")
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

    def diffusionloss(self, temp, t_delta=None):
        """Estimates temperature loss from diffusion
        though a time range.

        Returns positive values.

        Args:
            temp (float): Current temperature

        Returns:
            delta_temp (float): Positive value representing estimated
            temperature loss, in possibly implicit timedelta.
        """
        # assert isinstance(t_delta, pd._libs.tslibs.timedeltas.TimeDelta)
        # hours = float(t_delta.value) / 1e9 / 60 / 60  # from nanoseconds
        # return 0.4 * hours  # This is an eyeballing estimate.

        intercept = self.heatlossdiffusionmodel[0]
        coef = self.heatlossdiffusionmodel[1]
        # (the t_delta is implicit in the diffusion model)
        return -(intercept + coef * temp)


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


def watertemp_requirement(timestamp, vacation=False, prices=None):
    """Todo: the hour for legionella should be after the in the cheapest
    sunday-hour."""
    hour = timestamp.hour
    weekday = timestamp.weekday()  # 0 is Monday, 6 is Sunday
    legionelladay = 6

    if prices is not None and "dayrank" in prices and weekday == legionelladay:
        # Find the cheapest sunday-hour, weekday and dayrank
        # is precomputed in the tibber module.
        cheapest_hours = prices.index[
            (prices["weekday"] == legionelladay) & (prices["dayrank"] == 1)
        ].tolist()
        if cheapest_hours:
            if hour == (cheapest_hours[0] + pd.Timedelta(hours=1)).hour:
                return 80
    if 16 <= hour <= 18:
        return 60  # For doing the dishes
    if vacation:
        return 30
    if hour < 6:
        return 40
    return 50


def predict_tempincrease(t_delta):
    """Predict how much the temperature will increase in the water by pure
    physical formulas, given the wattage of the heater and the time period

    No heatloss diffusion involved.

    Return:
        float: delta temperature.
    """

    # assert isinstance(t_delta, pd._libs.tslibs.timedeltas.TimeDelta)
    hours = float(t_delta.value) / 1e9 / 60 / 60  # from nanoseconds
    return 1 / CV / LITRES * WATTAGE * hours


def waterheaterkwh(t_delta):
    """Returns power usage in kwh"""
    hours = float(t_delta.value) / 1e9 / 60 / 60  # from nanoseconds
    kwh = hours * WATTAGE / 1000
    return kwh


def waterheatercost(powerprice: float, t_delta: pd.Timedelta) -> Tuple[float, float]:
    """Returns cost in NOK"""
    hours = float(t_delta.value) / 1e9 / 60 / 60  # from nanoseconds
    kwh = hours * WATTAGE / 1000
    return powerprice * kwh, kwh


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


async def make_weekly_profile(pers, vacation=False, plot=False):
    dframe = await pers.influxdb.get_series(SENSOR_ITEM)
    dframe.columns = ["watertemp"]
    meanwatertemp = dframe["watertemp"].mean()
    dframe["watertemp"].clip(lower=20, upper=85, inplace=True)
    # Two-stage resampling, finer resolution than 15 minutes from raw data
    # yields too erratic results. So we first resample to 15 min, and then
    # go smoothly to finer scales afterwards if wanted.
    dframe = dframe.resample(pd.Timedelta("60min")).mean().interpolate(method="linear")
    dframe = dframe.resample(PD_TIMEDELTA).mean().interpolate(method="linear")

    # Make the difference pr. timeinterval:
    dframe["waterdiff"] = dframe["watertemp"].diff().shift(-1)

    # Filter away temperature increases, this is when the heater is on
    # and that is not want we want to estimate from.
    dframe = dframe[dframe["waterdiff"] < 0]

    vacation = await pers.influxdb.get_series(VACATION_ITEM)
    vacation = vacation.resample(PD_TIMEDELTA).max().fillna(method="ffill")
    vacation.columns = ["vacation"]

    # Juxtapose the waterdiff and vacation series:
    dframe = pd.concat([dframe, vacation], join="inner", axis=1)

    away_rows = dframe["vacation"] > 0
    dframe_athome = dframe[~away_rows]
    dframe_away = dframe[away_rows]

    (intercept, coef) = make_heatloss_diffusion_model(dframe_away.copy(), plot=False)

    profile = waterusage_weekly(dframe_athome.copy())
    profile.to_csv("watertempprofile.csv")

    if plot:
        profile.plot(y="waterdiff")

    return (profile, (intercept, coef), meanwatertemp)


def waterusage_weekly(dframe, plot=False):
    """Averages a long time-series of waterusage pr. time into a type-profile
    for a week

    (appends columns to incoming dataframe)

    Returns:
        pd.Dataframe, indexed by "day", "hour" and "minute", and
        the value for each being the expected temperature drop (waterusage)
        within that time period
    """
    dframe["day"] = dframe.index.dayofweek
    dframe["hour"] = dframe.index.hour
    dframe["minute"] = dframe.index.minute
    profile = dframe.groupby(["day", "hour", "minute"]).mean()
    if plot:
        profile.plot.line(y="waterdiff")
        pyplot.show()
    return profile


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


async def main():
    # This is typically used for interactive testing.
    pers = pyrotun.persist.PyrotunPersistence()
    # Make the weekly water usage profile, and persist it:
    await pers.ainit(["tibber", "waterheater", "influxdb", "openhab"])
    prices_df = await pers.tibber.get_prices()
    # Grid rental is time dependent:
    prices_df["NOK/KWh"] += localpowerprice.get_gridrental(prices_df.index)

    currenttemp = await pers.openhab.get_item(SENSOR_ITEM, datatype=float)
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
        logger.warning("Water temperature below minimum, should force on")
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
    prices_df["mintemp"] = (
        prices_df.reset_index()["index"]
        .apply(watertemp_requirement, vacation=False, prices=prices_df)
        .values
    )
    prices_df.plot(drawstyle="steps-post", ax=ax, y="mintemp", color="blue", alpha=0.4)

    # Yesterdays temperatures shifted forward by 24h:
    hist_temps = await pers.influxdb.get_series(
        SENSOR_ITEM,
        since=datetime.datetime.now() - datetime.timedelta(hours=48),
    )
    # Smoothen historical curve:
    hist_temps = hist_temps.resample("15min").mean().interpolate(method="time")
    hist_temps.index = hist_temps.index.tz_convert("Europe/Oslo") + datetime.timedelta(
        hours=24
    )
    # ax.plot(
    #   hist_temps.index,
    #   hist_temps[SENSOR_ITEM],
    #    color="green",
    #    label="direct",
    #    alpha=0.7,
    # )

    ax2 = ax.twinx()
    prices_df.plot(drawstyle="steps-post", y="NOK/KWh", ax=ax2, alpha=0.2)

    # ax.set_xlim(left=prices_df.index.min(), right=prices_df.index.max())
    pyplot.show()
    await pers.aclose()


if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main(), debug=False)
