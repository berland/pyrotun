import asyncio
import networkx
import datetime
from sklearn import linear_model
import pandas as pd
from matplotlib import pyplot

import pyrotun
from pyrotun import persist

logger = pyrotun.getLogger(__name__)

TIMEDELTA_MINUTES = 30  # 60 is smooth enough. 30 and lower is too noisy.
PD_TIMEDELTA = str(TIMEDELTA_MINUTES) + "min"
SENSOR_ITEM = "Varmtvannsbereder_temperatur"
VACATION_ITEM = "Ferie"


class WaterHeater:
    def __init__(self):
        self.weekly_water_usage_frame = None

        # The global persistence object
        self.pers = None

    async def ainit(self, pers):
        self.pers = pers
        self.weekly_water_usage_frame = await weekly_profile(pers)

    async def future_temp_cost_graph():
        """Build the networkx Directed 2D ~lattice  Graph, with
        datetime on the x-axis and water-temperatur on the y-axis.

        Edges from nodes determined by (time, temp) has an associated
        cost in NOK.
        """
        prices_df = self.pers.tibber.get_prices_df()
        datetimes = pd.date_range(datetime.datetime.now(),
                prices_df.index.max(), freq=PD_TIMEDELTA, tz=tz)


async def waterusage_weekly(dframe, plot=False):
    dframe["day"] = dframe.index.dayofweek
    dframe["hour"] = dframe.index.hour
    dframe["minute"] = dframe.index.minute
    profile = dframe.groupby(["day", "hour", "minute"]).mean()
    if plot:
        profile.plot.line(y="waterdiff")
        pyplot.show()
    return profile


async def heatloss_diffusion_model(dframe_away, plot=False):
    """Estimate how fast temperature drops when nothing but heat
    diffusion is at play.

    Returns a linear model where the coefficient reveals
    how much the temperature decreases (in the implicit time interval) for
    one degree extra water temperature.

    """

    async def linearmodel(dframe, xvecs, yvec):
        model = linear_model.LinearRegression().fit(dframe[xvecs], dframe[yvec])
        return (model.intercept_, model.coef_[0])

    # Do we have a correlation between waterdiff and watertemp?
    vetted_rows = (dframe_away["watertemp"] < 75) & (dframe_away["waterdiff"] > -1)
    (intercept, coef) = await linearmodel(
        dframe_away[vetted_rows], ["watertemp"], "waterdiff"
    )

    # Difference should increase with increasing temperature..
    if plot:
        ax = dframe_away.plot.scatter(x="watertemp", y="waterdiff")
        dframe_away["linestdiff"] = intercept + dframe_away["watertemp"] * coef
        ax = dframe_away.plot.scatter(x="watertemp", y="linestdiff", color="red", ax=ax)
        pyplot.show()
    return (intercept, coef)






async def weekly_profile(pers, vacation=False):
    dframe = await pers.influxdb.get_series(SENSOR_ITEM)
    dframe.columns = ["watertemp"]
    dframe["watertemp"].clip(lower=20, upper=85, inplace=True)
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

    (intercept, coef) = await heatloss_diffusion_model(dframe_away.copy(), plot=False)

    profile = await waterusage_weekly(dframe_athome.copy(), plot=False)
    print(profile)
    profile.to_csv("watertempprofile.csv")
    # print(watertemps)
    # print(d_watertemps)


async def main():
    pers = pyrotun.persist.PyrotunPersistence()
    # Make the weekly water usage profile, and persist it:
    await pers.ainit() # This also initializes this very waterheater object.
    print(pers.waterheater.weekly_water_usage_frame)
    graph = await pers.waterheater.future_temp_cost_graph()

    #temp = await pers.influxdb.get_lastvalue("InneTemperatur")
    #print(temp)
    #prices = await pers.tibber.get_prices()
#
#    print(prices)

#    await weekly_profile(connections)


if __name__ == "__main__":
    asyncio.run(main())
