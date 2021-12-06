import datetime
import os

import pandas as pd
import pytz
import requests
import smappy

import pyrotun

logger = pyrotun.getLogger(__name__)


class SmappeeConnection:
    def __init__(self):

        self.mysmappee = None  # will be set by authenticate()
        self.locationid = None

        self.user = os.getenv("SMAPPEE_USER")
        self.token = os.getenv("SMAPPEE_TOKEN")
        self.pword = os.getenv("SMAPPEE_PW")

        if not (self.user and self.token and self.pword):
            raise ValueError("Smappee credentials not found")

        if self.authenticate():
            self.authenticated = True
            logger.info("Smappee authenticated")

        if self.authenticated:
            # Get a location ID, an integer for where the account is registered (?)
            self.locationid = self.mysmappee.get_service_locations()[
                "serviceLocations"
            ][0]["serviceLocationId"]

    def authenticate(self):
        self.mysmappee = smappy.Smappee(self.user, self.token)
        response = self.mysmappee.authenticate(self.user, self.pword)
        if response:
            return True
        return False

    def get_recent_df(self, minutes, aggregation=1):

        tz = pytz.timezone(os.getenv("TIMEZONE"))
        now = datetime.datetime.now().astimezone(tz)
        earlier = now - datetime.timedelta(minutes=minutes)
        # How to test if connection is active?

        try:
            dframe = self.mysmappee.get_consumption_dataframe(
                self.locationid, earlier, now, aggregation
            )
        except requests.exceptions.HTTPError:
            logger.error("Smappee cloud did not respond")
            return pd.DataFrame()

        if dframe.empty:
            self.authenticate()
            dframe = self.mysmappee.get_consumption_dataframe(
                self.locationid, earlier, now, aggregation
            )
            if dframe.empty:
                logger.error("Empty smappee dataframe in get_recent_df()")
            else:
                dframe.index = dframe.index.tz_convert(tz)
        return dframe

    def avg_watt_5min(self):
        # Need to go more than 5 minutes back in time, otherwise we
        # risk getting nothing in return.
        lastrow = self.get_recent_df(minutes=15).tail(1)
        print(lastrow)
        if lastrow.empty:
            return None
        avgwatt = (60 / 5) * float(lastrow["consumption"].values)
        return avgwatt

    def get_daily_df(self, aggregation="1hour"):
        """Get dataframe of consumption since midnight

        Aggregated pr. hour

        Returns:
            pd.DataFrame

        """
        agg_integer = {"1hour": 2, "5min": 1}[aggregation]
        self.mysmappee.re_authenticate()
        tz = pytz.timezone(os.getenv("TIMEZONE"))
        now = datetime.datetime.now().astimezone(tz)
        midnight = datetime.datetime.combine(
            datetime.date.today(), datetime.time.min
        ).astimezone(tz)

        dframe = self.mysmappee.get_consumption_dataframe(
            self.locationid, midnight, now, agg_integer
        )
        dframe.drop(
            ["solar", "gridExport", "selfConsumption", "gridImport", "selfSufficiency"],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        if dframe.empty:
            logger.error("Empty smappee dataframe in get_recent_df()")
            return pd.DataFrame()
        dframe.index = dframe.index.tz_convert(tz)
        return dframe

    def get_daily_cum(self):
        """Compute KWh up until now since midnight

        Returns:
            float
        """
        dframe = self.get_daily_df(aggregation="5min")
        if dframe.empty:
            return None
        return round(dframe["consumption"].sum() / 1000, 2)
