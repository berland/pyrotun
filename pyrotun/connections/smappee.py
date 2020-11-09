import os
import datetime
import pytz

import smappy


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
        avgwatt = (60 / 5) * float(lastrow["consumption"].values)
        return avgwatt

    def get_daily_cum(self):
        tz = pytz.timezone(os.getenv("TIMEZONE"))
        now = datetime.datetime.now().astimezone(tz)
        midnight = datetime.datetime.combine(
            datetime.date.today(), datetime.time.min
        ).astimezone(tz)

        dframe = self.mysmappee.get_consumption_dataframe(
            self.locationid, midnight, now, 1
        )
        if dframe.empty:
            logger.error("Empty smappee dataframe in get_recent_df()")
            return

        dframe.index = dframe.index.tz_convert(tz)

        # always_on_cum = round(dframe["alwaysOn"].sum() / 1000, 2)

        # consumption is in Wh, and pr. the timedelta in the index.
        return round(dframe["consumption"].sum() / 1000, 2)
