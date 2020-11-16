import os
import datetime

import pytz
import aiohttp
import pandas as pd
import tibber

import pyrotun

logger = pyrotun.getLogger(__name__)


class TibberConnection:
    def __init__(self):
        self.home = None
        self.authenticated = False

    async def _create_session(self):
        return aiohttp.ClientSession(
            headers={aiohttp.hdrs.USER_AGENT: f"brlndpyTibber/0.x.x"}
        )

    async def ainit(self):

        self.mytibber = None  # will be set by authenticate()
        self.home = None  # A Tibber home

        self.websession = None
        self.token = os.getenv("TIBBER_TOKEN")

        if not self.token:
            raise ValueError("Tibber token not found")

        if await self.authenticate():
            self.authenticated = True
            logger.info("Tibber authenticated")

    async def authenticate(self):
        if self.authenticated:
            return
        self.websession = await self._create_session()
        self.mytibber = tibber.Tibber(self.token, websession=self.websession)
        await self.mytibber.update_info()
        self.home = self.mytibber.get_homes()[0]
        await self.home.update_info()

    async def get_prices(self):
        """Prices in the dataframe is valid *forwards* in time"""
        #if not self.authenticated:
        #    await self.authenticate()
        await self.home.update_price_info()
        tz = pytz.timezone(os.getenv("TIMEZONE"))
        prices_df = pd.DataFrame.from_dict(self.home.price_total, orient="index")
        prices_df.columns = ["NOK/KWh"]
        prices_df.index = pd.to_datetime(prices_df.index).tz_convert(tz)
        return prices_df

    async def get_currentprice(self):
        """Get the current power price in øre/kwh"""
        tz = pytz.timezone(os.getenv("TIMEZONE"))
        nowhour = pd.to_datetime(
            datetime.datetime.now()
            .astimezone(tz)
            .replace(minute=0, second=0, microsecond=0)
        )

        prices = await self.get_prices()
        return prices.loc[nowhour, "NOK/KWh"]*100
