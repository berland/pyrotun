import os
import asyncio
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
        self.lastpriceframe = None

    async def _create_session(self):
        return aiohttp.ClientSession(
            headers={aiohttp.hdrs.USER_AGENT: f"brlndpyTibber/0.x.x"}
        )

    async def ainit(self, websession=None):
        logger.info("tibberconnections.ainit()")
        self.mytibber = None  # will be set by authenticate()
        self.home = None  # A Tibber home

        self.websession = websession
        if self.websession is None:
            self.websession = aiohttp.ClientSession()

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
        try:
            await self.mytibber.update_info()
            self.home = self.mytibber.get_homes()[0]
            await self.home.update_info()
        except asyncio.TimeoutError:
            self.authenticated = False

    async def get_prices(self):
        """Prices in the dataframe is valid *forwards* in time"""
        try:
            if not self.authenticated:
                await self.authenticate()
            await self.home.update_price_info()
        except asyncio.TimeoutError:
            logger.warning("Timeout connecting to Tibber")
            if self.lastpriceframe is not None:
                logger.warning("Using internal cache for price_df")
                return self.lastpriceframe
            else:
                logger.warning("Using on-disk frame for price_df")
                return pd.read_csv("/var/run/tibber_lastpriceframe.csv")
        tz = pytz.timezone(os.getenv("TIMEZONE"))
        prices_df = pd.DataFrame.from_dict(self.home.price_total, orient="index")
        prices_df.columns = ["NOK/KWh"]
        prices_df.index = pd.to_datetime(prices_df.index).tz_convert(tz)
        prices_df.to_csv("/var/run/tibber_lastpriceframe.csv")
        self.lastpriceframe = prices_df
        self.lastpriceframe_update = datetime.datetime.now()
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
        return prices.loc[nowhour, "NOK/KWh"] * 100
