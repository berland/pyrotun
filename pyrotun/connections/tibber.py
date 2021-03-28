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
        self.lastpriceframe_timestamp = None
        self._close_websession_in_aclose = False

    async def _create_session(self):
        return aiohttp.ClientSession(
            headers={aiohttp.hdrs.USER_AGENT: "brlndpyTibber/0.x.x"}
        )

    async def ainit(self, websession=None):
        logger.info("tibberconnections.ainit()")
        self.mytibber = None  # will be set by authenticate()
        self.home = None  # A Tibber home

        self.websession = websession
        if self.websession is None:
            self.websession = aiohttp.ClientSession()
            self._close_websession_in_aclose = True

        self.token = os.getenv("TIBBER_TOKEN")

        if not self.token:
            raise ValueError("Tibber token not found")

        if await self.authenticate():
            self.authenticated = True
            logger.info("Tibber authenticated")

    async def aclose(self):
        await self.mytibber.close_connection()
        if self._close_websession_in_aclose:
            self.websession.close()

    async def authenticate(self):
        if self.authenticated:
            return
        self.websession = await self._create_session()
        self.mytibber = tibber.Tibber(self.token, timeout=2, websession=self.websession)
        try:
            await self.mytibber.update_info()
            self.home = self.mytibber.get_homes()[0]
            await self.home.update_info()
            self.authenticated = True
        except asyncio.TimeoutError:
            self.authenticated = False

    async def get_prices(self):
        """Prices in the dataframe is valid *forwards* in time"""
        tz = pytz.timezone(os.getenv("TIMEZONE"))
        if (
            self.lastpriceframe_timestamp is not None
            and datetime.datetime.now() - self.lastpriceframe_timestamp
            < datetime.timedelta(hours=1)
        ):
            return self.lastpriceframe
        try:
            if not self.authenticated:
                logger.debug("Reauth in tibber")
                await self.authenticate()
            await self.home.update_price_info()
        except (asyncio.TimeoutError, AttributeError):
            logger.warning("Timeout connecting to Tibber")
            if self.lastpriceframe is not None:
                logger.warning("Using internal cache for price_df")
                return self.lastpriceframe
            else:
                logger.warning("Using on-disk frame for price_df")
                disk_frame = pd.read_csv(
                    "/var/tmp/tibber_lastpriceframe.csv", index_col=0
                )
                disk_frame.index = pd.to_datetime(
                    disk_frame.index, utc=True
                ).tz_convert(tz)

                return disk_frame
        prices_df = pd.DataFrame.from_dict(self.home.price_total, orient="index")
        prices_df.columns = ["NOK/KWh"]
        prices_df.index = pd.to_datetime(prices_df.index, utc=True).tz_convert(tz)
        prices_df["weekday"] = prices_df.index.weekday
        prices_df["dayrank"] = (
            prices_df.groupby("weekday")["NOK/KWh"].rank().astype(int)
        )
        prices_df.to_csv("/var/tmp/tibber_lastpriceframe.csv")
        self.lastpriceframe = prices_df
        self.lastpriceframe_timestamp = datetime.datetime.now()
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
        nowprice = prices.loc[nowhour, "NOK/KWh"] * 100
        if "dayrank" in prices:
            priceorder = prices.loc[nowhour, "dayrank"]
            relpriceorder = round(priceorder / float(len(prices)), 2)
        else:
            priceorder = None
            relpriceorder = None

        return (nowprice, priceorder, relpriceorder)
