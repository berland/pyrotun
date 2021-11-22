import asyncio
import datetime
import os

import aiohttp
import asyncsector
import pandas as pd

import pyrotun

logger = pyrotun.getLogger(__name__)


class SectorAlarmConnection:
    """Setup and maintain a connection to SectorAlarm web API"""

    def __init__(self):

        self.authenticated = None
        self.username = os.getenv("SECTORALARM_USER")
        self.pword = os.getenv("SECTORALARM_PW")
        self.siteid = os.getenv("SECTORALARM_SITEID")
        self.apiversion = os.getenv("SECTORALARM_APIVERSION")

        self.asyncsector = None
        self.websession = None
        self._close_websession_in_aclose = False

        if not (self.username and self.pword and self.siteid):
            raise ValueError("SectorAlarm credentials not found")

    async def ainit(self, websession):
        """Log in to the webpage"""
        logger.info("sectoralarm.ainit()")
        self.websession = websession

        if self.websession is None:
            self.websession = aiohttp.ClientSession()
            self._close_websession_in_aclose = True

        self.asyncsector = asyncsector.AsyncSector(
            self.websession, self.siteid, self.username, self.pword, self.apiversion
        )

        if await self.authenticate():
            self.authenticated = True
            logger.info("SectorAlarm authenticated")

    async def authenticate(self):
        if self.authenticated:
            return
        return await self.asyncsector.login()

    async def history_df(self):
        """Return a dataframe for the arm/lock history"""
        try:
            hist = await self.asyncsector.get_history()
        except aiohttp.client_exceptions.ClientConnectorError:
            logger.error("Failed to connect to sectoralarm, will retry logging in")
            hist = None
        if hist is None:
            # Something bad has happened, maybe logged out?
            self.authenticated = False
            await asyncio.sleep(10)  # Give them some slack..
            await self.authenticate()
            hist = await self.asyncsector.get_history()
        hist_df = pd.DataFrame(hist["LogDetails"])

        def fix_time(sectoralarm_datestr):
            return datetime.datetime.fromtimestamp(
                int(sectoralarm_datestr.replace("/Date(", "").replace(")/", ""))
                / 1000.0
            )

        hist_df.index = hist_df["Time"].apply(fix_time)
        del hist_df["Time"]
        return hist_df

    async def armed(self, hist=None):
        """Determine if current status is armed or not"""
        if hist is None:
            hist = await self.history_df()
        return (
            hist[hist["EventType"].str.endswith("armed")]["EventType"].values[0]
            == "armed"
        )

    async def locked(self, hist=None):
        """Determine if door is locked right now"""
        if hist is None:
            hist = await self.history_df()
        return (
            hist[hist["EventType"].str.endswith("lock")]["EventType"].values[0]
            == "lock"
        )

    async def aclose(self):
        """Tear down object"""
        if self._close_websession_in_aclose:
            self.websession.close()
