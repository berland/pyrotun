import os

import aiohttp
import dotenv
from pyunifiprotect import ProtectApiClient

import pyrotun

logger = pyrotun.getLogger(__name__)


dotenv.load_dotenv()
UNIFI_HOST = os.getenv("UNIFI_HOST")
UNIFI_PORT = os.getenv("UNIFI_PORT")
UNIFI_USERNAME = os.getenv("UNIFI_USERNAME")
UNIFI_PASSWORD = os.getenv("UNIFI_PASSWORD")


class UnifiProtectConnection:
    def __init__(self):
        self.protect = None
        self.websession = None
        self._close_websession_in_aclose = None

    async def ainit(self, websession=None):
        logger.info("unifiprotectconnection.ainit()")

        self.websession = websession
        if self.websession is None:
            self.websession = aiohttp.ClientSession()
            self._close_websession_in_aclose = True

        self.protect = ProtectApiClient(
            UNIFI_HOST,
            UNIFI_PORT,
            UNIFI_USERNAME,
            UNIFI_PASSWORD,
            verify_ssl=False,
            session=self.websession,
        )
        await self.protect.update()  # Sets up websocket

    async def aclose(self):
        if self._close_websession_in_aclose:
            self.websession.close()
