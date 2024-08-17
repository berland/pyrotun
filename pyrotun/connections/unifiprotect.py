import os

import aiohttp
import dotenv
import pyunifiprotect
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
        self.protect: ProtectApiClient = None
        self.websession = None
        self._close_websession_in_aclose = None
        self.ws_sub = None

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
        try:
            await self.protect.update()  # Sets up websocket
        except pyunifiprotect.exceptions.NotAuthorized:
            pass

    async def aclose(self):
        # Close ws subscription:
        self.ws_sub()

        if self._close_websession_in_aclose:
            self.websession.close()
