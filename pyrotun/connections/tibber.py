import os
import pytz
import aiohttp
import pandas as pd
import tibber

import pyrotun

logger = pyrotun.getLogger(__name__)


class TibberConnection:
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
        self.websession = await self._create_session()
        self.mytibber = tibber.Tibber(self.token, websession=self.websession)
        await self.mytibber.update_info()
        self.home = self.mytibber.get_homes()[0]
        await self.home.update_info()

    async def get_prices(self):
        await self.home.update_price_info()
        return pd.DataFrame.from_dict(self.home.price_total, orient="index")
