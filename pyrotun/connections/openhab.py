import aiohttp
import os
import openhab

import pyrotun

logger = pyrotun.getLogger(__name__)


class OpenHABConnection:
    def __init__(self, openhab_url="", websession=None):

        if not openhab_url:
            self.openhab_url = os.getenv("OPENHAB_URL")
        else:
            self.openhab_url = openhab_url

        if not self.openhab_url:
            raise ValueError("openhab_url not set")

        if websession is not None:
            self.websession = websession
        else:
            # For async usage.
            self.websession = aiohttp.ClientSession()

        # This is not async..
        self.client = openhab.openHAB(self.openhab_url)

    async def get_item(self, item_name):
        async with self.websession.get(
            self.openhab_url + "/items/" + str(item_name)
        ) as resp:
            resp = await resp.json()
            return resp["state"]

    async def set_item(self, item_name, new_state, log=None):
        if log:
            logger.info("OpenHAB: Setting %s to %s", item_name, str(new_state))
        async with self.websession.post(
            self.openhab_url + "/items/" + str(item_name), data=str(new_state)
        ) as resp:
            if resp.status != 200:
                logger.error(resp)

    def sync_get_item(self, item_name):
        return self.client.get_item(item_name).state

    def sync_set_item(self, item_name, new_state, log=False):
        if log:
            logger.info("OpenHAB: Setting %s to %s", item_name, str(new_state))
        self.client.get_item(item_name).state = new_state
