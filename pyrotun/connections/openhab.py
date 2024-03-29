import os

import aiohttp
import openhab

import pyrotun

logger = pyrotun.getLogger(__name__)


class OpenHABConnection:
    def __init__(self, openhab_url="", websession=None, readonly=False):
        self.readonly = readonly

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
        self.client = openhab.OpenHAB(self.openhab_url)

    async def get_item(self, item_name, datatype=str, backupvalue=None):
        assert self.openhab_url is not None
        async with self.websession.get(
            self.openhab_url + "/items/" + str(item_name)
        ) as response:
            resp = await response.json()
            if datatype == str:
                if "state" not in resp:
                    raise KeyError(f"{item_name} not found in response {resp}")
                return resp["state"]
            elif datatype == float:
                try:
                    return float(resp["state"])
                except ValueError:
                    logger.error(f"{item_name} was UNDEF")
                    return backupvalue
            elif datatype == bool:
                return resp["state"] == "ON"

    async def set_item(
        self, item_names, new_state, log=None, method="post", send_no_change=True
    ):
        if self.readonly:
            logger.info(
                "OpenHAB: Would have set %s to %s", str(item_names), str(new_state)
            )
            return
        if not isinstance(item_names, list):
            item_names = [item_names]
        for item_name in item_names:
            current_state = await self.get_item(item_name)
            if str(current_state) == str(new_state) and send_no_change is True:
                logger.debug(
                    "OpenHAB: No change in %s, value %s, still sending command",
                    item_name,
                    str(new_state),
                )
                # return  # If we don't push new commands, items will expire :(
            if log is True:
                logger.info("OpenHAB: Setting %s to %s", item_name, str(new_state))
            if log == "change" and str(current_state) != str(new_state):
                logger.info(
                    "OpenHAB: Changing %s from %s to %s",
                    item_name,
                    str(current_state),
                    str(new_state),
                )
            if method == "post":
                async with self.websession.post(
                    self.openhab_url + "/items/" + str(item_name), data=str(new_state)
                ) as resp:
                    if resp.status != 200:
                        logger.error(resp)
            elif method == "put":
                async with self.websession.put(
                    self.openhab_url + "/items/" + str(item_name) + "/state",
                    data=str(new_state),
                ) as resp:
                    if resp.status != 202:
                        logger.error(resp)
            # try:
            #     # Spare OpenHAB instance for a transition period:
            #     extra_host = self.openhab_url.replace("serv", "serve").replace(
            #         "8090", "8080"
            #     )
            #      async with self.websession.post(
            #         extra_host + "/items/" + str(item_name), data=str(new_state)
            #      ) as resp:
            #         if resp.status != 200:
            #             pass
            #             # logger.error(resp)
            # except OSError:  # as err:
            #     # logger.warning("Secondary OpenHAB instance not responding")
            #     # logger.warning(str(err))
            #     pass

    def sync_get_item(self, item_name):
        return self.client.get_item(item_name).state

    def sync_set_item(self, item_name, new_state, log=False):
        if self.readonly:
            logger.info("OpenHAB: Would have set %s to %s", item_name, str(new_state))
            return
        if log:
            logger.info("OpenHAB: Setting %s to %s", item_name, str(new_state))
        self.client.get_item(item_name).state = new_state
