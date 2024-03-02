import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Awaitable, Dict, Optional

import aiohttp
import yaml

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)
logger.setLevel(logging.DEBUG)


SERVER = "api.myuplink.com"
BASE_URL = f"https://{SERVER}/"
CONFIG_FILE = "myuplink_openhab.yml"


class MyuplinkConnection:
    def __init__(self, websession=None, _message_handler: Optional[Awaitable] = None):
        self.websession = websession
        self.access_token = None
        self.expires_in = 1000

        self._close_websession_in_aclose = False

        self.config: Optional[Dict[str, str]] = None  # Defines link to OpenHAB
        self.vvb_device_id = None
        self.parameterids: Optional[Dict[str, str]]

    async def ainit(self, websession=None):
        if websession is not None:
            self.websession = websession

        if self.websession is None:
            self.websession = aiohttp.ClientSession()
            self._close_websession_in_aclose = True

        if self.access_token is None:
            await self.acquire_token()
            asyncio.create_task(self.token_manager())

        if self.config is None:
            await self.acquire_config()

        result = await self.get("v2/systems/me")
        if result:
            self.vvb_device_id = json.loads(result)["systems"][0]["devices"][0]["id"]

        result = await self.get(f"v2/devices/{self.vvb_device_id}/points")
        if result:
            self.parameterids = {
                point["parameterName"]: point["parameterId"]
                for point in json.loads(result)
                if point["writable"]
            }

    async def acquire_config(self):
        config_file = Path(__file__).parent.parent / CONFIG_FILE
        if config_file.exists():
            rawconfig = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            self.config = {
                item["myuplinkname"]: item["openhab_item"] for item in rawconfig
            }

    async def aclose(self):
        assert self.websession is not None
        if self._close_websession_in_aclose:
            await self.websession.close()

    async def acquire_token(self) -> None:
        login_data = {
            "client_id": os.getenv("MYUPLINK_CLIENT_ID"),
            "client_secret": os.getenv("MYUPLINK_CLIENT_SECRET"),
            "grant_type": "client_credentials",
        }
        assert self.websession is not None
        async with self.websession.post(
            BASE_URL + "oauth/token", data=login_data
        ) as resp:
            json_response = await resp.json()
            self.access_token = json_response["access_token"]
            self.expires_in = int(json_response["expires_in"])
            logger.info("Acquired myuplink token")

    async def token_manager(self) -> None:
        assert self.websession is not None
        while True:
            await asyncio.sleep(int(self.expires_in / 1.5))
            await self.acquire_token()

    async def get(self, endpoint: str) -> Optional[str]:
        assert self.websession is not None
        assert self.access_token is not None
        async with self.websession.get(
            BASE_URL + endpoint,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "accept": "text/plain",
            },
        ) as resp:
            if resp.status != 200:
                print(resp.status)
                return None
            text_response = await resp.text()
            return text_response

    async def patch(self, endpoint: str, data: dict):
        assert self.websession is not None
        assert self.access_token is not None
        async with self.websession.patch(
            BASE_URL + endpoint,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            data=json.dumps(data),
        ) as resp:
            if resp.status != 200:
                text_response = await resp.text()
                print(text_response)

    async def send_to_myuplink(self, parametername, value):
        assert self.parameterids is not None
        await self.patch(
            f"v2/devices/{self.vvb_device_id}/points",
            data={self.parameterids[parametername]: value},
        )
