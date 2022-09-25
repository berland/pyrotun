import asyncio
import logging
import os
from pathlib import Path

import aiohttp
import websockets
import yaml
from websockets.datastructures import Headers

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)


SERVER = "sdk.iotiliti.cloud"
BASE_URL = f"https://{SERVER}/homely/"
CONFIG_FILE = "homelyopenhab.yml"
logger.setLevel(logging.DEBUG)


class HomelyConnection:
    def __init__(self, websession=None):
        self.websession = websession
        self.access_token = None
        self.location_id = None
        self.config = None  # Defines the link to OpenHAB

        self.websocket = None
        self._close_websession_in_aclose = False

    async def ainit(self, websession=None):

        if websession is not None:
            self.websession = websession

        if self.websession is None:
            self.websession = aiohttp.ClientSession()
            self._close_websession_in_aclose = True

        if self.access_token is None:
            await self.acquire_token()

        if self.location_id is None:
            await self.acquire_location_id()

        if self.config is None:
            await self.acquire_config()

    async def aclose(self):
        logger.info("closing")
        if self._close_websession_in_aclose:
            await self.websession.close()

    async def acquire_config(self):
        config_file = Path(__file__).parent.parent / CONFIG_FILE
        if config_file.exists():
            self.config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        else:
            logger.warning(f"{CONFIG_FILE} not found, using dummy")
            # Example config
            self.config = [
                {
                    "name": "Røyksensor gang oppe",
                    "openhab_item": "Sensor_Gangoppe_taktemperatur",
                    "featurepath": "temperature/states/temperature/value",
                    "type": float,
                },
                {
                    "name": "Røyksensor gang oppe",
                    "openhab_item": "Sensor_Gangoppe_brann",
                    "featurepath": "alarm/states/fire/value",
                    "type": bool,
                },
            ]

    async def acquire_token(self) -> None:
        login_body = {
            "username": os.getenv("HOMELY_USER"),
            "password": os.getenv("HOMELY_PW"),
        }
        async with self.websession.post(
            BASE_URL + "oauth/token", json=login_body
        ) as resp:
            json_response = await resp.json()
            self.access_token = json_response["access_token"]
            logger.info("Acquired homely token")

    async def acquire_location_id(self, index=0):
        async with self.websession.get(
            BASE_URL + "locations",
            headers={"Authorization": f"Bearer {self.access_token}"},
        ) as resp:
            locations = await resp.json()
            self.location_id = locations[index]["locationId"]

    async def get_data(self) -> dict:
        logger.info("Polling for all homely data...")
        response = None
        while response is None or response.status == 401:
            response = await self.websession.get(
                BASE_URL + f"home/{self.location_id}",
                headers={"Authorization": f"Bearer {self.access_token}"},
            )
            if response.status == 401:
                logger.info("Need to reauthenticate with homely")
                await asyncio.sleep(1)
                await self.acquire_token()
                await self.acquire_location_id()
            else:
                data = await response.json()
                return data

    async def setup_websocket(self):
        _e_headers = Headers()
        _e_headers["token"] = f"Bearer {self.access_token}"
        _e_headers["locationId"] = self.location_id
        async with websockets.connect(
            # f"wss://{server}:443",
            f"wss://{SERVER}",
            # open_timeout=None,
            extra_headers=_e_headers
            # {
            # "locationId": location_id,
            # "token": f"Bearer {access_token}",
            # },
        ) as websocket:
            print(websocket)
            await websocket.recv()


def find(path: str, data: dict):
    value = data
    for key in path.split("/"):
        value = value[key]
    return value
