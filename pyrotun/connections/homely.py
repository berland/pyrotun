import asyncio
import logging
import os
import pprint
from pathlib import Path

import aiohttp
# import socketio
import yaml
from websockets.datastructures import Headers
from typing import Awaitable, Optional
import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SIO = None # socketio.AsyncClient(logger=logger, reconnection_attempts=3)

SERVER = "sdk.iotiliti.cloud"
BASE_URL = f"https://{SERVER}/homely/"
CONFIG_FILE = "homelyopenhab.yml"


class HomelyConnection:
    def __init__(self, websession=None, message_handler: Optional[Awaitable] = None):
        self.websession = websession
        self.message_handler = message_handler
        self.access_token = None
        self.refresh_token = None
        self.location_id = None
        self.config = None  # Defines the link to OpenHAB

        self.websocket_client = None
        self._close_websession_in_aclose = False

    async def ainit(self, websession=None):

        if websession is not None:
            self.websession = websession

        if self.websession is None:
            self.websession = aiohttp.ClientSession()
            self._close_websession_in_aclose = True

        if self.access_token is None:
            await self.acquire_token()
            asyncio.create_task(self.token_manager())

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
            self.refresh_token = json_response["refresh_token"]
            logger.info("Acquired homely token")

    async def token_manager(self) -> None:
        """Run as a task"""
        while True:
            await asyncio.sleep(1000)
            async with self.websession.post(
                BASE_URL + "oauth/refresh-token",
                json={"refresh_token": self.refresh_token},
            ) as resp:
                json_response = await resp.json()
                self.access_token = json_response["access_token"]
                self.refresh_token = json_response["refresh_token"]
                logger.info("Refreshed homely token")

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

    async def run_websocket(self):
        disconnects = 0
        @SIO.event
        async def connect():
            logger.info("Connected to homely websocket server")
            if disconnects > 0:
                logger.warning(f"(after {disconnects} disconnects")

        @SIO.event
        async def disconnect():
            logger.warning("Disconnected to homely websocket server")
            disconnects += 1

        @SIO.on("event")
        async def on_message(data):
            if self.message_handler is not None:
                await self.message_handler(data)
            else:
                pprint.pprint(data)

        url = (
            f"https://{SERVER}"
            f"?locationId={self.location_id}"
            f"&token=Bearer%20{self.access_token}"
        )
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "locationId": self.location_id,
        }
        await SIO.connect(url, headers)
        await SIO.wait()
