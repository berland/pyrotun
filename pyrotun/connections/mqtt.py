import os

import dotenv

import asyncio_mqtt
import pyrotun

logger = pyrotun.getLogger(__name__)

dotenv.load_dotenv()

assert os.getenv("MQTT_HOST"), "You must provide MQTT_HOST as env variable"


class MqttConnection:
    def __init__(self, host="", port=1883):

        if not host:
            self.host = os.getenv("MQTT_HOST")
        else:
            self.host = host

        self.client = asyncio_mqtt.Client(self.host)

        if not self.host:
            raise ValueError("MQTT_HOST not provided")

    async def ainit(self):
        logger.info("Connecting to MQTT server %s", self.host)
        await self.client.__aenter__()  # yes yes, should have used context manager

    async def aclose(self):
        await self.cleint.__aexit__()
