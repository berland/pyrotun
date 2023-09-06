import asyncio
import logging
import os
from pathlib import Path

import aiohttp
import dotenv
import skodaconnect
import yaml

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

dotenv.load_dotenv()

logger.setLevel(logging.DEBUG)

CONFIG_FILE = "skodaopenhab.yml"

COMPONENTS = {
    "sensor": "sensor",
    "binary_sensor": "binary_sensor",
    "lock": "lock",
    "device_tracker": "device_tracker",
    "switch": "switch",
}


class SkodaConnection:
    def __init__(self, websession=None):
        self.websession = websession
        self.logged_in = False
        self._close_websession_in_aclose = False
        self.skodaconnect = None
        self.instruments = None

        self.config = None  # Mapping to OpenHAB

    async def ainit(self, websession=None):
        if websession is not None:
            self.websession = websession

        if self.websession is None:
            self.websession = aiohttp.ClientSession(
                headers={"Connection": "keep-alive"}
            )
            self._close_websession_in_aclose = True

        if self.skodaconnect is None:
            self.skodaconnect = skodaconnect.Connection(
                self.websession,
                os.getenv("SKODACONNECT_USERNAME"),
                os.getenv("SKODACONNECT_PASSWORD"),
                False,
            )

        self.logged_in = await self.login()

        if self.config is None:
            await self.acquire_config()

    async def login(self) -> bool:
        try:
            self.logged_in = await self.skodaconnect.doLogin()
            logger.info("Skoda authenticated")
            await self.skodaconnect.get_vehicles()
            self.instruments = self.get_instruments()
            return True
        except skodaconnect.exceptions.SkodaException as ex:
            logger.error(f"Skoda login unsuccessful: {ex}")
            return False

    def get_instruments(self) -> set:
        instruments = set()
        for vehicle in self.skodaconnect.vehicles:
            dashboard = vehicle.dashboard(mutable=True, miles=False)
            for instrument in (
                instrument
                for instrument in dashboard.instruments
                if instrument.component in COMPONENTS
                # and is_enabled(instrument.slug_attr)
            ):
                instruments.add(instrument)
        return instruments

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
            self.config = {"battery_level": "EnyaqBatteryState"}

    async def get_data(self):
        logger.info("Polling for all Skoda data...")
        await self.skodaconnect.update_all()

    async def post_to_openhab(self):
        await self.get_data()
        logger.info("Posting Skoda data to OpenHAB")
        for instrument in self.instruments:
            print(f"{instrument.attr} {instrument.state}")
            if instrument.attr in self.config:
                print(f" - posting to {self.config[instrument.attr]}")


async def main():
    skoda = SkodaConnection()
    await skoda.ainit()
    await skoda.post_to_openhab()
    await skoda.aclose()


if __name__ == "__main__":
    asyncio.run(main())
