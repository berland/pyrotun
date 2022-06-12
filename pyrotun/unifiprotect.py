import asyncio
import os
from pathlib import Path
from typing import Dict, Union

import aiofiles
import dotenv
from pyunifiprotect import ProtectApiClient
from pyunifiprotect.data import WSSubscriptionMessage

import pyrotun
from pyrotun import persist

logger = pyrotun.getLogger(__name__)
dotenv.load_dotenv()
UNIFI_HOST = os.getenv("UNIFI_HOST")
UNIFI_PORT = os.getenv("UNIFI_PORT")
UNIFI_USERNAME = os.getenv("UNIFI_USERNAME")
UNIFI_PASSWORD = os.getenv("UNIFI_PASSWORD")

CAMERA_FILENAME: Path = Path("/etc/openhab/html/garagecamera.png")

# Global variable to make it available to the websocket
# callback
PERS = None

# Map status items from the websocket events from unifi over
# to OpenHAB switch items, inverted if needed.
SWITCH_ITEMS: Dict[str, Dict[str, Union[str, bool]]] = {
    "is_dark": {"item": "Sensor_Garasje_Opplyst", "inverted": True},
    "is_motion_detected": {
        "item": "Sensor_Garasje_bevegelse",
        "inverted": False,
    },
}


async def fetch_snapshot(protect: ProtectApiClient, filename: Path) -> None:
    if not filename.parent.exists():
        filename = Path(filename.name)
    camera_id = list(protect.bootstrap.cameras.keys())[0]
    pngbytes = await protect.get_camera_snapshot(camera_id)
    if pngbytes:
        async with aiofiles.open(filename, "wb") as filehandle:
            await filehandle.write(pngbytes)
            await filehandle.flush()
            logger.info(f"Wrote {len(pngbytes)} bytes of PNG to {filename}")
    else:
        logger.error("fGot {len(bytes) bytes from camera")


def process_camera_event(message: WSSubscriptionMessage):
    """This function is called whenever an event (a message) occurs
    on the websocket from unifiprotect."""
    for key in SWITCH_ITEMS.keys():
        if key in message.changed_data:
            value = bool(message.changed_data[key])
            if SWITCH_ITEMS[key].get("inverted", False):
                value = not value
            bool2str = {True: "ON", False: "OFF"}
            asyncio.ensure_future(
                PERS.openhab.set_item(  # type: ignore
                    SWITCH_ITEMS[key]["item"], bool2str[value], log=True
                )
            )


async def main(pers=None, waitforever=True):
    """Fetch a snapshot and setup websocket listener.
    Then waits forever to ensure the loop is running."""
    global PERS
    if pers is None:
        pers = persist.PyrotunPersistence()
        await pers.ainit(requested=["unifiprotect", "openhab"])
    PERS = pers

    pers.unifiprotect.protect.subscribe_websocket(process_camera_event)
    await fetch_snapshot(pers.unifiprotect.protect, CAMERA_FILENAME)

    if waitforever:
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
