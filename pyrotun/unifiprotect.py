import asyncio
import os
from pathlib import Path

import aiofiles
import dotenv
from pyunifiprotect import ProtectApiClient

import pyrotun
from pyrotun import persist

logger = pyrotun.getLogger(__name__)
dotenv.load_dotenv()
UNIFI_HOST = os.getenv("UNIFI_HOST")
UNIFI_PORT = os.getenv("UNIFI_PORT")
UNIFI_USERNAME = os.getenv("UNIFI_USERNAME")
UNIFI_PASSWORD = os.getenv("UNIFI_PASSWORD")

CAMERA_FILENAME: Path = Path("/etc/openhab/html/garagecamera.png")
CAMERA_SMALL: Path = Path("/etc/openhab/html/garagecamera_small.png")


async def fetch_snapshot(protect: ProtectApiClient) -> None:
    filename = CAMERA_FILENAME
    if not CAMERA_FILENAME.parent.exists():
        filename = Path(CAMERA_FILENAME.name)
    camera_id = list(protect.bootstrap.cameras.keys())[0]
    pngbytes = await protect.get_camera_snapshot(camera_id)
    if pngbytes:
        async with aiofiles.open(filename, "wb") as filehandle:
            await filehandle.write(pngbytes)
            await filehandle.flush()
            logger.info(f"Wrote {len(pngbytes)} bytes of PNG to {filename}")
    else:
        logger.error("fGot {len(bytes) bytes from camera")


async def main(pers=None):
    """Fetch snapshot from camera and save to png file on disk"""
    if pers is None:
        pers = persist.PyrotunPersistence()
    await pers.ainit(requested=["unifiprotect"])
    await fetch_snapshot(pers.unifiprotect.protect)
    await pers.unifiprotect.protect.async_disconnect_ws()
    await pers.unifiprotect.protect.close_session()


if __name__ == "__main__":
    asyncio.run(main())
