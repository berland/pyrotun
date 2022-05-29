import asyncio
import os

import aiofiles
import dotenv
from pyunifiprotect import ProtectApiClient

dotenv.load_dotenv()
UNIFI_HOST = os.getenv("UNIFI_HOST")
UNIFI_PORT = os.getenv("UNIFI_PORT")
UNIFI_USERNAME = os.getenv("UNIFI_USERNAME")
UNIFI_PASSWORD = os.getenv("UNIFI_PASSWORD")

CAMERA_FILENAME = "/etc/openhab/html/garagecamera.png"
CAMERA_SMALL = "/etc/openhab/html/garagecamera_small.png"


async def main(pers=None):
    """Fetch snapshot from camera and save to png file on disk"""
    protect = ProtectApiClient(
        UNIFI_HOST, UNIFI_PORT, UNIFI_USERNAME, UNIFI_PASSWORD, verify_ssl=False
    )
    await protect.update()
    camera_id = list(protect.bootstrap.cameras.keys())[0]

    pngbytes = await protect.get_camera_snapshot(camera_id)
    if pngbytes:
        async with aiofiles.open(CAMERA_FILENAME, "wb") as filehandle:
            await filehandle.write(pngbytes)
            await filehandle.flush()
    await protect.async_disconnect_ws()
    await protect.close_session()


if __name__ == "__main__":
    asyncio.run(main())
