import asyncio
import os

import dotenv

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)


async def main():
    dotenv.load_dotenv()

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        os.getenv("UNIFI_RTSP"),
        "-vframes",
        "1",
        "/etc/openhab/html/camera.jpg",
    )
    await process.wait()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
