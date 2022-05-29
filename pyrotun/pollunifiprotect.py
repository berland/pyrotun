import asyncio

import aiofiles
import dotenv

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

PERS = None


async def main(pers=PERS):
    dotenv.load_dotenv()

    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
    if pers.unifiprotect is None:
        await pers.ainit(["unifiprotect"])

    protect = pers.unifiprotect.protect

    camera_id = list(protect.bootstrap.cameras.keys())[0]

    pngbytes = await protect.get_camera_snapshot(camera_id)
    if pngbytes:
        async with aiofiles.open("camera.png", "wb") as filehandle:
            await filehandle.write(pngbytes)
            await filehandle.flush()
            print("done")


if __name__ == "__main__":
    PERS = pyrotun.persist.PyrotunPersistence()
    asyncio.get_event_loop().run_until_complete(main())
