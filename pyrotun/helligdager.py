import asyncio
import datetime
import holidays

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)


async def main(pers):
    if datetime.datetime.now() in holidays.Norway():
        logger.info("Det er fridag")
        await pers.openhab.set_item("Fridag", "ON")
    else:
        logger.info("Det er ikke fridag")
        await pers.openhab.set_item("Fridag", "OFF")


if __name__ == "__main__":
    pers = pyrotun.persist.PyrotunPersistence()
    asyncio.run(pers.ainit())
    asyncio.run(main(pers))
    asyncio.run(pers.aclose())
