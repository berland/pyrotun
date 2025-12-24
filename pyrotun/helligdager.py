import asyncio
import datetime

import holidays

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)


async def main(pers):
    now = datetime.datetime.now()
    if now in holidays.Norway() or (now.day == 24 and now.month == 12):
        logger.info("Det er fridag")
        await pers.openhab.set_item("Fridag", "ON")
    else:
        logger.info("Det er ikke fridag")
        await pers.openhab.set_item("Fridag", "OFF")


async def amain():
    pers = pyrotun.persist.PyrotunPersistence()
    await pers.ainit(["openhab"])
    await main(pers)
    await pers.aclose()


if __name__ == "__main__":
    asyncio.run(amain())
