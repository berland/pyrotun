"""
Main script for my house

Runs continously as a service, calls underlying tools/scripts
from asyncio at regular intervals (similar to crontab)

"""
import asyncio

import aiocron
import dotenv

import pyrotun
import pyrotun.helligdager
import pyrotun.yrmelding
import pyrotun.pollsmappee
import pyrotun.polltibber
import pyrotun.houseshadow
import pyrotun.vent_calculations
import pyrotun.connections.smappee
import pyrotun.connections.openhab
import pyrotun.connections.mqtt
import pyrotun.connections.tibber
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

EVERY_SECOND = "* * * * * *"
EVERY_15_SECOND = "* * * * * */15"
EVERY_MINUTE = "* * * * *"
EVERY_5_MINUTE = "*/5 * * * *"
EVERY_8_MINUTE = "*/8 * * * *"
EVERY_15_MINUTE = "*/15 * * * *"
EVERY_HOUR = "0 * * * *"

PERS = None


# @aiocron.crontab(EVERY_MINUTE)
# async def heartbeat():
#    logger.info("<puls>")


@aiocron.crontab(EVERY_15_SECOND)
async def vent_calc():
    logger.info("Running ventilation calculations")
    await pyrotun.vent_calculations.main(PERS)


@aiocron.crontab(EVERY_5_MINUTE)
async def pollsmappe():
    # Todo use the same connection instead of reauth.
    logger.info("pollsmappee")
    await pyrotun.pollsmappee.main(PERS)


@aiocron.crontab(EVERY_HOUR)
async def helligdager():
    logger.info("helligdager")
    await pyrotun.helligdager.main(PERS)


@aiocron.crontab(EVERY_15_MINUTE)
async def polltibber():
    logger.info("polling tibber")
    await pyrotun.polltibber.main(PERS)


@aiocron.crontab(EVERY_8_MINUTE)
async def waterheater_controller():
    logger.info("Running waterheater controller")
    await pyrotun.waterheater.controller(PERS)


@aiocron.crontab(EVERY_HOUR)
async def yrmelding():
    logger.info("yrmelding")
    await pyrotun.yrmelding.main(PERS)


@aiocron.crontab(EVERY_15_MINUTE)
async def houseshadow():
    logger.info("houseshadow")
    pyrotun.houseshadow.main("shadow.svg")


async def at_startup(pers):
    await pyrotun.vent_calculations.main(pers)
    await pyrotun.polltibber.main(pers)

    await pyrotun.pollsmappee.main(pers)
    pyrotun.houseshadow.main("shadow.svg")
    await pyrotun.yrmelding.main(pers)
    await pyrotun.helligdager.main(pers)
    await pyrotun.waterheater.controller(pers)


if __name__ == "__main__":
    logger.info("Starting pyrotun service loop")
    dotenv.load_dotenv(verbose=True)
    pers = pyrotun.persist.PyrotunPersistence()
    PERS = pers
    loop = asyncio.get_event_loop()
    loop.run_until_complete(pers.ainit())
    loop.run_until_complete(at_startup(pers))
    loop.run_forever()
