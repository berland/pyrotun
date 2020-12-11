"""
Addon-service to OpenHAB for things that are better programmed in CPython rather
than Jython (inside OpenHAB).

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
import pyrotun.discord
import pyrotun.exercise_uploader
import pyrotun.dataspike_remover
import pyrotun.polar_dump

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
    logger.info(" ** Ventilation calculations")
    await pyrotun.vent_calculations.main(PERS)


@aiocron.crontab(EVERY_5_MINUTE)
async def pollsmappe():
    # Todo use the same connection instead of reauth.
    logger.info(" ** Pollsmappee")
    await pyrotun.pollsmappee.main(PERS)


@aiocron.crontab(EVERY_HOUR)
async def helligdager():
    logger.info(" ** Helligdager")
    await pyrotun.helligdager.main(PERS)


@aiocron.crontab(EVERY_15_MINUTE)
async def polltibber():
    logger.info(" ** Polling tibber")
    await pyrotun.polltibber.main(PERS)


@aiocron.crontab(EVERY_8_MINUTE)
async def waterheater_controller():
    logger.info(" ** Waterheater controller")
    await pyrotun.waterheater.controller(PERS)


@aiocron.crontab(EVERY_HOUR)
async def yrmelding():
    logger.info(" ** Yrmelding")
    await pyrotun.yrmelding.main(PERS)


@aiocron.crontab(EVERY_15_MINUTE)
async def houseshadow():
    logger.info(" ** Houseshadow")
    pyrotun.houseshadow.main("/etc/openhab2/html/husskygge.svg")


@aiocron.crontab(EVERY_15_MINUTE)
async def polar_dump_now():
    """Blocking(!)"""
    logger.info(" ** Polar dumper")
    pyrotun.polar_dump.main()


async def at_startup(pers):

    tasks = list()
    tasks.append(asyncio.create_task(pyrotun.dataspike_remover.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.vent_calculations.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.polltibber.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.pollsmappee.main(pers)))
    tasks.extend(await pyrotun.discord.main(pers, gather=False))
    tasks.extend(await pyrotun.exercise_uploader.main(pers))
    tasks.append(asyncio.create_task(pyrotun.houseshadow.amain("shadow.svg")))
    tasks.append(pyrotun.waterheater.controller(pers))
    tasks.append(pyrotun.yrmelding.main(pers))
    tasks.append(pyrotun.helligdager.main(pers))

    # This "blocks" because at least the discord modules contains
    # infinite generators.
    asyncio.gather(*tasks)


if __name__ == "__main__":
    logger.info("Starting pyrotun service loop")
    dotenv.load_dotenv(verbose=True)
    pers = pyrotun.persist.PyrotunPersistence()
    PERS = pers
    loop = asyncio.get_event_loop()
    loop.run_until_complete(pers.ainit())
    loop.run_until_complete(at_startup(pers))
    loop.run_forever()
