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
import pyrotun.poweranalysis
import pyrotun.houseshadow
import pyrotun.floors
import pyrotun.vent_calculations
import pyrotun.discord
import pyrotun.disruptive
import pyrotun.exercise_uploader
import pyrotun.dataspike_remover
import pyrotun.polar_dump
import pyrotun.pollsectoralarm
import pyrotun.powermodels

import pyrotun.connections.smappee
import pyrotun.connections.sectoralarm
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
EVERY_DAY = "0 0 * * *"
EVERY_MIDNIGHT = EVERY_DAY

PERS = None


@aiocron.crontab(EVERY_15_SECOND)
async def poll_sectoralarm():
    logger.info(" ** Polling sectoralarm")
    await pyrotun.pollsectoralarm.main(PERS)


@aiocron.crontab(EVERY_15_SECOND)
async def vent_calc():
    logger.info(" ** Ventilation calculations")
    await pyrotun.vent_calculations.main(PERS)


@aiocron.crontab(EVERY_5_MINUTE)
async def pollsmappe():
    logger.info(" ** Pollsmappee")
    await pyrotun.pollsmappee.main(PERS)


@aiocron.crontab(EVERY_MIDNIGHT)
async def reset_daily_cum():
    await PERS.openhab.set_item("Smappee_day_cumulative", 0)


@aiocron.crontab(EVERY_HOUR)
async def helligdager():
    logger.info(" ** Helligdager")
    await pyrotun.helligdager.main(PERS)


@aiocron.crontab(EVERY_15_MINUTE)
async def polltibber():
    logger.info(" ** Polling tibber")
    await pyrotun.polltibber.main(PERS)


@aiocron.crontab(EVERY_MIDNIGHT)
async def calc_power_savings_yesterday():
    logger.info(" ** Calculating power cost savings yesterday")
    await pyrotun.poweranalysis.estimate_savings_yesterday(PERS, dryrun=False)


@aiocron.crontab(EVERY_8_MINUTE)
async def waterheater_controller():
    await asyncio.sleep(60)  # No need to overlap with bathfloor
    logger.info(" ** Waterheater controller")
    await pyrotun.waterheater.controller(PERS)


@aiocron.crontab(EVERY_8_MINUTE)
async def floors_controller():
    logger.info(" ** Floor controller")
    await pyrotun.floors.main(PERS)


@aiocron.crontab(EVERY_HOUR)
async def estimate_savings():
    # 3 minutes after every hour
    await asyncio.sleep(60 * 3)
    logger.info(" ** Waterheater 24h saving estimation")
    await pyrotun.waterheater.estimate_savings(PERS)


@aiocron.crontab(EVERY_HOUR)
async def yrmelding():
    logger.info(" ** Yrmelding")
    await pyrotun.yrmelding.main(PERS)


@aiocron.crontab(EVERY_HOUR)
async def sunheatingmodel():
    logger.info(" ** sunheating model")
    sunmodel = await pyrotun.powermodels.sunheating_model(PERS)
    PERS.powermodels.sunheatingmodel = sunmodel


@aiocron.crontab(EVERY_15_MINUTE)
async def houseshadow():
    logger.info(" ** Houseshadow")
    pyrotun.houseshadow.main("/etc/openhab2/html/husskygge.svg")


@aiocron.crontab(EVERY_15_MINUTE)
async def polar_dump_now():
    """Blocking(!)"""
    logger.info(" ** Polar dumper")
    pyrotun.polar_dump.main()


@aiocron.crontab(EVERY_15_MINUTE)
async def spikes():
    logger.info(" ** Dataspike remover")
    await pyrotun.dataspike_remover.main(PERS, readonly=False)


async def at_startup(pers):

    tasks = list()
    tasks.append(
        asyncio.create_task(pyrotun.dataspike_remover.main(pers, readonly=False))
    )
    tasks.append(asyncio.create_task(pyrotun.vent_calculations.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.polltibber.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.pollsmappee.main(pers)))

    # discord.main() returns a list of tasks, the task is an async generator.
    tasks.extend(await pyrotun.discord.main(pers, gather=False))

    # disruptive.main() starts its own executor to run a sync function, but
    # it will never finish.
    tasks.append(asyncio.create_task(pyrotun.disruptive.main(pers)))

    # Sets up an async generator:
    tasks.append(asyncio.create_task(pyrotun.exercise_uploader.main(pers)))

    tasks.append(asyncio.create_task(pyrotun.houseshadow.amain("shadow.svg")))

    tasks.append(pyrotun.floors.main(pers))
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
    # we probably never get here..
    loop.run_forever()
