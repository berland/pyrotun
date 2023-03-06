"""
Addon-service to OpenHAB for things that are better programmed in CPython rather
than Jython (inside OpenHAB).

Runs continously as a service, calls underlying tools/scripts
from asyncio at regular intervals (similar to crontab)
"""
import asyncio
import datetime
from typing import Any, List

import aiocron
import dotenv

import pyrotun
import pyrotun.connections.hass
import pyrotun.connections.homely
import pyrotun.connections.openhab
import pyrotun.connections.skoda
import pyrotun.connections.smappee
import pyrotun.connections.solis
import pyrotun.connections.tibber
import pyrotun.connections.unifiprotect
import pyrotun.dataspike_remover
import pyrotun.discord
import pyrotun.disruptive
import pyrotun.exercise_uploader
import pyrotun.floors
import pyrotun.hasslink
import pyrotun.helligdager
import pyrotun.houseshadow
import pyrotun.persist
import pyrotun.polar_dump
import pyrotun.pollhomely
import pyrotun.pollskoda
import pyrotun.pollsmappee
import pyrotun.pollsolis
import pyrotun.polltibber
import pyrotun.poweranalysis
import pyrotun.powercontroller
import pyrotun.powermodels
import pyrotun.skyss
import pyrotun.unifiprotect
import pyrotun.vent_calculations
import pyrotun.yrmelding

logger = pyrotun.getLogger(__name__)

EVERY_SECOND = "* * * * * *"
EVERY_10_SECOND = "* * * * * */10"
EVERY_15_SECOND = "* * * * * */15"
EVERY_MINUTE = "* * * * *"
EVERY_5_MINUTE = "*/5 * * * *"
EVERY_8_MINUTE = "*/8 * * * *"
EVERY_15_MINUTE = "*/15 * * * *"
EVERY_HOUR = "0 * * * *"
EVERY_DAY = "0 0 * * *"
EVERY_MIDNIGHT = EVERY_DAY


def setup_crontabs(pers):
    """Registers coroutines for execution via crontab syntax.

    Requires the persistence object to be initialized."""

    @aiocron.crontab(EVERY_15_SECOND)
    async def do_hasslink():
        await asyncio.sleep(0.5)
        logger.pyrotun("Linking Homeassistant")
        await pyrotun.hasslink.link_hass_states_to_openhab(pers)

    @aiocron.crontab(EVERY_15_SECOND)
    async def poll_unifiprotect():
        await asyncio.sleep(3)
        logger.info("Getting garage camera snapshot")
        await pyrotun.unifiprotect.fetch_snapshot(
            pers.unifiprotect.protect, pyrotun.unifiprotect.CAMERA_FILENAME
        )

    @aiocron.crontab(EVERY_10_SECOND)
    async def poll_homely():
        logger.info(" ** Polling Homely")
        await pyrotun.pollhomely.amain(pers)

    @aiocron.crontab(EVERY_MINUTE)
    async def poll_skoda():
        logger.info(" ** Polling Skoda")
        await pyrotun.pollskoda.amain(pers)

    @aiocron.crontab(EVERY_15_SECOND)
    async def vent_calc():
        logger.info(" ** Ventilation calculations")
        await pyrotun.vent_calculations.main(pers)

    @aiocron.crontab(EVERY_15_SECOND)
    async def poll_skyss():
        await asyncio.sleep(5)  # No need to overlap with ventilation
        logger.info(" ** Polling skyss")
        await pyrotun.skyss.main(pers)

    @aiocron.crontab(EVERY_MINUTE)
    async def pollsolis():
        await asyncio.sleep(2)
        logger.info(" ** Pollsolis")
        await pyrotun.pollsolis.post_solisdata_to_openhab(pers)

    @aiocron.crontab(EVERY_5_MINUTE)
    async def pollsmappe():
        await asyncio.sleep(10)
        logger.info(" ** Pollsmappee")
        await pyrotun.pollsmappee.main(pers)

    @aiocron.crontab(EVERY_MIDNIGHT)
    async def reset_daily_cum():
        await pers.openhab.set_item("Smappee_day_cumulative", 0)

    @aiocron.crontab(EVERY_HOUR)
    async def helligdager():
        logger.info(" ** Helligdager")
        await pyrotun.helligdager.main(pers)

    @aiocron.crontab(EVERY_15_MINUTE)
    async def polltibber():
        logger.info(" ** Polling tibber")
        await pyrotun.polltibber.main(pers)

    @aiocron.crontab(EVERY_MIDNIGHT)
    async def calc_power_savings_yesterday():
        logger.info(" ** Calculating power cost savings yesterday")
        await pyrotun.poweranalysis.estimate_savings_yesterday(pers, dryrun=False)

    @aiocron.crontab(EVERY_MINUTE)
    async def update_thishour_powerestimate():
        estimate = await pyrotun.powercontroller.estimate_currenthourusage(pers)
        await pers.openhab.set_item("EstimatedKWh_thishour", estimate)

    @aiocron.crontab(EVERY_MINUTE)
    async def turn_off_if_overshooting_powerusage():
        await asyncio.sleep(15)
        await pyrotun.powercontroller.amain(pers)

    @aiocron.crontab(EVERY_HOUR)
    async def update_thismonth_nettleie():
        if datetime.datetime.now().hour == 0:
            # We have a bug that prevents correct calculation
            # the first hour of every day..
            return
        await asyncio.sleep(30)  # Wait for AMS data to propagate to Influx
        logger.info(" ** Updating nettleie")
        await pyrotun.powercontroller.update_effekttrinn(pers)

    @aiocron.crontab(EVERY_8_MINUTE)
    async def floors_controller():
        logger.info(" ** Floor controller")
        await pyrotun.floors.main(pers)

    @aiocron.crontab(EVERY_8_MINUTE)
    async def waterheater_controller():
        await asyncio.sleep(60)  # No need to overlap with floors_controller
        logger.info(" ** Waterheater controller")
        await pyrotun.waterheater.controller(pers)

    @aiocron.crontab(EVERY_HOUR)
    async def estimate_savings():
        # 3 minutes after every hour
        await asyncio.sleep(60 * 3)
        logger.info(" ** Waterheater 24h saving estimation")
        await pyrotun.waterheater.estimate_savings(pers)

    @aiocron.crontab(EVERY_HOUR)
    async def yrmelding():
        logger.info(" ** Yrmelding")
        await pyrotun.yrmelding.main(pers)

    @aiocron.crontab(EVERY_HOUR)
    async def sunheatingmodel():
        logger.info(" ** sunheating model")
        sunmodel = await pyrotun.powermodels.sunheating_model(pers)
        pers.powermodels.sunheatingmodel = sunmodel

    @aiocron.crontab(EVERY_15_MINUTE)
    async def houseshadow():
        await asyncio.sleep(5)
        logger.info(" ** Houseshadow")
        pyrotun.houseshadow.main("/etc/openhab/html/husskygge.svg")

    @aiocron.crontab(EVERY_15_MINUTE)
    async def polar_dump_now():
        logger.info(" ** Polar dumper")
        pyrotun.polar_dump.main()

    @aiocron.crontab(EVERY_15_MINUTE)
    async def spikes():
        logger.info(" ** Dataspike remover")
        await pyrotun.dataspike_remover.main(pers, readonly=False)


async def at_startup(pers) -> List[Any]:
    """Schedule coroutines for immediate execution as tasks, return
    list of the scheduled tasks in order to avoid gc."""

    # Keep a strong reference to each task we construct, in
    # order to avoid them being garbage collected before completion:
    tasks = []
    tasks.append(
        asyncio.create_task(pyrotun.dataspike_remover.main(pers, readonly=False))
    )
    tasks.append(asyncio.create_task(pyrotun.vent_calculations.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.polltibber.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.pollsmappee.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.powercontroller.update_effekttrinn(pers)))
    tasks.append(asyncio.create_task(pyrotun.discord.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.disruptive.main(pers)))
    tasks.append(
        asyncio.create_task(pyrotun.unifiprotect.main(pers, waitforever=False))
    )

    # Sets up an async generator:
    tasks.append(asyncio.create_task(pyrotun.exercise_uploader.main(pers)))

    tasks.append(asyncio.create_task(pyrotun.houseshadow.amain("shadow.svg")))

    tasks.append(asyncio.create_task(pyrotun.floors.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.waterheater.controller(pers)))
    tasks.append(asyncio.create_task(pyrotun.yrmelding.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.helligdager.main(pers)))
    # tasks.append(asyncio.create_task(pyrotun.pollhomely.supervise_websocket(pers)))

    return tasks


async def main():
    logger.info("Starting pyrotun service")
    pers = pyrotun.persist.PyrotunPersistence()
    await pers.ainit(requested="all")

    startup_tasks = await at_startup(pers)
    assert startup_tasks
    # (these tasks are kept in memory forever..)

    setup_crontabs(pers)

    await asyncio.Event().wait()


if __name__ == "__main__":
    dotenv.load_dotenv(verbose=True)
    asyncio.run(main(), debug=False)
