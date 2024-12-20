"""
Addon-service to OpenHAB for things that are better programmed in CPython rather
than Jython (inside OpenHAB).

Runs continously as a service, calls underlying tools/scripts
from asyncio at regular intervals (similar to crontab)
"""

import asyncio
from typing import Any, List

import aiocron
import dotenv

import pyrotun
import pyrotun.connections.homely
import pyrotun.connections.openhab
import pyrotun.connections.tibber
import pyrotun.dataspike_remover
import pyrotun.discord
import pyrotun.elvatunheating
import pyrotun.floors
import pyrotun.houseshadow
import pyrotun.persist
import pyrotun.pollhomely
import pyrotun.polltibber
import pyrotun.poweranalysis
import pyrotun.powermodels
import pyrotun.yrmelding

logger = pyrotun.getLogger(__name__)

EVERY_SECOND = "* * * * * *"
EVERY_4_SECOND = "* * * * * */4"
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

    # @aiocron.crontab(EVERY_10_SECOND)
    # async def poll_homely():
    #    logger.info(" ** Polling Homely")
    #    await pyrotun.pollhomely.amain(pers)

    @aiocron.crontab(EVERY_HOUR)
    async def optimize_heating_setpoint():
        await asyncio.sleep(10)  # Ensure new power price is live.
        gang_setpoint = await pers.openhab.get_item(
            "Namronovn_Gang_600w_setpoint", datatype=float
        )
        if gang_setpoint > 12:
            logger.info(" ** We are present, not touching setpoints")
            return
        logger.info(" ** Optimize heating setpoint")
        setpoint = await pers.elvatunheating.controller()
        logger.info(f"Calculated optimal setpoint: {setpoint}")
        await pers.openhab.set_item("Setpoint_optimized", str(setpoint))

    @aiocron.crontab(EVERY_DAY)
    async def update_heatingmodel():
        logger.info(" ** Update heating model")
        await pers.elvatunheating.update_heatingmodel()
        await pers.openhab.set_item(
            "WattPrHeatedDegree",
            pers.elvatunheating.powerusagemodel["powermodel"].coef_[0][0],
        )
        await pers.openhab.set_item(
            "WattPrOutsidedifference",
            pers.elvatunheating.powerusagemodel["powermodel"].coef_[0][1],
        )
        await pers.openhab.set_item(
            "WattEquilibrium",
            pers.elvatunheating.powerusagemodel["powermodel"].intercept_[0],
        )

    @aiocron.crontab(EVERY_HOUR)
    async def update_public_ip():
        logger.info(" ** Public IP")
        async with pers.websession.get("https://api.ipify.org") as response:
            if response.ok:
                ip = await response.content.read()
                await pers.openhab.set_item("PublicIP", ip.decode("utf-8"))

    @aiocron.crontab(EVERY_15_MINUTE)
    async def polltibber():
        logger.info(" ** Polling tibber")
        await pyrotun.polltibber.main(pers)

    @aiocron.crontab(EVERY_HOUR)
    async def yrmelding():
        logger.info(" ** Yrmelding")
        await pyrotun.yrmelding.main(pers)

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
    tasks.append(asyncio.create_task(pyrotun.polltibber.main(pers)))

    tasks.append(asyncio.create_task(pyrotun.yrmelding.main(pers)))

    return tasks


async def main():
    logger.info("Starting pyelvtun service")
    pers = pyrotun.persist.PyrotunPersistence()

    await pers.ainit(
        requested=["openhab", "influxdb", "tibber", "homely", "yr", "elvatunheating"]
    )

    startup_tasks = await at_startup(pers)
    assert startup_tasks
    # (these tasks are kept in memory forever..)

    setup_crontabs(pers)

    await asyncio.Event().wait()


if __name__ == "__main__":
    dotenv.load_dotenv(verbose=True)
    asyncio.run(main())
