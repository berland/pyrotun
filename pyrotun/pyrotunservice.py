"""
Main script for my house

Runs continously as a service, calls underlying tools/scripts
from asyncio at regular intervals (similar to crontab)

"""
import asyncio

import aiocron

import pyrotun
import pyrotun.helligdager
import pyrotun.yrmelding
import pyrotun.pollsmappee
import pyrotun.polltibber
import pyrotun.houseshadow
import pyrotun.connections.smappee
import pyrotun.connections.openhab
import pyrotun.connections.mqtt
import pyrotun.connections.tibber


logger = pyrotun.getLogger(__name__)

EVERY_SECOND = "* * * * * *"
EVERY_MINUTE = "* * * * *"
EVERY_5_MINUTE = "*/5 * * * *"
EVERY_15_MINUTE = "*/15 * * * *"
EVERY_HOUR = "0 * * * *"

CONNECTIONS = {
    "smappee": pyrotun.connections.smappee.SmappeeConnection(),
    "mqtt": pyrotun.connections.mqtt.MqttConnection(),
    "openhab": pyrotun.connections.openhab.OpenHABConnection(),
    "tibber": None,
    # must be awaited # "tibber":  pyrotun.connections.tibber.TibberConnection(),
}


@aiocron.crontab(EVERY_MINUTE)
async def heartbeat():
    logger.info("<puls>")


@aiocron.crontab(EVERY_5_MINUTE)
async def pollsmappe():
    # Todo use the same connection instead of reauth.
    logger.info("pollsmappee")
    pyrotun.pollsmappee.main(CONNECTIONS)


@aiocron.crontab(EVERY_HOUR)
async def helligdager():
    logger.info("helligdager")
    pyrotun.helligdager.main(CONNECTIONS)

@aiocron.crontab(EVERY_5_MINUTE)
async def polltibber():
    logger.info("polling tibber")
    await pyrotun.polltibber.main(CONNECTIONS)

@aiocron.crontab(EVERY_HOUR)
async def yrmelding():
    logger.info("yrmelding")
    pyrotun.yrmelding.main(CONNECTIONS)


@aiocron.crontab(EVERY_15_MINUTE)
async def houseshadow():
    logger.info("houseshadow")
    pyrotun.houseshadow.main("shadow.svg")


async def at_startup():
    await pyrotun.polltibber.main(CONNECTIONS)
    pyrotun.pollsmappee.main(CONNECTIONS)
    pyrotun.houseshadow.main("shadow.svg")
    pyrotun.yrmelding.main(CONNECTIONS)
    pyrotun.helligdager.main(CONNECTIONS)


if __name__ == "__main__":
    logger.info("Starting pyrotun service loop")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(at_startup())
    loop.run_forever()
