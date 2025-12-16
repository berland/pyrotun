"""
Addon-service to OpenHAB for things that are better programmed in CPython rather
than Jython (inside OpenHAB).

Runs continously as a service, calls underlying tools/scripts
from asyncio at regular intervals (similar to crontab)
"""

import asyncio
import datetime
import json
import os
import traceback
from pathlib import Path
from typing import Any, List

import aiocron
import aiohttp
import dotenv

import pyrotun
import pyrotun.connections.hass
import pyrotun.connections.homely
import pyrotun.connections.openhab
import pyrotun.connections.smappee
import pyrotun.connections.solis
import pyrotun.connections.tibber
import pyrotun.dataspike_remover

# import pyrotun.discord
# import pyrotun.disruptive
import pyrotun.exercise_analyzer
import pyrotun.exercise_uploader
import pyrotun.floors
import pyrotun.hasslink
import pyrotun.helligdager
import pyrotun.houseshadow
import pyrotun.persist
import pyrotun.polar_dump
import pyrotun.pollhomely
import pyrotun.pollmyuplink
import pyrotun.pollskoda
import pyrotun.pollsmappee
import pyrotun.pollsolis
import pyrotun.polltibber
import pyrotun.poweranalysis
import pyrotun.powercontroller
import pyrotun.powermodels
import pyrotun.vent_calculations
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


async def send_pushover(message: str, title: str = "Async App Error"):
    PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
    payload = {
        "token": os.getenv("PUSHOVER_APIKEY"),
        "user": os.getenv("PUSHOVER_USER"),
        "title": title,
        "message": message,
        "priority": 1,
    }

    async with (
        aiohttp.ClientSession() as session,
        session.post(PUSHOVER_URL, data=payload) as resp,
    ):
        if resp.status != 200:
            err = await resp.text()
            logger.error(f"Failed to send pushover: {err}")


def global_exception_handler(loop, context):
    exc = context.get("exception")

    if exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        message = tb
    else:
        # No exception object, just a message
        message = context.get("message", "Unknown asyncio exception")

    # Pushover limit ~1024 chars
    message = message[-1000:]

    if "identity.vwgroup.io/signin" in message:
        return

    asyncio.create_task(
        send_pushover(
            title="Unhandled asyncio exception",
            message=message,
        )
    )


def get_alexa_serial_to_devicename() -> dict:
    thingsfile = Path("/etc/openhab/things/amazonechocontrol.things")
    if not thingsfile.exists():
        return {}
    lines = [
        line
        for line in Path("/etc/openhab/things/amazonechocontrol.things")
        .read_text(encoding="utf-8")
        .splitlines()
        if "Thing" in line
    ]
    themap = {}
    for line in lines:
        tokens = (
            line.replace("=", " ")
            .replace("[", "")
            .replace("]", "")
            .replace('"', "")
            .split()
        )
        themap[tokens[4]] = tokens[2]
    return themap


ALEXA_SERIAL_TO_DEVICENAME = get_alexa_serial_to_devicename()


def setup_crontabs(pers):
    """Registers coroutines for execution via crontab syntax.

    Requires the persistence object to be initialized."""

    async def get_alexa_last_command():
        # Band-aid until alexa binding is updated.
        async with pers.websession.get(
            "http://raaserv.r40:8090/amazonechocontrol/berlandaccount/"
            "PROXY/api/activities?startTime=&size=1&offset=1"
        ) as response:
            if response.ok:
                jsondata = await response.json(content_type="text/html;charset=utf-8")
                serialnumber = jsondata["activities"][0]["sourceDeviceIds"][0][
                    "serialNumber"
                ]
                devicename = ALEXA_SERIAL_TO_DEVICENAME[serialnumber]
                json_inside_json = jsondata["activities"][0]["description"]
                command = (
                    json.loads(json_inside_json)["summary"]
                    .removeprefix("echo ")
                    .removeprefix("alexa ")
                )
                itemname = devicename.title() + "Alexa_lastCommand"
                currentlastcommand = await pers.openhab.get_item(itemname)
                if command and not currentlastcommand.startswith(command):
                    await pers.openhab.set_item(
                        itemname, command, method="put", log=True, send_no_change=False
                    )

    @aiocron.crontab(EVERY_15_SECOND)
    async def do_myuplink():
        await asyncio.sleep(0.2)
        logger.info("Linking Myuplink")
        await pyrotun.pollmyuplink.update_openhab(pers)

    # @aiocron.crontab(EVERY_15_SECOND)
    async def do_hasslink():
        await asyncio.sleep(0.5)
        logger.info("Linking Homeassistant")
        await pyrotun.hasslink.link_hass_states_to_openhab(pers)

    @aiocron.crontab(EVERY_15_SECOND)
    async def poll_unifiprotect():
        await asyncio.sleep(3)
        logger.info("Getting garage camera snapshot")
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i",
            os.getenv("UNIFI_RTSP"),
            "-vframes",
            "1",
            "/etc/openhab/html/garagecamera.png",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_data, stderr_data = await process.communicate()
        return_code = await process.wait()
        if return_code:
            logger.error(stdout_data.decode())
            logger.error(stderr_data.decode())

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

    # @aiocron.crontab(EVERY_15_SECOND)
    async def poll_skyss():
        return
        await asyncio.sleep(5)  # No need to overlap with ventilation
        logger.info(" ** Polling skyss")
        await pyrotun.skyss.main(pers)

    @aiocron.crontab(EVERY_MINUTE)
    async def pollsolis():
        await asyncio.sleep(2)
        logger.info(" ** Pollsolis")
        await pyrotun.pollsolis.post_solisdata_to_openhab(pers)

    # @aiocron.crontab(EVERY_5_MINUTE)
    async def pollsmappe():
        await asyncio.sleep(10)
        logger.info(" ** Pollsmappee")
        await pyrotun.pollsmappee.main(pers)

    # @aiocron.crontab(EVERY_MIDNIGHT)
    async def reset_daily_cum():
        await pers.openhab.set_item("Smappee_day_cumulative", 0)

    @aiocron.crontab(EVERY_HOUR)
    async def update_public_ip():
        logger.info(" ** Public IP")
        async with pers.websession.get("https://api.ipify.org") as response:
            if response.ok:
                ip = await response.content.read()
                await pers.openhab.set_item("PublicIP", ip.decode("utf-8"))

    @aiocron.crontab(EVERY_HOUR)
    async def helligdager():
        logger.info(" ** Helligdager")
        await pyrotun.helligdager.main(pers)

    @aiocron.crontab(EVERY_15_MINUTE)
    async def polltibber():
        logger.info(" ** Polling tibber")
        await pyrotun.polltibber.main(pers)

    # @aiocron.crontab(EVERY_MIDNIGHT)
    # async def calc_power_savings_yesterday():
    #     logger.info(" ** Calculating power cost savings yesterday")
    #     await pyrotun.poweranalysis.estimate_savings_yesterday(pers, dryrun=False)

    @aiocron.crontab(EVERY_MINUTE)
    async def update_thishour_powerestimate():
        estimate = await pyrotun.powercontroller.estimate_currenthourusage(pers)
        await pers.openhab.set_item("EstimatedKWh_thishour", estimate)

    #    @aiocron.crontab(EVERY_MINUTE)
    #    async def turn_off_if_overshooting_powerusage():
    #        await asyncio.sleep(15)
    #        await pyrotun.powercontroller.amain(pers)

    @aiocron.crontab(EVERY_HOUR)
    async def update_thismonth_nettleie():
        if datetime.datetime.now().hour == 0:
            # We have a bug that prevents correct calculation
            # the first hour of every day..
            return
        await asyncio.sleep(30)  # Wait for AMS data to propagate to Influx
        logger.info(" ** Updating nettleie")
        await pyrotun.powercontroller.update_effekttrinn(pers)

    # @aiocron.crontab(EVERY_8_MINUTE)
    # async def floors_controller():
    #    logger.info(" ** Floor controller")
    #    await pyrotun.floors.main(pers)

    # @aiocron.crontab(EVERY_8_MINUTE)
    # async def waterheater_controller():
    #     await asyncio.sleep(60)  # No need to overlap with floors_controller
    #     logger.info(" ** Waterheater controller")
    #     await pyrotun.waterheater.controller(pers)

    # @aiocron.crontab(EVERY_HOUR)
    # async def estimate_savings():
    #    # 3 minutes after every hour
    #    await asyncio.sleep(60 * 3)
    #    logger.info(" ** Waterheater 24h saving estimation")
    #    await pyrotun.waterheater.estimate_savings(pers)

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

    @aiocron.crontab(EVERY_5_MINUTE)
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
    # tasks.append(asyncio.create_task(pyrotun.discord.main(pers)))
    # tasks.append(asyncio.create_task(pyrotun.disruptive.main(pers)))
    # tasks.append(
    #    asyncio.create_task(pyrotun.unifiprotect.main(pers, waitforever=False))
    # )

    # Sets up an async generator:
    tasks.append(asyncio.create_task(pyrotun.exercise_uploader.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.exercise_analyzer.main(pers)))

    tasks.append(asyncio.create_task(pyrotun.houseshadow.amain("shadow.svg")))

    # tasks.append(asyncio.create_task(pyrotun.floors.main(pers)))
    # tasks.append(asyncio.create_task(pyrotun.waterheater.controller(pers)))
    tasks.append(asyncio.create_task(pyrotun.yrmelding.main(pers)))
    tasks.append(asyncio.create_task(pyrotun.helligdager.main(pers)))
    # tasks.append(asyncio.create_task(pyrotun.pollhomely.supervise_websocket(pers)))

    return tasks


async def main():
    logger.info("Starting pyrotun service")
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(global_exception_handler)
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
