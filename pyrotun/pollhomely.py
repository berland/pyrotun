import argparse
import asyncio
import pprint
from functools import partial
from typing import Any, List

import dotenv

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

ALARM_ARMED_ITEM = "AlarmArmert"

ALARM_LOCATION_NAME = "Råtun 40"

# States not listed (ARMED_STAY, ARMED_NIGHT, BREACHED) leave the item untouched
ALARMSTATE_MAP = {
    "ARMED_AWAY": "ON",
    "DISARMED": "OFF",
    "ARM_PENDING": "OFF",
    "ARM_AWAY_PENDING": "OFF",
    "ARM_STAY_PENDING": "OFF",
    "ARM_NIGHT_PENDING": "OFF",
}

PERS = None


async def amain(pers=None, dryrun=False, debug=False, do_websocket=False):
    close_pers = False
    if pers is None:
        PERS = pyrotun.persist.PyrotunPersistence()
        dotenv.load_dotenv()
        close_pers = True
    else:
        PERS = pers
    if PERS.homely is None:
        await PERS.ainit(["homely", "openhab"])
        assert PERS is not None

    try:
        data: List[dict] = await PERS.homely.get_data()  # One item pr. location
    except ValueError as ex:
        if "504 Gateway Time-out" in str(ex):
            return
        raise ex

    if debug:
        pprint.pprint(data)

    if not dryrun:
        for location_data in data:
            try:
                await update_openhab(PERS, location_data)
            except KeyError as err:
                logger.error(
                    "There is no key data in what came from homely, "
                    f"look: {location_data}, got error {err}"
                )
    if do_websocket:
        websocket_supervisor = asyncio.create_task(
            supervise_websocket(PERS, dryrun=dryrun)
        )
        await asyncio.wait([websocket_supervisor])

    if close_pers:
        await PERS.aclose()


async def print_websocket_message(data):
    print(f"Websocket message: {pprint.pformat(data)}", flush=True)


async def supervise_websocket(pers, dryrun=False):
    """Keep the homely websocket alive indefinitely, with backoff.

    In dryrun mode, messages are only printed, not sent to OpenHAB."""
    if dryrun:
        pers.homely.message_handler = print_websocket_message
    else:
        pers.homely.message_handler = partial(
            update_openhab_from_websocket_message, pers
        )
    delay = 2
    while True:
        started = asyncio.get_running_loop().time()
        try:
            await pers.homely.run_websocket()
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa
            logger.error(f"Homely websocket failed: {err!r}")
        if asyncio.get_running_loop().time() - started > 300:
            delay = 2  # It was healthy for a while
        logger.warning(f"Websocket died, re-establishing in {delay} seconds")
        await asyncio.sleep(delay)
        delay = min(delay * 2, 120)


async def update_openhab_from_websocket_message(pers, data):
    if data["type"] == "device-state-changed":
        # A device may link to multiple openhab items.
        oh_items = [
            conf_item
            for conf_item in pers.homely.config
            if conf_item.get("id") == data["data"]["deviceId"]
        ]
        for change in data["data"]["changes"]:
            oh_item = [
                conf_item
                for conf_item in oh_items
                if conf_item.get("featurepath", "").startswith(change.get("feature"))
            ]
            if len(oh_item) > 1:
                logger.error("Too many openhab_items found for data change from homely")
                logger.error("That is a bug, fix")
            if len(oh_item) == 1:
                oh_item = oh_item[0]
                oh_value, method = to_openhab_value(oh_item, change["value"])
                await set_item_logged(
                    pers, "websocket", oh_item["openhab_item"], oh_value, method
                )
    elif data["type"] == "alarm-state-changed":
        location_name = pers.homely.location_names.get(data["data"]["locationId"])
        oh_value = ALARMSTATE_MAP.get(data["data"]["state"])
        if location_name == ALARM_LOCATION_NAME and oh_value is not None:
            await set_item_logged(pers, "websocket", ALARM_ARMED_ITEM, oh_value)
    else:
        pprint.pprint(data)


async def set_item_logged(pers, source, item, value, method="post"):
    """Set an OpenHAB item, logging whether it came from polling or websocket"""
    logger.info("OpenHAB: Setting %s to %s (homely %s)", item, value, source)
    await pers.openhab.set_item(item, value, log=False, method=method)


def to_openhab_value(conf_item: dict, value: Any) -> tuple:
    """Convert a homely value to an OpenHAB value and the http method to use"""
    if str(conf_item.get("type")) == "bool":
        return {True: "ON", False: "OFF"}[value], "post"
    if str(conf_item.get("type")) == "door":
        return {True: "OPEN", False: "CLOSED"}[value], "put"
    return str(value), "post"


async def update_openhab(pers, data):
    # "Alarm state" is not a "device" in the homely API response. Handled
    # outside the yaml file..
    if data["name"].strip() == ALARM_LOCATION_NAME:
        oh_value = ALARMSTATE_MAP.get(data["alarmState"])
        if oh_value is not None:
            await set_item_logged(pers, "polling", ALARM_ARMED_ITEM, oh_value)

    devices_used = set(value["name"] for value in pers.homely.config)
    homely_devices = set(value["name"] for value in data["devices"])
    unused_homely_devices = set(homely_devices) - set(devices_used)
    notexisting_homely_devices = set(devices_used) - set(homely_devices)
    if unused_homely_devices:
        logger.warning(f"Unused homely devices: {unused_homely_devices}")
    if notexisting_homely_devices:
        logger.warning(f"Not existing homely devices: {notexisting_homely_devices}")
    for conf_item in pers.homely.config:
        if conf_item["name"] in notexisting_homely_devices:
            continue
        item_dict = next(
            item for item in data["devices"] if item["name"] == conf_item["name"]
        )
        value = fold("features/" + conf_item["featurepath"], item_dict)
        if value is None:
            logger.warning(
                f"{item_dict['name']} was None for path {conf_item['featurepath']}"
            )
            continue

        oh_value, method = to_openhab_value(conf_item, value)

        await set_item_logged(
            pers, "polling", conf_item["openhab_item"], oh_value, method
        )


def fold(path: str, data: dict, separator: str = "/") -> Any:
    """Lookup in deep dictionary, a 'fold'"""
    value = data
    for key in path.split(separator):
        value = value[key]
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Debug mode, pretty print json data"
    )
    parser.add_argument(
        "--dryrun",
        "--dry",
        action="store_true",
        help="Dry run, do not submit to OpenHAB",
    )
    parser.add_argument(
        "--websocket",
        "-w",
        action="store_true",
        help="Listen to the websocket after polling. With --dryrun, the "
        "polled data and each received message are printed only.",
    )
    args = parser.parse_args()
    asyncio.run(
        amain(
            pers=None,
            dryrun=args.dryrun,
            debug=args.debug or (args.dryrun and args.websocket),
            do_websocket=args.websocket,
        )
    )
