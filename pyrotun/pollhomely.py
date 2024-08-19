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
        PERS.homely.message_handler = partial(
            update_openhab_from_websocket_message, PERS
        )

    data: List[dict] = await PERS.homely.get_data()  # One item pr. location

    if debug:
        pprint.pprint(data)

    if not dryrun:
        for location_data in data:
            await update_openhab(PERS, location_data)

    if do_websocket:
        websocket_supervisor = asyncio.create_task(supervise_websocket(PERS))
        await asyncio.wait([websocket_supervisor])

    if close_pers:
        await PERS.aclose()


async def supervise_websocket(pers):
    """Retry the websocket indefinetely..."""
    errors = 0
    while True:
        ws_task = asyncio.create_task(pers.homely.run_websocket())
        await asyncio.wait([ws_task])
        errors += 1
        if errors > 10:
            logger.error("Giving up on homely websocket, failed 10 times")
            return
        logger.warning("Websocket died, re-establishing in 2 seconds")
        await asyncio.sleep(2)


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
                await pers.openhab.set_item(
                    oh_item["openhab_item"],
                    str(change["value"]),
                    log=True,
                    method="post",
                )
    else:
        pprint.pprint(data)


async def update_openhab(pers, data):
    # "Alarm state" is not a "device" in the homely API response. Handled
    # outside the yaml file..
    if data["name"].strip() == "Råtun 40":
        alarmstate_map = {"ARMED_AWAY": "ON", "DISARMED": "OFF"}
        await pers.openhab.set_item(
            ALARM_ARMED_ITEM, alarmstate_map[data["alarmState"]], log=True
        )

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

        if str(conf_item.get("type")) == "bool":
            oh_value = {True: "ON", False: "OFF"}[value]
            method = "post"
        elif str(conf_item.get("type")) == "door":
            oh_value = {True: "OPEN", False: "CLOSED"}[value]
            method = "put"
        else:
            oh_value = str(value)
            method = "post"

        await pers.openhab.set_item(
            conf_item["openhab_item"], oh_value, log=True, method=method
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
    args = parser.parse_args()
    asyncio.run(amain(pers=None, dryrun=args.dryrun, debug=args.debug))
