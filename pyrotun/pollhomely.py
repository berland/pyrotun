import asyncio

import dotenv

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

ALARM_ARMED_ITEM = "AlarmArmert"


async def amain(pers=None, debug=False):
    close_pers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        dotenv.load_dotenv()
        close_pers = True
    if pers.homely is None:
        await pers.ainit(["homely"])
    if pers.openhab is None:
        await pers.ainit(["openhab"])

    data = await pers.homely.get_data()
    await update_openhab(pers, data)

    if close_pers:
        await pers.aclose()


async def update_openhab(pers, data):
    # "Alarm state" is not a "device" in the homely API response. Handled
    # outside the yaml file..
    alarmstate_map = {"ARMED": "ON", "DISARMED": "OFF"}
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


def fold(path: str, data: dict):
    """Lookup in deep dictionary, a 'fold'"""
    value = data
    for key in path.split("/"):
        value = value[key]
    return value


if __name__ == "__main__":
    asyncio.run(amain(pers=None))
