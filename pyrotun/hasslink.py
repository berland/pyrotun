import asyncio
from pathlib import Path

import dotenv
import yaml

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

CONFIG_HASSLINK = "hasslink.yml"


async def amain(pers=None, debug=False):
    close_pers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        dotenv.load_dotenv()
        close_pers = True
    if pers.hass is None or pers.openhab is None:
        await pers.ainit(["hass", "openhab"])

    await link_hass_states_to_openhab(pers)

    if close_pers:
        await pers.aclose()


async def link_hass_states_to_openhab(pers):
    config = yaml.safe_load(Path(CONFIG_HASSLINK).read_text(encoding="utf-8"))

    # Homeassistant to OpenHAB:
    for sensor in config["sensors"]:
        state = await pers.hass.get_item(
            sensor["entity_id"], attribute=sensor.get("attribute")
        )
        if state:
            await pers.openhab.set_item(
                sensor["openhab_item"], str(state), log=True, send_no_change=False
            )

    # OpenHAB to Homeassistant:
    for service in config["services"]:
        state = await pers.openhab.get_item(service["openhab_item"])
        await pers.hass.set_item(
            service["service_path"],
            service["entity_id"],
            service["attribute"],
            str(state),
        )


if __name__ == "__main__":
    asyncio.run(amain())
