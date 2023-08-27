import asyncio
from pathlib import Path

import dotenv
import yaml

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

CONFIGSOLIS = "solisopenhab.yml"


async def amain(pers=None, debug=False):
    close_pers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        dotenv.load_dotenv()
        close_pers = True
    if pers.solis is None or pers.openhab is None:
        await pers.ainit(["solis", "openhab"])

    await post_solisdata_to_openhab(pers)

    if close_pers:
        await pers.aclose()


async def post_solisdata_to_openhab(pers) -> None:
    logger.info("Posting Solis data to OpenHAB")
    config = yaml.safe_load(Path(CONFIGSOLIS).read_text(encoding="utf-8"))
    data = await pers.solis.get_data()
    if not data.get("success", False):
        logger.error("Error message from Solis cloud: " + data.get("msg", "empty"))
        return
    for conf_item in config:
        await pers.openhab.set_item(
            conf_item["openhab_item"],
            str(data["data"][conf_item["solisname"]]),
            log=True,
        )


if __name__ == "__main__":
    asyncio.run(amain())
