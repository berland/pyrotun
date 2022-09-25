import asyncio

import dotenv

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)


async def amain(pers=None, debug=False):
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        dotenv.load_dotenv()
        close_pers = True
    if pers.skoda is None:
        await pers.ainit(["skoda"])
    if pers.openhab is None:
        await pers.ainit(["openhab"])

    await post_to_openhab(pers, debug=debug)

    if close_pers:
        await pers.aclose()


async def post_to_openhab(pers, debug):
    await pers.skoda.get_data()
    logger.info("Posting Skoda data to OpenHAB")
    for instrument in pers.skoda.instruments:
        if debug:
            print(f"{instrument.attr} {instrument.state}")
        oh_value = instrument.state
        if instrument.attr in pers.skoda.config:
            if "multiplier" in pers.skoda.config[instrument.attr]:
                oh_value = (
                    float(oh_value) * pers.skoda.config[instrument.attr]["multiplier"]
                )
            await pers.openhab.set_item(
                pers.skoda.config[instrument.attr]["openhab_item"],
                str(oh_value),
                log=True,
            )


if __name__ == "__main__":
    asyncio.run(amain())
