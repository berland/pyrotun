import argparse
import asyncio
import json

import dotenv

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

PERS = None


async def amain(pers=None, dryrun=False, debug=False):
    close_pers = False
    if pers is None:
        PERS = pyrotun.persist.PyrotunPersistence()
        dotenv.load_dotenv()
        close_pers = True
    else:
        PERS = pers

    if PERS.myuplink is None:
        await PERS.ainit(["myuplink", "openhab"])
    assert PERS is not None
    assert PERS.myuplink is not None

    while PERS.myuplink.access_token is None:
        await asyncio.sleep(0.01)
    while PERS.myuplink.vvb_device_id is None:
        await asyncio.sleep(0.01)

    if debug:
        res = await PERS.myuplink.get(
            f"v2/devices/{PERS.myuplink.vvb_device_id}/points"
        )
        assert res is not None
        datapoints = {
            point["parameterName"]: point["value"] for point in json.loads(res)
        }

        for p_name, value in datapoints.items():
            print(f"{p_name} = {value}")

    if not dryrun:
        await update_openhab(PERS)

    if close_pers:
        await PERS.aclose()


async def update_openhab(pers):
    logger.info("Pulling data from myuplink to OpenHAB")
    assert pers.myuplink is not None
    result = None
    attempt = 0
    while result is None and attempt < 5:
        attempt += 1
        result = await pers.myuplink.get(
            f"v2/devices/{pers.myuplink.vvb_device_id}/points"
        )
        datapoints = {
            point["parameterName"]: point["value"] for point in json.loads(result)
        }
        for p_name, value in datapoints.items():
            if p_name in pers.myuplink.config:
                await pers.openhab.set_item(
                    pers.myuplink.config[p_name], value, log=True
                )
        if result is None:
            await asyncio.sleep(5)


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
