import asyncio
import json

import dotenv

import pyrotun
import pyrotun.connections.openhab
from pyrotun import persist

logger = pyrotun.getLogger()


dotenv.load_dotenv()

SKYSSAPI = "http://skyss.giantleap.no/public/"

# https://avgangsvisning.skyss.no/
SKYSS_STOPS = {"lagunen": 12017633, "rådal": 59775}
HTTP_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-us",
    "X-Language-Locale": "en-NO_NO",
    "User-Agent": "iOS(9.2)/Skyss(1.0.1)/1.0",
}


async def main(pers=None):

    close_pers_here: bool = False
    if pers is None:
        pers = persist.PyrotunPersistence()
        close_pers_here = True
        await pers.ainit(requested=["openhab"])

    bybanenfralagunen = await get_departures(pers, stop_id=SKYSS_STOPS["lagunen"])

    if "PassingTimes" in bybanenfralagunen:
        await pers.openhab.set_item(
            "NesteBybane", bybanenfralagunen["PassingTimes"][0]["DisplayTime"], log=True
        )
        await pers.openhab.set_item(
            "Neste2Bybane",
            bybanenfralagunen["PassingTimes"][1]["DisplayTime"],
            log=True,
        )
    else:
        logger.error(f"Mangling response from skyss: {bybanenfralagunen}")

    if close_pers_here:
        await pers.aclose()


async def get_departures(pers, stop_id: int, hours: int = 12) -> str:
    """Returns dict (from json) with departure information"""
    async with pers.websession.get(
        SKYSSAPI + "departures",
        params={"Hours": hours, "StopIdentifiers": stop_id},
    ) as response:
        if response.ok:
            return json.loads(_fix_strange_json(await response.text()))
        logger.error(f"Got {response.status} on url {response.real_url}")
        return ""


def _fix_strange_json(almostjson: str) -> str:
    """The API might return javascript(???) (maybe called JSONP)

    Remove that noise."""
    if almostjson.startswith("callback(") and almostjson.endswith(")"):
        return almostjson[:-1].replace("callback(", "")
    return almostjson


if __name__ == "__main__":
    asyncio.run(main())
