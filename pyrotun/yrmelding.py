import os
import asyncio

from lxml import objectify

import pyrotun
from pyrotun import persist
import pyrotun.connections.openhab

logger = pyrotun.getLogger()


async def main(pers=None):
    close_pers_here = False  # todo: use context manager?
    if pers is None:
        pers = persist.PyrotunPersistence()
        close_pers_here = True
    await pers.ainit(requested="openhab")

    weather = await get_weather(persistence=pers)

    for item, value in map_weatherdict_to_openhab(weather).items():
        await pers.openhab.set_item(item, round(value, 1), log=True)
    if close_pers_here:
        await pers.aclose()


def map_weatherdict_to_openhab(weather):
    return {
        "YrmeldingNaa": weather["weatherquality"][0],
        "YrmeldingNestetime": weather["weatherquality"][1],
        "YrmeldingNeste3timer": sum(weather["weatherquality"][0:3]) / 3,
        "YrmeldingNeste6timer": sum(weather["weatherquality"][0:3]) / 6,
        "YrMaksTempNeste6timer": max(weather["temperatures"]),
    }


async def get_weather(xmlfile=None, persistence=None):
    """Returns:

    dict with keys:
        nexthours: list of next hours forecast, 1 is nice weather, higher is
            worse.
        nexttemps: list of next temperatures, first is current hour
        currenttext: string with weather now
        nexthourtext: string with weather next hour
    """
    if xmlfile is None:
        xmlfile = os.getenv("FORECAST_XMLFILE")
    if not xmlfile:
        logger.error("provided xmlfile was empty")
        return
    logger.info("Downloading: %s", str(xmlfile))

    resp = await persistence.websession.get(xmlfile)
    xmlstring = await resp.read()  # read() to get bytes.
    # print(xmlstring)
    root = objectify.fromstring(xmlstring)
    # print(root)
    # root = objectify.parse(xmlfile).getroot()
    print(root)

    nexthours = [
        int(
            root.find("forecast")
            .find("tabular")
            .findall("time")[x]
            .symbol.attrib["number"]
        )
        for x in range(0, 6)
    ]
    nexttemps = [
        float(
            root.find("forecast")
            .find("tabular")
            .findall("time")[x]
            .temperature.attrib["value"]
        )
        for x in range(0, 6)
    ]

    currenttext = (
        root.find("forecast").find("tabular").findall("time")[0].symbol.attrib["name"]
    )
    nexthourtext = (
        root.find("forecast").find("tabular").findall("time")[1].symbol.attrib["name"]
    )

    return {
        "weatherquality": nexthours,
        "temperatures": nexttemps,
        "now_description": currenttext,
        "nexthour_description": nexthourtext,
    }


if __name__ == "__main__":
    asyncio.run(main())
