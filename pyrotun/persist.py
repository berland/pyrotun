import asyncio
from typing import List, Union

import aiohttp

from pyrotun import connections, elvatunheating, getLogger, powermodels, waterheater

logger = getLogger(__name__)


class PyrotunPersistence:
    def __init__(self, readonly=False):
        logger.info("Initializing PyrotunPersistence")
        self.websession = aiohttp.ClientSession()

        self.hass = None
        self.homely = None
        self.influxdb = None
        self.openhab = None
        self.myuplink = None
        self.powermodels = None
        self.skoda = None
        self.smappee = None
        self.solis = None
        self.tibber = None
        self.unifiprotect = None
        self.waterheater = None
        self.yr = None
        self.elvatunheating = None

        # If true, no values are never sent anywhere
        self.readonly = readonly

    async def ainit(self, requested: Union[str, List[str]] = "all"):
        logger.info("PyrotunPersistence.ainit()")
        if "hass" in requested or "all" in requested:
            self.hass = connections.hass.HassConnection(websession=self.websession)
        if "homely" in requested or "all" in requested:
            self.homely = connections.homely.HomelyConnection(
                websession=self.websession
            )
            asyncio.create_task(self.homely.ainit(websession=self.websession))
        if "myuplink" in requested or "all" in requested:
            self.myuplink = connections.myuplink.MyuplinkConnection(
                websession=self.websession
            )
            asyncio.create_task(self.myuplink.ainit(websession=self.websession))
        if "openhab" in requested or "all" in requested:
            self.openhab = connections.openhab.OpenHABConnection(
                websession=self.websession, readonly=self.readonly
            )
        if "influxdb" in requested or "all" in requested:
            self.influxdb = connections.influxdb.InfluxDBConnection()

        if "waterheater" in requested or "all" in requested:
            self.waterheater = waterheater.WaterHeater()
            asyncio.create_task(self.waterheater.ainit(self))

        if "tibber" in requested or "all" in requested:
            self.tibber = connections.tibber.TibberConnection()
            await self.tibber.ainit(websession=self.websession)

        #if "smappee" in requested or "all" in requested:
        #self.smappee = connections.smappee.SmappeeConnection()

        #if "skoda" in requested or "all" in requested:
        #    self.skoda = connections.skoda.SkodaConnection()
        #    await self.skoda.ainit(websession=self.websession)

        if "solis" in requested or "all" in requested:
            self.solis = connections.solis.SolisConnection()
            await self.solis.ainit(websession=self.websession)

        if "yr" in requested or "all" in requested:
            self.yr = connections.yr.YrConnection()
            await self.yr.ainit(websession=self.websession)

        if "powermodels" in requested or "all" in requested:
            self.powermodels = powermodels.Powermodels()
            await self.powermodels.ainit(pers=self)

        if "elvatunheating" in requested:
            self.elvatunheating = elvatunheating.ElvatunHeating()
            await self.elvatunheating.ainit(pers=self)

    async def aclose(self):
        logger.info("Tearing down pyrotunpersistence")
        if self.tibber is not None:
            await self.tibber.aclose()

        if self.skoda is not None:
            await self.skoda.aclose()

        await self.websession.close()
