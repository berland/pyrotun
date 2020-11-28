import aiohttp

from pyrotun import connections, waterheater, getLogger

logger = getLogger(__name__)


class PyrotunPersistence:
    def __init__(self):
        logger.info("Initializing PyrotunPersistence")
        self.websession = aiohttp.ClientSession()
        self.tibber = None
        self.influxdb = None
        self.waterheater = None
        self.smappee = None
        self.openhab = None

    async def ainit(self, requested="all"):
        logger.info("PyrotunPersistence.ainit()")
        if "openhab" in requested or "all" in requested:
            self.openhab = connections.openhab.OpenHABConnection(
                websession=self.websession
            )
        if "influxdb" in requested or "all" in requested:
            self.influxdb = connections.influxdb.InfluxDBConnection()

        if "waterheater" in requested or "all" in requested:
            self.waterheater = waterheater.WaterHeater()
            await self.waterheater.ainit(self)

        if "tibber" in requested or "all" in requested:
            self.tibber = connections.tibber.TibberConnection()
            await self.tibber.ainit(websession=self.websession)

        if "smappee" in requested or "all" in requested:
            self.smappee = connections.smappee.SmappeeConnection()

    async def aclose(self):
        logger.info("Tearing down pyrotunpersistence")
        if self.tibber is not None:
            await self.tibber.aclose()
        await self.websession.close()
