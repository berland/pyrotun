import aiohttp

from pyrotun import connections, getLogger, powermodels, waterheater

logger = getLogger(__name__)


class PyrotunPersistence:
    def __init__(self, readonly=False):
        logger.info("Initializing PyrotunPersistence")
        self.websession = aiohttp.ClientSession()
        self.tibber = None
        self.influxdb = None
        self.waterheater = None
        self.smappee = None
        self.sectoralarm = None
        self.openhab = None
        self.mqtt = None
        self.unifiprotect = None
        self.yr = None
        self.powermodels = None

        # If true, no values are ever sent anywhere
        self.readonly = readonly

    async def ainit(self, requested="all"):
        logger.info("PyrotunPersistence.ainit()")
        if "openhab" in requested or "all" in requested:
            self.openhab = connections.openhab.OpenHABConnection(
                websession=self.websession, readonly=self.readonly
            )
        if "influxdb" in requested or "all" in requested:
            self.influxdb = connections.influxdb.InfluxDBConnection()

        if "waterheater" in requested or "all" in requested:
            self.waterheater = waterheater.WaterHeater()
            await self.waterheater.ainit(self)

        if "tibber" in requested or "all" in requested:
            self.tibber = connections.tibber.TibberConnection()
            await self.tibber.ainit(websession=self.websession)

        if "sectoralarm" in requested or "all" in requested:
            self.sectoralarm = connections.sectoralarm.SectorAlarmConnection()
            await self.sectoralarm.ainit(websession=self.websession)

        if "smappee" in requested or "all" in requested:
            self.smappee = connections.smappee.SmappeeConnection()

        if "mqtt" in requested or "all" in requested:
            self.mqtt = connections.mqtt.MqttConnection()
            await self.mqtt.ainit()

        if "yr" in requested or "all" in requested:
            self.yr = connections.yr.YrConnection()
            await self.yr.ainit(websession=self.websession)

        if "powermodels" in requested or "all" in requested:
            self.powermodels = powermodels.Powermodels()
            await self.powermodels.ainit(pers=self)

        if "unifiprotect" in requested or "all" in requested:
            self.unifiprotect = connections.unifiprotect.UnifiProtectConnection()
            await self.unifiprotect.ainit()

    async def aclose(self):
        logger.info("Tearing down pyrotunpersistence")
        if self.tibber is not None:
            await self.tibber.aclose()

        await self.websession.close()

        if self.mqtt is not None:
            await self.mqtt.aclose()
