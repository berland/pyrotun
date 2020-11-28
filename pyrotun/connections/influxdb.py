import os
import pytz

from aioinflux import InfluxDBClient

class InfluxDBConnection:
    def __init__(self, host=""):

        if host:
            self.host = host
        else:
            self.host = os.getenv("INFLUXDB_HOST")

        if not self.host:
            raise ValueError("INFLUXDB_HOST not provided")

        self.client = InfluxDBClient(output="dataframe", db="openhab_db", host=self.host)

    async def get_series(self, item):
        resp = await self.client.query("SELECT * FROM " + item)
        return resp

    async def get_lastvalue(self, item):
        resp = await self.client.query("SELECT last(value) FROM " + item)
        return resp["last"].values[0]
