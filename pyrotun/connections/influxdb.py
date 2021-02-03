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

        self.client = InfluxDBClient(
            output="dataframe", db="openhab_db", host=self.host
        )

    async def get_series(self, item):
        resp = await self.client.query(f"SELECT * FROM {item}")
        resp.columns = [item]
        return resp

    async def get_series_grouped(
        self, item, aggregator="mean", time="1h", condition=""
    ):

        resp = await self.client.query(
            (
                f"SELECT {aggregator}(value) FROM {item} {condition} "
                f"group by time({time})"
            )
        )
        resp.columns = [item]
        return resp

    async def get_lastvalue(self, item):
        resp = await self.client.query(f"SELECT last(value) FROM {item}")
        return resp["last"].values[0]

    async def get_measurements(self):
        """Return a list of all measurements in database"""
        resp = await self.client.query("SHOW MEASUREMENTS")
        return resp["name"].values

    async def dframe_query(self, query):
        resp = await self.client.query(query)
        return resp
