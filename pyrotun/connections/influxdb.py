import datetime
import os
from typing import Optional, Union

import pandas as pd
from aioinflux import InfluxDBClient


class InfluxDBConnection:
    def __init__(self, host="", port=""):
        if host:
            self.host = host
        else:
            self.host = os.getenv("INFLUXDB_HOST")

        if port:
            self.port = port
        else:
            self.port = os.getenv("INFLUXDB_PORT")

        if not self.host:
            raise ValueError("INFLUXDB_HOST not provided")

        if not self.port:
            raise ValueError("INFLUXDB_PORT not provided")

        self.client = InfluxDBClient(
            output="dataframe", db="openhab_db", host=self.host, port=self.port
        )

    async def get_series(
        self,
        item,
        since: Optional[datetime.datetime] = None,
        upuntil: Optional[datetime.datetime] = None,
    ) -> pd.Series:
        sincestr = ""
        upuntilstr = ""
        if since is not None:
            sincestr = f"where time > '{since}'"
        if upuntil is not None:
            if sincestr == "":
                upuntilstr = f"where time < '{upuntil}'"
            else:
                upuntilstr = f" and time < '{upuntil}'"
        query = f"SELECT * FROM {item} {sincestr} {upuntilstr}"
        resp = await self.client.query(query)
        if isinstance(resp, dict) and not resp:
            return pd.Series()
        if isinstance(resp, pd.DataFrame) and resp.empty:
            return pd.Series()
        assert isinstance(resp, pd.DataFrame)
        if len(resp.columns) > 1:
            # OH3 changed how it writes to Influx series
            resp.columns = ["item", item]
        else:
            resp.columns = [item]
        return resp[[item]]

    async def get_series_grouped(
        self, item, aggregator="mean", time="1h", condition=""
    ) -> pd.DataFrame:
        query = (
            f"SELECT {aggregator}(value) FROM {item} {condition} group by time({time})"
        )
        resp = await self.client.query(query)
        resp.columns = [item]
        return resp

    async def get_item(self, item, ago=0, unit="h", datatype=None):
        resp = await self.client.query(
            f"SELECT last(value) FROM {item} where time < now() - {ago}{unit}"
        )
        if datatype is int:
            return int(resp["last"].values[0])
        if datatype is float:
            return float(resp["last"].values[0])
        if datatype is bool:
            return int(resp["last"].values[0]) == 1
        return str(resp["last"].values[0])

    async def get_lastvalue(self, item):
        resp = await self.client.query(f"SELECT last(value) FROM {item}")
        return resp["last"].values[0]

    async def item_age(self, item, unit="minutes"):
        resp = await self.client.query(f"SELECT last(value) FROM {item}")
        divisors = {"minutes": 60, "seconds": 1, "hours": 60 * 60}
        assert unit in divisors
        return (
            pd.Timestamp(datetime.datetime.utcnow(), tz="utc") - resp.index[0]
        ).seconds / divisors[unit]

    async def get_measurements(self):
        """Return a list of all measurements in database"""
        resp = await self.client.query("SHOW MEASUREMENTS")
        return resp["name"].values

    async def dframe_query(self, query) -> Union[dict, pd.DataFrame]:
        resp = await self.client.query(query)
        return resp
