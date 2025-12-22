# Borrowed from Apache 2 code at
# https://github.com/ZuinigeRijder/SolisCloud2PVOutput
#
# API doc: https://solis-service.solisinverters.com/helpdesk/attachments/2043393248854

import base64
import datetime
import hashlib
import hmac
import json
import logging
import os

import dotenv
from aiohttp import ContentTypeError

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

dotenv.load_dotenv()
logger.setLevel(logging.DEBUG)

VERB = "POST"
CONTENT_TYPE = "application/json"
USER_STATION_LIST = "/v1/api/userStationList"
INVERTER_LIST = "/v1/api/inverterList"
INVERTER_DETAIL = "/v1/api/inverterDetail"


class SolisConnection:
    def __init__(self, websession=None):
        self.websession = websession
        self.api_id = None
        self.api_secret = None
        self.api_url = None

        self.inverter_id = None
        self.inverter_sn = None

    async def ainit(self, websession=None):
        if websession is not None:
            self.websession = websession
        self.api_id = os.getenv("SOLIS_KEYID")
        self.api_secret = os.getenv("SOLIS_KEYSECRET")
        self.api_url = os.getenv("SOLIS_API_URL")

        id_sn = await self.get_inverter_id_sn()
        self.inverter_id = id_sn["id"]
        self.inverter_sn = id_sn["sn"]

    async def get_solis_cloud_data(self, url_part, data) -> dict:
        assert self.api_secret is not None
        assert self.api_id is not None
        assert self.websession is not None
        md5 = base64.b64encode(hashlib.md5(data.encode("utf-8")).digest()).decode(
            "utf-8"
        )
        now = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%a, %d %b %Y %H:%M:%S GMT"
        )
        encrypt_str = (
            VERB + "\n" + md5 + "\n" + CONTENT_TYPE + "\n" + now + "\n" + url_part
        )
        hmac_obj = hmac.new(
            self.api_secret.encode("utf-8"),
            msg=encrypt_str.encode("utf-8"),
            digestmod=hashlib.sha1,
        )
        authorization = (
            "API "
            + self.api_id
            + ":"
            + base64.b64encode(hmac_obj.digest()).decode("utf-8")
        )
        headers = {
            "Content-MD5": md5,
            "Content-Type": CONTENT_TYPE,
            "Date": now,
            "Authorization": authorization,
        }
        async with self.websession.post(
            self.api_url + url_part, data=data.encode("utf-8"), headers=headers
        ) as response:
            try:
                json_response: dict = await response.json()
            except ContentTypeError as err:
                logger.error(err)
                return {}
            return json_response

    async def get_data(self):
        data: dict = await self.get_solis_cloud_data(
            INVERTER_DETAIL,
            json.dumps({"id": self.inverter_id, "sn": self.inverter_sn}),
        )
        # Fix units
        if (
            "data" in data
            and "apparentPowerStr" in data.get("data", [])
            and data["data"]["apparentPowerStr"] == "kVA"
        ):
            data["data"]["apparentPower"] *= 1000
            data["data"]["apparentPowerStr"] = "VA"
        return data

    async def get_inverter_id_sn(self) -> dict:
        assert self.api_id is not None
        content: dict = await self.get_solis_cloud_data(
            USER_STATION_LIST, '{"userid":"' + self.api_id + '"}'
        )
        station_info = content["data"]["page"]["records"][0]
        inverter_data: dict = await self.get_solis_cloud_data(
            INVERTER_LIST, '{"stationId":"' + station_info["id"] + '"}'
        )
        inverter_info = inverter_data["data"]["page"]["records"][0]
        return {"id": inverter_info["id"], "sn": inverter_info["sn"]}
