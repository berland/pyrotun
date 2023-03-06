import os

import aiohttp
import dotenv

import pyrotun

logger = pyrotun.getLogger(__name__)
dotenv.load_dotenv()


class HassConnection:
    def __init__(self, hass_url="", websession=None):
        if not hass_url:
            self.hass_url = os.getenv("HASS_URL")
        else:
            self.hass_url = hass_url

        if not self.hass_url:
            raise ValueError("hass_url not set")

        if websession is not None:
            self.websession = websession
        else:
            # For async usage.
            self.websession = aiohttp.ClientSession()

        self.token = os.getenv("HASS_TOKEN")
        assert self.token, "HASS_TOKEN must be set as an env variable"

    async def get_item(self, entity_id, attribute=None, datatype=str):
        async with self.websession.get(
            self.hass_url + "/api/states/" + str(entity_id),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        ) as resp:
            resp = await resp.json()
            if attribute is None:
                return resp["state"]
            return resp["attributes"][attribute]

    async def set_item(self, service_path, entity_id, attribute_name, new_state):
        service_path = service_path.strip("/")
        url = f"{self.hass_url}/api/services/{service_path}"
        data = {"entity_id": entity_id, attribute_name: str(new_state)}
        async with self.websession.post(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json=data,
        ) as resp:
            if resp.status not in [200, 201]:
                logger.error(resp)
