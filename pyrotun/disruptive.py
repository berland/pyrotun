import asyncio
import concurrent
import json

import requests
from requests.exceptions import ChunkedEncodingError

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)


BASE_URL = "https://dweet.io"

thing = "foobarcomfoobarcom"

sensors = {
    "bjehrg8pismg008hqa20": {"wateritem": "Sensor_Gulvutenfordusj_vann"},
    "bjehsk8pismg008hqac0": {"tempitem": "Sensor_Bakganggulv_temperatur"},
    "bjeiidu7gpvg00cjo7qg": {"tempitem": "Sensor_Teleskap_temperatur"},
    "bjr2lmtp0jt000a5h700": {
        "tempitem": "Sensor_Televeggverksted_temperatur",
        "humitem": "Sensor_Televeggverksted_fuktighet",
    },
    "bjek7udp0jt000aqcqe0": {"tempitem": "Sensor_TilluftSofastue_temperatur"},
    "bjeiks5ntbig00e43k8g": {"tempitem": "Sensor_InniVarmepumpe_temperatur"},
}


def process_dweets(pers):
    for dweet in listen_for_dweets_from("foobarcomfoobarcom", timeout=None):
        try:
            event = dweet["content"]["event"]
            data = event["data"]
            sensor = event["targetName"].split("/")[-1]
            logger.debug(json.dumps(dweet, indent=2, sort_keys=True))
            if event["eventType"] == "networkStatus":
                logger.debug(
                    "ping from disruptive sensor " + dweet["content"]["labels"]["name"]
                )
            elif event["eventType"] == "temperature":
                pers.openhab.sync_set_item(
                    sensors[sensor]["tempitem"],
                    data["temperature"]["value"],
                    log=True,
                )
            elif event["eventType"] == "humidity":
                pers.openhab.sync_set_item(
                    sensors[sensor]["humitem"],
                    data["humidity"]["relativeHumidity"],
                    log=True,
                )
                pers.openhab.sync_set_item(
                    sensors[sensor]["tempitem"],
                    data["humidity"]["temperature"],
                    log=True,
                )
            elif event["eventType"] == "waterPresent":
                pers.openhab.sync_set_item(
                    sensors[sensor]["wateritem"],
                    {"NOT_PRESENT": "OFF", "PRESENT": "ON"}[
                        data["waterPresent"]["state"]
                    ],
                    log=True,
                )
            else:
                logger.warning("Unprocessed Disruptive event! Unknown sensor?")
        except KeyError:
            print(f"Not able to parse dweet {dweet}")
            pass
        except ValueError as err:
            print("Misconfiguration in OpenHAB?")
            print(err)
            pass


def _listen_for_dweets_from_response(response):
    """Yields dweets as received from dweet.io's streaming API"""
    streambuffer = ""
    for byte in response.iter_content(chunk_size=2000):
        if byte:
            streambuffer += byte.decode("ascii")
            try:
                dweet = json.loads(streambuffer.splitlines()[1])
            except (IndexError, ValueError):
                continue
            if isinstance(dweet, str):
                yield json.loads(dweet)
            streambuffer = ""


def listen_for_dweets_from(thing_name, timeout=900, key=None, session=None):
    """Create a real-time subscription to dweets"""
    url = BASE_URL + "/listen/for/dweets/from/{0}".format(thing_name)
    session = session or requests.Session()
    if key is not None:
        params = {"key": key}
    else:
        params = None

    while True:
        try:
            request = requests.Request("GET", url, params=params).prepare()
            resp = session.send(request, stream=True, timeout=timeout)
            for x in _listen_for_dweets_from_response(resp):
                yield x
        except (
            ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ):
            pass


async def main(pers=None):
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit("openhab")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    result = await asyncio.get_event_loop().run_in_executor(
        executor, process_dweets, pers
    )
    print(f"main() is finished with result {result}")


if __name__ == "__main__":
    pers = pyrotun.persist.PyrotunPersistence()
    asyncio.get_event_loop().run_until_complete(main())
