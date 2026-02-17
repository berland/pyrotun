import asyncio
import datetime
import json
import logging
import os
import pprint
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import httpx
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from pyrotun import exercise_analyzer

VERIFY_TOKEN = "vapourfly"  # Only used for initial handshake
logger = logging.getLogger(__name__)
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task1 = asyncio.create_task(heartbeat_logger())
    task2 = asyncio.create_task(shoe_poller())
    yield
    task1.cancel()
    task2.cancel()
    with suppress(asyncio.CancelledError):
        await task1
    with suppress(asyncio.CancelledError):
        await task2


app = FastAPI(lifespan=lifespan)

TOKEN_FILE = Path("/home/berland/.stra_tokens")
CLIENT_SECRET = os.environ.get("STRAVA_CLIENT_SECRET")
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")

SHOE_NOTIFICATIONS_SENT: set[str] = set()
GEAR_ID_PR_DATE: dict[datetime.date, str] = {}
ACTIVITIES_TO_BE_POLLED_FOR_GEAR: set[str] = set()
UNDEFINED_SHOE: str = "g25935987"

HEARTBEAT = 240  # seconds

logging.basicConfig(level=logging.INFO)


async def heartbeat_logger():
    while True:
        counter = 0
        logger.info(f" [ strava-app heartbeat ({HEARTBEAT}s) ] ")
        logger.info(f"   {ACTIVITIES_TO_BE_POLLED_FOR_GEAR=}")
        logger.info(f"   {GEAR_ID_PR_DATE=}")
        logger.info(f"   {SHOE_NOTIFICATIONS_SENT=}")
        while counter < HEARTBEAT:
            counter += 1
            await asyncio.sleep(1)


async def shoe_poller():
    while True:
        for activity_id in ACTIVITIES_TO_BE_POLLED_FOR_GEAR:
            logger.info(f"Polling activity {activity_id} for gear")
            activity = await get_activity(activity_id)
            if activity["gear_id"] != UNDEFINED_SHOE:
                logger.info(f" * Success, found gear to be {activity['gear_id']}")
                GEAR_ID_PR_DATE[
                    datetime.datetime.fromisoformat(
                        str(activity["start_date_local"])
                    ).date()
                ] = activity["gear_id"]
                ACTIVITIES_TO_BE_POLLED_FOR_GEAR.remove(activity_id)
                await asyncio.sleep(1)
        await asyncio.sleep(60 * 5)


def load_tokens() -> dict[str, str | int]:
    try:
        return json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise Exception(f"No tokens found. Construct {TOKEN_FILE}") from e


def save_tokens(tokens: dict[str, str | int]):
    logger.info("Persisting tokens to disk")
    TOKEN_FILE.write_text(json.dumps(tokens, indent=2), encoding="utf-8")


def refresh_token_if_needed(force: bool = False) -> str:
    tokens = load_tokens()
    now = int(time.time())
    if force:
        logger.info("Force refresh of tokens")
    if force or ("expires_at" not in tokens or int(tokens["expires_at"]) <= now):
        if not force:
            logger.info("Access token expired, refreshing...")
        response = requests.post(
            "https://www.strava.com/oauth/token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
            },
        )
        if response.status_code == 200:
            new_tokens = response.json()
            # Update tokens dict with new values
            tokens["access_token"] = new_tokens["access_token"]
            tokens["refresh_token"] = new_tokens["refresh_token"]
            tokens["expires_at"] = new_tokens["expires_at"]
            tokens["expires_at_iso"] = datetime.datetime.fromtimestamp(
                new_tokens["expires_at"]
            ).isoformat()
            save_tokens(tokens)
            print("Tokens refreshed and saved.")
        else:
            raise Exception(
                f"Failed to refresh token: {response.status_code} {response.text}"
            )
    return str(tokens["access_token"])


@app.get("/strava-webhook")
def verify_subscription(request: Request):
    params = dict(request.query_params)
    print(params)
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return {"hub.challenge": params.get("hub.challenge")}
    return JSONResponse(status_code=403, content={"error": "Invalid verify token"})


@app.post("/strava-webhook")
async def receive_event(payload: dict) -> tuple[str, int]:
    print("Received event from Strava:")
    pprint.pprint(payload)

    activity_id = payload.get("object_id")
    if payload.get("aspect_type") == "delete":
        return "", 200

    await process_activity_update(activity_id, payload.get("aspect_type", ""))

    return "", 200


async def get_activity(activity_id: str) -> dict:
    access_token = refresh_token_if_needed()
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


def print_athlete_info():
    access_token = refresh_token_if_needed()
    url = "https://www.strava.com/api/v3/athlete"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)

    print(response)
    print(response.json())


async def process_activity_update(activity_id: str, aspect_type: str):
    access_token = refresh_token_if_needed()
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    print(url)
    print(headers)
    response = requests.get(url, headers=headers)

    print(response)
    activity: dict = response.json()
    print("*** ACTIVITY INFO ***")
    pprint.pprint(upper_dict_layer(activity))
    # pprint.pprint(activity)
    print("*** ACTIVITY END ***")

    updates = {}
    if activity.get("device_name", "") == "Zwift Run":
        start_time = datetime.datetime.fromisoformat(str(activity["start_date_local"]))
        if float(activity.get("average_speed", "5")) < 2.54:  # 6:30 min/km
            print("Neppe jeg som har løpt på Zwift, setter til privat")
            updates["private"] = True
            updates["visibility"] = "only_me"
        elif start_time.weekday() == 0 and 8 < start_time.hour < 11:  # Mandag
            updates["name"] = "Gym på jobben"
            updates["description"] = "Zwift"
        print(f"Submitting activity updates: {updates}")
        update_activity(activity_id, updates)

    updates = await exercise_analyzer.make_description_from_stravaactivity(activity)
    if updates and "Run" in activity["name"]:
        print(f"Submitting activity updates: {updates}")
        if "Til jobb" in updates["name"]:
            ACTIVITIES_TO_BE_POLLED_FOR_GEAR.add(activity["id"])
        update_activity(activity_id, updates)

    if "start_date_local" in activity:
        counter = 0
        tcxfilename = None
        while counter < 30 * 60 and tcxfilename is None:
            tcxfilename = find_nearby_file(activity["start_date_local"])
            counter += 1
            await asyncio.sleep(1)
        if tcxfilename:
            updates = await exercise_analyzer.make_description_from_tcx(tcxfilename)
            if updates and "Run" in activity["name"]:
                print(f"Submitting activity updates: {updates}")
                update_activity(activity_id, updates)
            else:
                print(
                    "Computed updates, but not submitting "
                    f"('Run' not present in name): {updates}"
                )
        else:
            logger.error(
                f"TCX file never appeared on disk for {activity['start_date_local']=}"
            )

    if aspect_type == "create" and "Run" in activity["name"]:
        await check_and_notify_about_undefined_shoe(activity)

    if aspect_type == "update" and "jobb" in activity.get("name", ""):
        activity_date = datetime.datetime.fromisoformat(
            activity.get("start_date_local", "")
        ).date()
        if (
            "Til jobb" in activity.get("name", "")
            and activity.get("gear_id", "") != UNDEFINED_SHOE
        ):
            GEAR_ID_PR_DATE[activity_date] = activity.get("gear_id", "")
            print(f"User actually has set gear id to something: {GEAR_ID_PR_DATE=}")
        if (
            "Hjem fra jobb" in activity.get("name", "")
            and activity.get("gear_id", "") == UNDEFINED_SHOE
            and activity_date in GEAR_ID_PR_DATE
        ):
            updates["gear_id"] = GEAR_ID_PR_DATE[activity_date]
            print(f"Sending update, now we have shoe pair: {updates}")
            update_activity(activity_id, updates)


async def check_and_notify_about_undefined_shoe(activity: dict):
    if activity.get("manual"):
        return

    if activity["id"] in SHOE_NOTIFICATIONS_SENT:
        return

    gear = activity.get("gear", {})
    if "Undefined" in gear.get("name", ""):
        edit_url = f"https://www.strava.com/activities/{activity['id']}/edit"
        message = f"⚠️ Velg sko for Strava-økt: {edit_url}"

        data = {
            "token": os.getenv("PUSHOVER_APIKEY", ""),
            "user": os.getenv("PUSHOVER_USER", ""),
            "title": "Strava skovalg",
            "message": message,
        }

        pushover_resp = requests.post(
            "https://api.pushover.net/1/messages.json", data=data
        )
        print("Pushover sent:", pushover_resp.status_code)
        SHOE_NOTIFICATIONS_SENT.add(activity["id"])


def update_activity(activity_id: str, data):
    access_token = refresh_token_if_needed()
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.put(url, headers=headers, data=data)

    if response.status_code == 200:
        print("✅ Activity updated.")
        return response.json()
    else:
        print(f"❌ Failed to update activity: {response.status_code}")
        print(response.text)
        return None


def find_nearby_file(iso_timestamp_from_strava, seconds_range=3) -> Path | None:
    base_dir = Path("/home/berland/polar_dump")
    fmt = "%Y-%m-%dT%H:%M:%S"
    base_time = datetime.datetime.strptime(iso_timestamp_from_strava.rstrip("Z"), fmt)

    for delta_sec in range(-seconds_range, seconds_range + 1):
        candidate_time = base_time + datetime.timedelta(seconds=delta_sec)
        candidate_path = base_dir / candidate_time.strftime(fmt) / "tcx"
        if candidate_path.exists():
            return candidate_path.parent  # Return the first match
    return None


def upper_dict_layer(d: dict) -> dict:
    """Strips dict values that are dicts or lists"""
    return {
        k: v
        for k, v in dict(d).items()
        if not isinstance(v, dict) and not isinstance(v, list)
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/oauth/callback")
async def exchange_token(request: Request):
    """This function is called by Strava when the User clicks 'Authorize' to
    authorize this app. To get to the authorize webpage, use a browser where
    user is logged in at Strava and visit

    https://www.strava.com/oauth/authorize?client_id=XXXXXXX&response_type=code&redirect_uri=https://XXXXXXXXX.ngrok-free.app/oauth/callback&scope=read,activity:read_all,activity:write&approval_prompt=force
    """
    params = dict(request.query_params)
    code = params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code parameter")
    token_url = "https://www.strava.com/api/v3/oauth/token"

    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=data)

    if response.status_code != 200:
        return "Invalid signature", 403

    tokens = response.json()

    save_tokens(
        {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "expires_at": tokens["expires_at"],
            "expires_at_iso": datetime.datetime.fromtimestamp(
                tokens["expires_at"]
            ).isoformat(),
        }
    )
    return RedirectResponse(url="/dashboard")
