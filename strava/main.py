import datetime
import json
import logging
import os
import pprint
import time
from pathlib import Path

import httpx
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from pyrotun import exercise_analyzer

app = FastAPI()
import asyncio

VERIFY_TOKEN = "vapourfly"
logger = logging.getLogger(__name__)
load_dotenv()
app = FastAPI()

TOKEN_FILE = "/home/berland/.stra_tokens"
CLIENT_SECRET = os.environ.get("STRAVA_CLIENT_SECRET")
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")


logging.basicConfig(level=logging.INFO)

HEARTBEAT = 240  # seconds
async def heartbeat_logger():
    while True:
        counter = 0
        while counter < HEARTBEAT:
            counter+=1
            await asyncio.sleep(1)  # Uvicorn will not exit during this second
        logger.info(f" [ strava-app heartbeat ({HEARTBEAT}s) ] ")


def load_tokens():
    try:
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise Exception(f"No tokens found. Construct {TOKEN_FILE}") from e


def save_tokens(tokens):
    logger.info("Persisting tokens to disk")
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)


def refresh_token_if_needed(force=False):
    tokens = load_tokens()
    now = int(time.time())
    if force:
        logger.info("Force refresh of tokens")
    if force or ("expires_at" not in tokens or tokens["expires_at"] <= now):
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
    return tokens["access_token"]


@app.get("/strava-webhook")
def verify_subscription(request: Request):
    params = dict(request.query_params)
    print(params)
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return {"hub.challenge": params.get("hub.challenge")}
    return JSONResponse(status_code=403, content={"error": "Invalid verify token"})


@app.post("/strava-webhook")
async def receive_event(payload: dict):
    print("Received event from Strava:")
    pprint.pprint(payload)

    activity_id = payload.get("object_id")
    if payload.get("aspect_type") == "delete":
        return "", 200

    await process_activity_update(activity_id, payload.get("aspect_type", ""))

    return "", 200



@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/oauth/callback")
async def exchange_token(request: Request):
    """This function is called by Strava when the User clicks 'Authorize' to authorize
    this app. To get to the authorize webpage, use a browser where user is logged in at Strava
    and visit

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
    #pprint.pprint(activity)
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
                print(f"Computed updates, but not submitting ('Run' not present in name): {updates}")
        else:
            logger.error(f"TCX file never appeared on disk for {activity['start_date_local']=}")


    if aspect_type == "create":
        await check_and_notify_about_undefined_shoe(activity)


async def check_and_notify_about_undefined_shoe(activity: dict):
    if activity.get("manual"):
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

        pushover_resp = requests.post("https://api.pushover.net/1/messages.json", data=data)
        print("Pushover sent:", pushover_resp.status_code)

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
    base_dir=Path("/home/berland/polar_dump")
    fmt = "%Y-%m-%dT%H:%M:%S"
    base_time = datetime.datetime.strptime(iso_timestamp_from_strava.rstrip("Z"), fmt)

    for delta_sec in range(-seconds_range, seconds_range + 1):
        candidate_time = base_time + datetime.timedelta(seconds=delta_sec)
        candidate_path = base_dir / candidate_time.strftime(fmt) / "tcx"
        if candidate_path.exists():
            return candidate_path.parent  # Return the first match
    return None

async def download_tcx(activity_id: str, save_path: str) -> None:
    access_token = refresh_token_if_needed()
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    tcx_url = f"https://www.strava.com/activities/{activity_id}/export_tcx"
    headers = {"Authorization": f"Bearer {access_token}"}
    print(f"Headers when downloacing tcx: {headers}")

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    # print("*** ACT info again")
    # pprint.pprint(response.text)

    async with httpx.AsyncClient() as client:
        response = await client.get(tcx_url, headers=headers)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Failed to download TCX"
        )

    print(response.text)

def upper_dict_layer(d: dict) -> dict:
    return {k: v for k, v in dict(d).items() if not isinstance(v, dict) and not isinstance(v, list)}
