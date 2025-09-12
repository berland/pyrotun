import datetime
import hashlib
import hmac
import json
import logging
import os
import pprint
import time

import httpx
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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


async def heartbeat_logger():
    while True:
        logger.info(" [ strava-app heartbeat (180s) ] ")
        await asyncio.sleep(180)  # Log every 60 seconds


@app.on_event("startup")
async def startup_event():
    await asyncio.sleep(5)
    # refresh_token_if_needed(force=True)
    # print_athlete_info()
    # print_activity_info(15786634920)
    # print_activity_info("15786634920")
    # print_activity_info("16634920")
    asyncio.create_task(heartbeat_logger())


def load_tokens():
    try:
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_tokens(tokens):
    logger.info("Persisting tokens to disk")
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)


def refresh_token_if_needed(force=False):
    tokens = load_tokens()
    if not tokens:
        raise Exception(f"No tokens found. Construct {TOKEN_FILE}")

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
    if False:
        print("THERE was a signature")
        print(payload)
        raw_body = payload
        expected_signature = hmac.new(
            key=CLIENT_SECRET.encode("utf-8"),
            msg=raw_body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            return "Invalid signature", 403

    print("Received event from Strava:")
    pprint.pprint(payload)
    # Handle different event types here
    activity_id = payload.get("object_id")
    print_activity_info(activity_id)
    # await download_tcx(activity_id, "activity.tcx")
    if payload.get("aspect_type") == "create":
        update_activity(activity_id, "Morgenrun", "..")
        # await download_tcx(activity_id, "activity.tcx")

    return "", 200

    print()


@app.get("/oauth/callback")
async def exchange_token(request: Request):
    """This function is called by Strava when the User clicks 'Authorize' to authorize
    this app. To get to the authorize webpage, use a browser where user is logged in at Strava
    and visit

    https://www.strava.com/oauth/authorize?client_id=XXXXXXX&response_type=code&redirect_uri=https://XXXXXXXXX.ngrok-free.app/oauth/callback&scope=read,activity:read_all&approval_prompt=force
    """
    params = dict(request.query_params)
    code = params.get("code")
    state = params.get("state")  # Optional, for CSRF protection if you use it
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


def print_athlete_info():
    access_token = refresh_token_if_needed()
    url = "https://www.strava.com/api/v3/athlete"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)

    print(response)
    print(response.json())


def print_activity_info(activity_id: str):
    access_token = refresh_token_if_needed()
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    print(url)
    print(headers)
    response = requests.get(url, headers=headers)

    print(response)
    activity = response.json()
    print("*** ACTIVITY INFO ***")
    pprint.pprint(activity)
    print("*** ACTIVITY END ***")


def update_activity(activity_id: str, new_title: str, new_description: str):
    access_token = refresh_token_if_needed()
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    data = {"name": new_title, "description": new_description}
    response = requests.put(url, headers=headers, data=data)

    if response.status_code == 200:
        print("✅ Activity updated.")
        return response.json()
    else:
        print(f"❌ Failed to update activity: {response.status_code}")
        print(response.text)
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
