import pprint
import hashlib
import hmac
import json
import os
import time

import httpx
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

VERIFY_TOKEN = "vapourfly"

load_dotenv()
app = FastAPI()

TOKEN_FILE = "/home/berland/.stra_tokens"
CLIENT_SECRET = os.environ.get("STRAVA_CLIENT_SECRET")
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")


def load_tokens():
    try:
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_tokens(tokens):
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f)


def refresh_token_if_needed():
    tokens = load_tokens()
    if not tokens:
        raise Exception("No tokens found. User must authenticate first.")

    now = int(time.time())
    if tokens["expires_at"] <= now:
        print("Access token expired, refreshing...")
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

    print(f"Received event from Strava:")
    pprint.pprint(payload)
    # Handle different event types here
    activity_id = payload.get("object_id")
    print_activity_info(activity_id)
    await download_tcx(activity_id, "activity.tcx")
    if payload.get("aspect_type") == "create":
        update_activity(activity_id, "Raaserv app was here", "yay, long desc")
        # await download_tcx(activity_id, "activity.tcx")

    return "", 200


@app.get("/exchange_token")
async def exchange_token(code: str):
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
        }
    )


def print_activity_info(activity_id):
    access_token = refresh_token_if_needed()
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {"Authorization": f"Bearer {access_token}"}

    meta = requests.get(url, headers=headers)

    activity = meta.json()
    print("*** ACTIVITY INFO ***")
    pprint.pprint(activity)
    print("*** ACTIVITY END ***")


def update_activity(activity_id, new_title, new_description):
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
