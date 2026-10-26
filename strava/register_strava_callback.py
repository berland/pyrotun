"""One-off script to (re-)register the Strava webhook subscription.

Run this once after the FastAPI app (main.py) is up and reachable at
main.CALLBACK_BASE_URL (nginx on bobserv reverse-proxies that permanent
public URL to this machine's FastAPI port). Only needs to be re-run if
the callback URL ever changes.
"""

import os
import time

import requests
from dotenv import load_dotenv
from main import CALLBACK_BASE_URL, VERIFY_TOKEN

load_dotenv()

PORT = 5444
STRAVA_API = "https://www.strava.com/api/v3/push_subscriptions"
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")


def register_strava_webhook(public_url: str):
    callback_url = f"{public_url}/strava-webhook"

    # Optional: Delete existing subscriptions first
    print("Looking for existing subscriptions...")
    response = requests.get(
        STRAVA_API, params={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
    )
    if response.status_code == 200:
        print(response)
        for sub in response.json():
            print(f"Deleting existing subscriptions: {sub}")
            requests.delete(
                f"{STRAVA_API}/{sub['id']}",
                params={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET},
            )
    else:
        print(f"Could not talk to subscription API, got {response}")

    # Now register new webhook
    response = requests.post(
        STRAVA_API,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "callback_url": callback_url,
            "verify_token": VERIFY_TOKEN,
        },
    )

    if response.status_code == 201:
        print("Webhook registered successfully.")
    else:
        print("Failed to register webhook:", response.text)


def wait_for_fastapi(url=f"http://localhost:{PORT}/health", timeout=30, interval=1):
    """
    Wait until the FastAPI app responds at the given URL or timeout.

    Args:
        url (str): The URL to check (ideally a lightweight healthcheck endpoint)
        timeout (int): Max seconds to wait
        interval (int): Seconds between retries

    Returns:
        bool: True if server responded, False if timed out
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"FastAPI is up! Responded in {time.time() - start_time:.1f}s")
                return True
        except requests.ConnectionError:
            pass  # Server not up yet

        if time.time() - start_time > timeout:
            print(f"Timeout waiting for FastAPI after {timeout}s")
            return False

        time.sleep(interval)


if __name__ == "__main__":
    wait_for_fastapi()
    print(f"Registering Strava webhook at {CALLBACK_BASE_URL}...")
    register_strava_webhook(CALLBACK_BASE_URL)
