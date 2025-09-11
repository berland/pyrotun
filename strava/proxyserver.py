import os
import time

import requests
from dotenv import load_dotenv
from pyngrok import ngrok

load_dotenv()

PORT = 5444
STRAVA_API = "https://www.strava.com/api/v3/push_subscriptions"
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
VERIFY_TOKEN = "vapourfly"

print(CLIENT_ID)
print(CLIENT_SECRET)


def start_ngrok():
    # Open a HTTP tunnel on the specified port
    http_tunnel = ngrok.connect(PORT, bind_tls=True)
    print(f"ngrok tunnel started at {http_tunnel.public_url}")
    return http_tunnel.public_url


def register_strava_webhook(public_url):
    callback_url = f"{public_url}/strava-webhook"

    # Optional: Delete existing subscriptions first
    existing = requests.get(
        STRAVA_API, params={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
    )
    for sub in existing.json():
        print(f"Deleting existing subscriptions: {sub}")
        requests.delete(
            f"{STRAVA_API}/{sub['id']}",
            params={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET},
        )

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


if __name__ == "__main__":
    public_url = start_ngrok()
    register_strava_webhook(public_url)

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Shutting down")
