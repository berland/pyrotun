import os
import time

import requests
from dotenv import load_dotenv
from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokError

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
    ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))
    http_tunnel = ngrok.connect(PORT, bind_tls=True)
    print(f"ngrok tunnel started at {http_tunnel.public_url}")
    return http_tunnel.public_url


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


if __name__ == "__main__":
    print("Starting ngrok...")
    try:
        public_url = start_ngrok()
    except PyngrokNgrokError:
        print("Waiting for previous ngrok connection to deregister..")
        time.sleep(5)
        public_url = start_ngrok()

    print("Registering Strava webhook...")
    register_strava_webhook(public_url)

    try:
        while True:
            print(" [ ngrok-proxy heartbeat (240s) ]")
            time.sleep(240)
    except KeyboardInterrupt:
        print("Shutting down")
