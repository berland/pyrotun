"""Bridging MQTT messages to Discord via Discord webhooks"""
import asyncio
import os

import asyncio_mqtt
import dotenv

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

dotenv.load_dotenv()

assert os.getenv("MQTT_HOST"), "You must proviode MQTT_HOST as an env variable"


async def main(pers=None):
    if pers is None:
        dotenv.load_dotenv()
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["websession"])
    assert pers.websession is not None

    async with asyncio_mqtt.Client(os.getenv("MQTT_HOST")) as client:
        async with client.messages() as messages:
            await client.subscribe("discordmessage/send")
            async for message in messages:
                await push_to_discord(message.payload.decode(), pers)


async def push_to_discord(message, pers):
    logger.info(f"Sending to discord: {message}")
    await pers.websession.post(
        url=os.getenv("DISCORD_WEBHOOK"), data={"content": message}
    )


if __name__ == "__main__":
    assert os.getenv("DISCORD_WEBHOOK"), "You must set the env variable DISCORD_WEBHOOK"
    asyncio.run(main())
