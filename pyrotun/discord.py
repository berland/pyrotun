"""Bridging MQTT messages to Discord via Discord webhooks"""
import os
import asyncio

import dotenv

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

dotenv.load_dotenv()


async def main(pers=None):
    if pers is None:
        dotenv.load_dotenv()
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["mqtt", "websession"])
    assert pers.mqtt is not None
    assert pers.websession is not None

    # Add MQTT listener:
    manager = pers.mqtt.client.filtered_messages("discordmessage/send")
    # Get message generator from context manager:
    messages = await manager.__aenter__()
    # Add a task to process the message generator:
    task = asyncio.create_task(push_many_to_discord(messages, pers))

    await pers.mqtt.client.subscribe("discordmessage/#")

    await asyncio.gather(task)


async def push_many_to_discord(messages, pers):
    """Loop over an async generator that provides messages to push
    to Discord"""
    async for message in messages:
        await push_to_discord(message.payload.decode(), pers)


async def push_to_discord(message, pers):
    logger.info("Sending to discord: %s", message)
    url = os.getenv("DISCORD_WEBHOOK")
    data = {"content": message}
    await pers.websession.post(url=str(url), data=data)


if __name__ == "__main__":
    assert os.getenv("DISCORD_WEBHOOK"), "You must set the env variable DISCORD_WEBHOOK"
    asyncio.run(main())
