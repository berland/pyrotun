import asyncio
import dotenv

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)

PERS = None


async def main(pers=PERS):
    dotenv.load_dotenv()

    closepers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        closepers = True
    if pers.sectoralarm is None:
        await pers.ainit(["sectoralarm", "openhab"])

    hist = await pers.sectoralarm.history_df()
    armed = await pers.sectoralarm.armed(hist)
    locked = await pers.sectoralarm.locked(hist)

    oh_bool = {True: "ON", False: "OFF"}
    await pers.openhab.set_item("AlarmArmert", oh_bool[armed], log="change")
    await pers.openhab.set_item("DoorLocked", oh_bool[locked], log="change")

    if closepers:
        await pers.websession.close()

if __name__ == "__main__":
    PERS = pyrotun.persist.PyrotunPersistence()
    asyncio.get_event_loop().run_until_complete(main())
