import asyncio
import os

import dotenv
from aiohttp import ClientSession
from myskoda import MySkoda

import pyrotun
import pyrotun.persist

logger = pyrotun.getLogger(__name__)


async def amain(pers=None, debug=False):
    close_pers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        dotenv.load_dotenv()
        close_pers = True
    if pers.openhab is None:
        await pers.ainit(["openhab"])
        assert pers.openhab is not None

    async with ClientSession() as session:
        myskoda = MySkoda(session)
        await myskoda.connect(os.getenv("SKODA_USERNAME"), os.getenv("SKODA_PASSWORD"))
        charge = await myskoda.get_charging(os.getenv("SKODA_VIN"))

    if charge.status:
        await pers.openhab.set_item(
            "EnyaqBatteryState",
            str(charge.status.battery.state_of_charge_in_percent),
                    log=True,
                )
        await pers.openhab.set_item(
            "EnyaqRange",
            str(float(charge.status.battery.remaining_cruising_range_in_meters) / 1000),
                    log=True,
                )
        await pers.openhab.set_item(
            "EnyaqChargingPower",
            str(charge.status.charge_power_in_kw),
                    log=True,
                )

    if close_pers:
        await pers.aclose()



if __name__ == "__main__":
    asyncio.run(amain())
