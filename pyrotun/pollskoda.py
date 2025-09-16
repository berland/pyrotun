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
        positions = await myskoda.get_positions(os.getenv("SKODA_VIN"))

    if charge.status:
        await pers.openhab.set_item(
            "EnyaqChargeState",
            str(charge.status.state), log=True
        )
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
        await pers.openhab.set_item(
            "EnyaqChargeTarget",
            str(charge.settings.target_state_of_charge_in_percent),
                    log=True,
                )

    if positions:
        if "IN_MOTION" in str(positions.errors):
            await pers.openhab.set_item(
            "EnyaqInMotion",
                "ON",
                    log=True,
                )
        else:
            await pers.openhab.set_item(
            "EnyaqInMotion",
                "OFF",
                    log=True,
                )
        if positions.positions:
            lat = positions.positions[0].gps_coordinates.latitude
            lon = positions.positions[0].gps_coordinates.longitude
            await pers.openhab.set_item(
            "EnyaqCoordinates",
                f"{lat},{lon}",
                    log=True,
                )
            async with ClientSession() as session:
                response = await session.get(f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json")

                if response.status == 200:
                    data = await response.json()
                    address = data["address"]
                    country=""
                    if address["country"] not in ["Norge", "Norway"]:
                        country = f", {address['country']}"
                    await pers.openhab.set_item(
                    "EnyaqPositionHumanreadable",
                        f"{address['road']}, {address['neighbourhood']}, {address['county']}{country}",
                            log=True,
                        )

    if close_pers:
        await pers.aclose()



if __name__ == "__main__":
    asyncio.run(amain())
