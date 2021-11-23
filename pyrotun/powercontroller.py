"""Module for controlling/truncating house power usage

To be run()'ed every minute.

"""
import asyncio
import datetime
from pathlib import Path
from typing import Dict, List

import aiofiles
import numpy as np
import pandas as pd
import yaml

from pyrotun import persist

CURRENT_POWER_ITEM = "AMSpower"
HOURUSAGE_ESTIMATE_ITEM = "WattHourEstimate"
MAXHOURWATT_LASTMONTH_ITEM = "MaxHourwatt_lastmonth"
FLOORSFILE = Path(__file__).absolute().parent / "floors.yml"


def run(pers, dry=True):

    temp_plan = pers.powermodels["temperatureplan"]
    assert isinstance(temp_plan, pd.DataFrame)


async def get_powerloads(pers) -> pd.DataFrame:
    """Make a dataframe with columns:
    * openhab_switch_name (none if controlled via setpoint)
    * openhab_setpoint
    * is_on (None, "YES" or "NO") guess if this is load is on
    * floorname  # None if not mentioned in floor.yml
    * current_setpoint_state
    * planned_setpoint
    * lastchange (default 60, in minutes
    * on_need  - higher values means important to turn on
    * wattage"""

    loads: List[dict] = []

    loads.append(
        {
            "switch_item": "Verksted_varmekabler_nattsenking",
            "inverted_switch": True,
            "wattage": 1700,
            "lastchange": await pers.influxdb.item_age(
                "Verksted_varmekabler_nattsenking", unit="minutes"
            ),
            "on_need": max(
                12 - await pers.openhab.get_item("VerkstedTemperatur", datatype=float),
                0,
            ),
        }
    )
    loads.append(
        {
            "switch_item": "Garasje_varmekabler_nattsenking",
            "inverted_switch": True,
            "wattage": 1700,
            "lastchange": await pers.influxdb.item_age(
                "Garasje_varmekabler_nattsenking", unit="minutes"
            ),
            "on_need": max(
                6
                - await pers.openhab.get_item(
                    "Sensor_Garasje_temperatur", datatype=float
                ),
                0,
            ),
        }
    )

    loads.append(
        {
            "switch_name": "Varmtvannsbereder_bryter",
            "wattage": 2800,
            "is_on": await pers.openhab.get_item("Varmtvannsbereder_bryter"),
            "lastchange": await pers.influxdb.item_age(
                "Varmtvannsbereder_bryter", unit="minutes"
            ),
        }
    )

    async with aiofiles.open(FLOORSFILE, "r", encoding="utf-8") as filehandle:
        contents = await filehandle.read()
    floors = yaml.safe_load(contents)

    # Varmepumpe?

    for floor in floors:
        thisfloor = floors[floor].copy()  # needed?
        print(floor)
        if isinstance(thisfloor["setpoint_item"], list):
            bryter_name = thisfloor["setpoint_item"][0].replace(
                "SetpointHeating", "bryter"
            )
        else:
            bryter_name = thisfloor["setpoint_item"].replace(
                "SetpointHeating", "bryter"
            )

        if "sensor_item" in thisfloor.keys():
            meas_temp = await pers.openhab.get_item(
                thisfloor["sensor_item"], datatype=float
            )
            target_temp = 25 + thisfloor.get("delta", 0)
            if meas_temp is not None:
                on_need = max(target_temp - meas_temp, 0)
                thisfloor.update(
                    {
                        "on_need": on_need,
                    }
                )
            if isinstance(thisfloor["setpoint_item"], str):
                # Can be a list for floors with multiple thermostats
                thisfloor.update(
                    {
                        "is_on": await pers.openhab.get_item(bryter_name),
                    }
                )
        else:
            print(f"no sensor for floor {floor}")
        thisfloor.update(
            {
                "lastchange": await pers.influxdb.item_age(
                    bryter_name,
                    unit="minutes",
                )
            }
        )

        loads.append(thisfloor)

    return pd.DataFrame(loads).drop(["cooling_rate", "heating_rate"], axis="columns")


async def control_powerusage(pers) -> None:
    """Run this to contain the power usage to set limits"""

    # A dataframe that contains the hourly plan for power usage
    # the coming hours, indexed by datetimes in local time zone
    # Must have a planned_Wh column
    powerplan = None  # pers.powerplan.get_planned_wh()

    if powerplan is None:
        powerplan = 4800

    estimated_wh = pers.openhab.get_item("EstimatedKWh_thishour")

    overshoot = estimated_wh - powerplan

    powerload_df = await get_powerloads()

    actions = _decide(overshoot, powerload_df)
    for action, appliance in actions.items():
        turn(action, appliance)


def _decide(overshoot: int, powerload_df: pd.DataFrame):
    assert isinstance(overshoot, int)
    assert isinstance(powerload_df, pd.DataFrame)
    actions: List[Dict[str, dict]] = []
    if powerload_df.empty:
        return actions
    if "lastchange" not in powerload_df.columns:
        powerload_df["lastchange"] = np.nan
    if "is_on" not in powerload_df.columns:
        powerload_df["is_on"] = np.nan
    if "on_need" not in powerload_df.columns:
        powerload_df["on_need"] = np.nan

    # Need an index as a column
    powerload_df.reset_index(inplace=True)

    # Sort all rows for candidacy for change:
    powerload_df["change_candidate"] = 0
    powerload_df.loc[
        (powerload_df["lastchange"] > 5) & (powerload_df["is_on"] != "NO"),
        "change_candidate",
    ] = 1
    powerload_df.loc[
        (powerload_df["lastchange"] > 5) & (powerload_df["is_on"] != "YES"),
        "change_candidate",
    ] = -1
    if overshoot > 0:
        remainder_overshoot = overshoot

        while remainder_overshoot > 0 and not powerload_df.empty:
            turnmeoff = (
                powerload_df.sort_values(["change_candidate", "on_need"])
                .tail(1)
                .to_dict(orient="records")[0]
            )
            actions.append({"OFF": turnmeoff})
            powerload_df.drop(axis=0, index=turnmeoff["index"], inplace=True)
            remainder_overshoot -= turnmeoff["wattage"]
    else:
        remainder_undershoot = -overshoot
        while remainder_undershoot > 0 and not powerload_df.empty:
            turnmeon = (
                powerload_df.sort_values(["change_candidate", "on_need"])
                .head(1)
                .to_dict(orient="records")[0]
            )
            actions.append({"ON": turnmeon})
            powerload_df.drop(axis=0, index=turnmeon["index"], inplace=True)
            remainder_overshoot -= turnmeon["wattage"]
    return actions


async def turn(action: str, device: dict) -> None:
    """Perform an action on a specific power device.

    Will send an ON or OFF, or will adjust a setpoint, based on
    dictionary contents.

    Knows to do inverted ON/OFF for verksted/garasje
    garasje is 2700W
    verksted is 1600W
    """
    print(f" *** Turning {action} {device}")


async def estimate_currenthourusage(pers) -> int:
    lasthour = datetime.datetime.utcnow().replace(second=0, minute=0, microsecond=0)
    lastminute = datetime.datetime.utcnow() - datetime.timedelta(minutes=1)

    query = f"SELECT * FROM {CURRENT_POWER_ITEM} WHERE time > '{lasthour}'"

    lasthour_df = await pers.influxdb.dframe_query(query)

    # Use last minute for extrapolation:
    lastminute = await pers.influxdb.dframe_query(
        f"SELECT mean(*) FROM {CURRENT_POWER_ITEM} WHERE time > '{lastminute}'"
    )
    return _estimate_currenthourusage(lasthour_df["value"], lastminute.values[0][0])


def _estimate_currenthourusage(
    lasthour_series: pd.Series, lastminute_value: float
) -> int:
    if lasthour_series.empty:
        return round(lastminute_value)
    time_min = lasthour_series.index.min()
    time_max = time_min + datetime.timedelta(hours=1)
    lasthour_s = lasthour_series.resample("s").mean().fillna(method="ffill")
    remainder_hour = pd.Series(
        index=pd.date_range(
            start=lasthour_s.index[-1] + datetime.timedelta(seconds=1),
            end=time_max - datetime.timedelta(seconds=1),  # end at :59:59
            freq="s",
        ),
        data=lastminute_value,
    )
    full_hour = pd.concat([lasthour_s, remainder_hour], axis=0)
    return round(full_hour.mean())


async def main() -> None:
    pers = persist.PyrotunPersistence()
    await pers.ainit(["influxdb", "openhab"])
    est = await estimate_currenthourusage(pers)
    print(f"Estimated power usage for current hour is: {est} Wh")

    powerloads = await get_powerloads(pers)
    print(powerloads)
    await pers.aclose()


if __name__ == "__main__":
    asyncio.run(main())
