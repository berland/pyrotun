"""Module for controlling/truncating house power usage

To be run()'ed every minute.

"""
import argparse
import asyncio
import datetime
from pathlib import Path
from typing import Dict, List

import aiofiles
import numpy as np
import pandas as pd
import yaml

import pyrotun
from pyrotun import persist

logger = pyrotun.getLogger(__name__)

CURRENT_POWER_ITEM = "AMSpower"
BACKUP_POWER_ITEM = "Smappee_avgW_5min"
HOURUSAGE_ESTIMATE_ITEM = "EstimatedKWh_thishour"
MAXHOURWATT_LASTMONTH_ITEM = "MaxHourwatt_lastmonth"
FLOORSFILE = Path(__file__).absolute().parent / "floors.yml"

ONOFF2YESNO = {"ON": "YES", "OFF": "NO"}
INV_ONOFF2YESNO = {"ON": "NO", "OFF": "YES"}


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
            "is_on": INV_ONOFF2YESNO[
                await pers.openhab.get_item("Verksted_varmekabler_nattsenking")
            ],
            "on_need": 12
            - await pers.openhab.get_item("VerkstedTemperatur", datatype=float),
        }
    )
    loads.append(
        {
            "switch_item": "Garasje_varmekabler_nattsenking",
            "inverted_switch": True,
            "wattage": 2400,
            "lastchange": await pers.influxdb.item_age(
                "Garasje_varmekabler_nattsenking", unit="minutes"
            ),
            "is_on": INV_ONOFF2YESNO[
                await pers.openhab.get_item("Garasje_varmekabler_nattsenking")
            ],
            "on_need": 6
            - await pers.openhab.get_item("Sensor_Garasje_temperatur", datatype=float),
        }
    )

    loads.append(
        {
            "switch_item": "Varmtvannsbereder_bryter",
            "wattage": 2800,
            "is_on": ONOFF2YESNO[
                await pers.openhab.get_item("Varmtvannsbereder_bryter")
            ],
            "lastchange": await pers.influxdb.item_age(
                "Varmtvannsbereder_bryter", unit="minutes"
            ),
            "on_need": await pers.openhab.get_item(
                "Varmtvannsbereder_temperaturtarget", datatype=float
            )
            - await pers.openhab.get_item(
                "Varmtvannsbereder_temperatur", datatype=float
            ),
            "measured": await pers.openhab.get_item(
                "Varmtvannsbereder_temperatur", datatype=float
            ),
        }
    )

    async with aiofiles.open(FLOORSFILE, "r", encoding="utf-8") as filehandle:
        contents = await filehandle.read()
    floors = yaml.safe_load(contents)

    master_temperature = await pers.openhab.get_item("Master_termostat", datatype=float)

    for floor in floors:
        thisfloor = floors[floor].copy()  # needed?
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
            thisfloor.update({"measured": meas_temp})
            target_temp = master_temperature + thisfloor.get("delta", 0)
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
                        "is_on": ONOFF2YESNO[await pers.openhab.get_item(bryter_name)],
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

    estimated_wh = await pers.openhab.get_item("EstimatedKWh_thishour", datatype=float)
    overshoot = int(estimated_wh - powerplan)
    logger.info("Current over/under-shoot is: %d", overshoot)

    powerload_df = await get_powerloads(pers)

    logger.info("Built dataframe of powerloads:")
    print(powerload_df)

    actions = _decide(overshoot, powerload_df)
    logger.info("I have decided on power actions:")
    print(yaml.dump(actions))
    for action in actions:
        act = list(action.keys())[0]  # ON or OFF
        turn(act, action[act])


def _decide(overshoot: int, powerload_df: pd.DataFrame):
    assert isinstance(overshoot, int)
    assert isinstance(powerload_df, pd.DataFrame)
    actions: List[Dict[str, dict]] = []
    if powerload_df.empty:
        return actions
    powerload_df = powerload_df.copy()
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

    if overshoot > 0:
        remainder_overshoot = overshoot

        # Prioritize non-recently-changed and ON appliances:
        powerload_df.loc[
            (powerload_df["lastchange"] > 5) & (powerload_df["is_on"] != "NO"),
            "change_candidate",
        ] = 1

        while remainder_overshoot > 0 and not powerload_df.empty:
            turnmeoff = (
                powerload_df.sort_values(
                    ["change_candidate", "on_need"], ascending=[True, False]
                )
                .tail(1)
                .dropna(axis="columns")
                .to_dict(orient="records")[0]
            )
            actions.append({"OFF": turnmeoff})
            powerload_df.drop(axis=0, index=turnmeoff["index"], inplace=True)
            remainder_overshoot -= turnmeoff["wattage"]
    else:
        remainder_undershoot = -overshoot

        # Prioritize non-recently-changed and OFF appliances:
        powerload_df.loc[
            (powerload_df["lastchange"] > 5) & (powerload_df["is_on"] != "YES"),
            "change_candidate",
        ] = -1

        while remainder_undershoot > 0 and not powerload_df.empty:
            turnmeon = (
                powerload_df.sort_values(
                    ["change_candidate", "on_need"], ascending=[True, False]
                )
                .head(1)
                .dropna(axis="columns")
                .to_dict(orient="records")[0]
            )
            actions.append({"ON": turnmeon})
            powerload_df.drop(axis=0, index=turnmeon["index"], inplace=True)
            remainder_undershoot -= turnmeon["wattage"]

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

    # Smapeee-dataene er egentlig 10 minutter forsinket :/
    backup_query = f"SELECT * FROM {BACKUP_POWER_ITEM} WHERE time > '{lasthour}'"

    lasthour_df = await pers.influxdb.dframe_query(query)
    lasthour_backup_df = await pers.influxdb.dframe_query(backup_query)

    # Merge and sort the two frames. Data from the backup is pr. 5 min, and will
    # drown if the 2-sec data is dense. The Smappee 10-min delay will affect though!

    merged_df = pd.concat(
        [lasthour_df, lasthour_backup_df], axis="index", sort=False
    ).sort_index()

    # Use last minute for extrapolation:
    lastminute = await pers.influxdb.dframe_query(
        f"SELECT mean(*) FROM {CURRENT_POWER_ITEM} WHERE time > '{lastminute}'"
    )
    if lastminute.empty:
        return _estimate_currenthourusage(merged_df["value"], None)
    return _estimate_currenthourusage(merged_df["value"], lastminute.values[0][0])


def _estimate_currenthourusage(
    lasthour_series: pd.Series, extrapolation_value: float = None
) -> int:
    assert isinstance(lasthour_series, pd.Series)
    if lasthour_series.empty:
        return round(extrapolation_value)
    time_min = lasthour_series.index.min()
    time_max = (time_min + datetime.timedelta(hours=1)).replace(second=0, minute=0, microsecond=0)
    lasthour_s = lasthour_series.resample("s").mean().fillna(method="ffill")


    if lasthour_s.index[-1] + datetime.timedelta(seconds=1) < time_max:
        if extrapolation_value is None:
            extrapolation_value = lasthour_s.tail(1).values[0]
        # Extrapolate through the rest of the hour:
        remainder_hour = pd.Series(
            index=pd.date_range(
                start=lasthour_s.index[-1] + datetime.timedelta(seconds=1),
                end=time_max - datetime.timedelta(seconds=1),  # end at :59:59
                freq="s",
            ),
            data=extrapolation_value,
        )
        full_hour = pd.concat([lasthour_s, remainder_hour], axis="index", sort=False)
    else:
        full_hour = pd.concat([lasthour_s], axis="index", sort=False)
    return round(full_hour.mean())


async def main(maketurns: bool = False) -> None:
    pers = persist.PyrotunPersistence()
    await pers.ainit(["influxdb", "openhab"])
    est = await estimate_currenthourusage(pers)
    print(f"Estimated power usage for current hour is: {est} Wh")

    powerloads = await get_powerloads(pers)
    print(powerloads)

    print("If overshoot by 1000, we would turn off:")
    print(_decide(1000, powerloads))
    print("If undershoot by 1000, we would turn on:")
    print(_decide(-1000, powerloads))

    if maketurns:
        await control_powerusage(pers)

    await pers.aclose()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maketurns", help="If true, will send actions to openhab", action="store_true"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    asyncio.run(main(maketurns=args.maketurns))
