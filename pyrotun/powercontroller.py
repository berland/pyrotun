"""Module for controlling/truncating house power usage

To be run()'ed every minute.

"""
import asyncio
import datetime
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import numpy as np
import pandas as pd
import pytz
import yaml

import pyrotun
from pyrotun import persist

logger = pyrotun.getLogger(__name__)

CURRENT_POWER_ITEM = "AMSpower"
CUMULATIVE_WH_ITEM = "AMS_cumulative_Wh"
HOURUSAGE_ESTIMATE_ITEM = "WattHourEstimate"
POWER_FOR_EFFEKTTRINN = "NettleieWatt"
CUMULATIVE_WH_ITEM = "AMS_cumulative_Wh"
FLOORSFILE = Path(__file__).absolute().parent / "floors.yml"

TZ = pytz.timezone(os.getenv("TIMEZONE"))  # type: ignore

PERS = None


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
                12
                - await pers.openhab.get_item(
                    "Sensor_Verkstedgulv_temperatur", datatype=float
                ),
                0,
            ),
            "sensor_item": "Sensor_Verkstedgulv_temperatur",
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
                    "Sensor_Garasjegulv_temperatur", datatype=float
                ),
                0,
            ),
            "sensor_item": "Sensor_Garasjegulv_temperatur",
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

    # Varmepumpe? Don't touch it...!

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
            thisfloor.update({"meas_temp": meas_temp})
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

    powerload_df = await get_powerloads(pers)

    actions = _decide(overshoot, powerload_df)
    for action, appliance in actions.items():
        await turn(pers, action, appliance)


def _decide(overshoot: int, powerload_df: pd.DataFrame) -> List[Dict[str, dict]]:
    """

    Args:
        overshoot: How much is the current wattage compared
            to what we want to consume right now. Positive
            number means we are using too much (and need to turn
            something off)
        powerload_df: Dataframe with current loads, one row
            pr consumer.
    """
    assert isinstance(overshoot, int)
    assert isinstance(powerload_df, pd.DataFrame)
    powerload_df = pd.DataFrame(powerload_df)
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
            print(powerload_df)
            turnmeoff = (
                powerload_df[powerload_df["is_on"] != "OFF"]
                .sort_values(["change_candidate", "on_need"])
                .tail(1)
                .dropna("columns")
                .to_dict(orient="records")[0]
            )
            print(turnmeoff)
            actions.append({"OFF": turnmeoff})
            powerload_df.drop(axis=0, index=turnmeoff["index"], inplace=True)
            remainder_overshoot -= turnmeoff["wattage"]
    else:
        remainder_undershoot = -overshoot
        while remainder_undershoot > 0 and not powerload_df.empty:
            turnmeon = (
                powerload_df[powerload_df["is_on"] != "ON"]
                .sort_values(["change_candidate", "on_need"])
                .head(1)
                .dropna("columns")
                .to_dict(orient="records")[0]
            )
            actions.append({"ON": turnmeon})
            powerload_df.drop(axis=0, index=turnmeon["index"], inplace=True)
            remainder_overshoot -= turnmeon["wattage"]
    return actions


async def turn(pers, action: str, device: Dict[str, Any]) -> None:
    """Perform an action on a specific power device.

    Will send an ON or OFF, or will adjust a setpoint, based on
    dictionary contents.

    Knows to do inverted ON/OFF for verksted/garasje
    garasje is 2700W
    verksted is 1600W
    """
    assert action in {"ON", "OFF"}

    logger.info(f" *** Turning {action} {device}")
    if "switch_item" in device and device["switch_item"] is not np.nan:
        # Simple switch item to flip:
        if device.get("inverted_switch", False):
            action = {"ON": "OFF", "OFF": "ON"}[action]
        await pers.openhab.set_item(device["switch_item"], action, log=True)
    elif "setpoint_item" in device and device["setpoint_item"] is not np.nan:
        if action == "ON":
            if isinstance(device["setpoint_item"], str):
                device["setpoint_item"] = [device["setpoint_item"]]
            for item in device["setpoint_item"]:
                await pers.openhab.set_item(
                    item,
                    math.ceil(device["meas_temp"] + device["setpoint_force"]),
                    log=True,
                )
        if action == "OFF":
            if isinstance(device["setpoint_item"], str):
                device["setpoint_item"] = [device["setpoint_item"]]
            for item in device["setpoint_item"]:
                await pers.openhab.set_item(
                    item,
                    math.floor(device["meas_temp"] - device["setpoint_force"]),
                    log=True,
                )


async def estimate_currenthourusage(pers) -> int:
    """Estimates what the hour usage in Wh will be for the current hour (at end
    of the hour)"""
    lasthour: datetime.datetime = datetime.datetime.utcnow().replace(
        second=0, minute=0, microsecond=0
    )
    lastminute: datetime.datetime = datetime.datetime.utcnow() - datetime.timedelta(
        minutes=1
    )

    query = f"SELECT * FROM {CURRENT_POWER_ITEM} WHERE time > '{lasthour}'"

    lasthour_df = await pers.influxdb.dframe_query(query)

    # Use last minute for extrapolation:
    lastminutes: pd.DataFrame = await pers.influxdb.dframe_query(
        f"SELECT mean(*) FROM {CURRENT_POWER_ITEM} WHERE time > '{lastminute}'"
    )
    if isinstance(lastminutes, dict) or lastminutes.empty:
        # If last minute fails, maybe intermittently missing new data:
        lastminutes = pd.DataFrame([2000])

    return _estimate_currenthourusage(lasthour_df["value"], lastminutes.values[0][0])


def _estimate_currenthourusage(
    lasthour_series: pd.Series, lastminute_value: float
) -> int:
    """This function is factored out from its parent function to facilitate
    testing"""
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


async def update_effekttrinn(pers):
    hourmaxes: List[int] = await monthly_hourmaxes(pers)
    top_watts = sorted(hourmaxes)[-3:]
    mean = sum(top_watts) / len(top_watts)
    await pers.openhab.set_item(POWER_FOR_EFFEKTTRINN, str(int(mean)), log=True)


def currentlimit_from_hourmaxes(hourmaxes_pr_day: List[float]):
    """Given the three highest daily maxes, determine what we currently
    allow as a hourly power consumption average.

    Special casing for the three first days of a month.

      Gjennomsnittet av de tre timene med høyest forbruk, på tre ulike dager i
      forrige måned, vil avgjøre hva slags trinn du havner i.
    """

    baseline = 10000  # Car charger makes it meaningless to try to be below 10k
    step = 5000
    safeguard = 50

    todays_hourmax = hourmaxes_pr_day[-1]
    dayofmonth = len(hourmaxes_pr_day)
    top_watts = sorted(hourmaxes_pr_day)[-3:]
    two_largest = top_watts[-2:]

    # Most of the days during the month, then we just accept the target
    # coming from the mean consumption:
    if dayofmonth > 3:
        print(sum(top_watts))
        while sum(top_watts) > 3 * baseline:
            baseline = baseline + step

        return max(
            todays_hourmax - safeguard,
            min(baseline - safeguard, 3 * baseline - sum(two_largest) - safeguard),
        )

    # Special casing the first three days, try to make mean below baseline:
    if dayofmonth == 3:
        # baseline is always 10000
        return max(
            todays_hourmax - safeguard,
            min(baseline - safeguard, 3 * baseline - sum(two_largest) - safeguard),
        )
    if dayofmonth == 2:
        return max(
            todays_hourmax - safeguard,
            min(baseline - safeguard, 2 * baseline - hourmaxes_pr_day[0] - safeguard),
        )
    if dayofmonth == 1:
        return max(todays_hourmax - safeguard, baseline - safeguard)


async def monthly_hourmaxes(
    pers, year: Optional[int] = None, month: Optional[int] = None
) -> List[float]:
    """
    Make a list pr. day of max hourmaxes in power consumption.

    In Influx, this data is available at around 13 seconds after every hour.

    """

    if year is None and month is None:
        # Make monthstart in UTC, but as timezone unaware object, this is what
        # influxdb needs:
        monthstart = (
            datetime.datetime.now()
            .replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            .astimezone(TZ)
            .astimezone(pytz.timezone("UTC"))
            .replace(tzinfo=None)
        )
        monthend = (
            datetime.datetime.now()
        )  # gir problem når denne akkurat bikker et nytt døgn,
        # should we just do timezone convert??
    else:
        assert year is not None
        assert month is not None
        monthstart = datetime.datetime(year, month, 1, 0, 0)
        if month < 12:
            nextmonth = month + 1
            nextyear = year
        else:
            nextmonth = 1
            nextyear = year + 1
        monthend = datetime.datetime(nextyear, nextmonth, 1, 0, 0) - datetime.timedelta(
            seconds=1
        )
    cumulative_hour_usage_thismonth: pd.Series = await pers.influxdb.get_series(
        CUMULATIVE_WH_ITEM, since=monthstart, upuntil=monthend
    )
    # Get local timezone again:
    cumulative_hour_usage_thismonth.index = (
        cumulative_hour_usage_thismonth.index.tz_convert(TZ)
    )
    if cumulative_hour_usage_thismonth.empty:
        return 0
    hourly_usage: pd.Series = (
        cumulative_hour_usage_thismonth.resample("1h").mean().diff()
    )[CUMULATIVE_WH_ITEM]

    # Get local timezone again:
    # hourly_usage.index = hourly_usage.index.tz_convert(TZ)
    # Shift so that watt usage is valid forwards in time:
    hourly_usage = hourly_usage.shift(-1)

    daily_maximum = hourly_usage.resample("1d").max()
    three_largest_daily_hourmax = daily_maximum.sort_values().tail(3)
    logger.info(f"Max daily hourmax'es: {three_largest_daily_hourmax.values} W")

    return list(daily_maximum.values)


async def main(pers=None) -> None:
    close_pers = False
    if pers is None:
        pers = persist.PyrotunPersistence()
        await pers.ainit(["influxdb", "openhab"])
        close_pers = True
    hourmaxes: List[int] = await monthly_hourmaxes(pers)
    upperlimit_watt = currentlimit_from_hourmaxes(hourmaxes)
    logger.info(f"Vi må holde oss under: {upperlimit_watt} W denne måneden")
    est = await estimate_currenthourusage(pers)
    logger.info(f"Estimated power usage for current hour is: {est} Wh")

    powerload_df = await get_powerloads(pers)

    if est > upperlimit_watt:
        logger.warning("Using too much power this hour, must turn off appliances")
        actions = _decide(est - upperlimit_watt, powerload_df)
        print(actions)
        for action_dict in actions:
            action = list(action_dict.keys())[0]
            await turn(pers, action, action_dict[action])  # Ugly data structure

    if close_pers:
        await pers.aclose()


if __name__ == "__main__":
    asyncio.run(main())
