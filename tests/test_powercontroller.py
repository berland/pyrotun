import datetime
from typing import List
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from pyrotun import persist, powercontroller


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "year, month, expected_bkk",
    [
        (2022, 7, 8100),
        (2022, 6, 8010),
        (2022, 5, 10360),
        (2022, 4, 9550),
        (2022, 3, 8950),
        (2022, 2, 7970),
        (2022, 1, 10120),
        (2021, 12, 11520),
        pytest.param(
            2021,
            11,
            9400,
            marks=pytest.mark.xfail(reason="missing data, our estimate is 8674"),
        ),
        # (2021, 10, 9380),  # Local data hiatus!!!
        # (2021, 9, 7890),
        # (2021, 8, 6430),
        # (2021, 7, 5300),
        # (2021, 6, 7710),
    ],
)
async def test_top_three_hourmax_this_month_vs_bkk(
    year: int, month: int, expected_bkk: int
):
    """Test that what we can calculate from local data resembles
    what BKK claims our nettleietrinn will end at.

    It seems we sometimes overestimate our own data, that is ok, then
    we at least keep on the right side of the cost."""
    pers = persist.PyrotunPersistence()
    await pers.ainit("influxdb")
    hourmaxes: List[float] = await powercontroller.monthly_hourmaxes(pers, year, month)
    print(hourmaxes)
    top_watts = [watt for watt in hourmaxes if not np.isnan(watt)]
    print(top_watts)
    top_watts = sorted(top_watts)[-3:]
    power = sum(top_watts) / len(top_watts)
    print(f"{power=}")
    print(f"{expected_bkk=}")
    assert power < expected_bkk + 240  # We allow overestimate our own data!
    assert power >= expected_bkk - 10  # We almost never underestimate


def test_estimate_currenthourusage():
    now = datetime.datetime.now()
    hourstart = now.replace(minute=0, second=0, microsecond=0)
    testminutes = 3
    wattseries = pd.Series(
        index=pd.date_range(hourstart, periods=testminutes * 30, freq="2s"),
        name="AMSpower",
        data=[500 + x * 100 for x in list(range(testminutes)) * 30],
    )

    est = powercontroller._estimate_currenthourusage(wattseries, 200)
    assert est == 220  # rounded to watt


def test_estimate_currenthourusage_athourstart():
    # Series of length 1 (only valid for 1 sec):
    assert (
        powercontroller._estimate_currenthourusage(
            pd.Series(
                index=[
                    datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
                ],
                data=[3000],  # Using 3000W the first second
            ),
            1,  # Assume 1W the remainder of the hour
        )
        == 2  # Rounded up to 2W
    )

    # Empty series might happen:
    est = powercontroller._estimate_currenthourusage(pd.Series(), 200)
    assert est == 200


@pytest.mark.asyncio
async def test_estimate_currenthourusage_for_real():
    """Assert that the current/live usage for the house is estimatable
    and that it follows within a quite broad range."""
    pers = persist.PyrotunPersistence()
    await pers.ainit("influxdb")
    est = await powercontroller.estimate_currenthourusage(pers)
    assert 600 < est < 20000
    await pers.aclose()


@pytest.mark.parametrize(
    "overshoot, powerload_df, expected_actions",
    [
        pytest.param(0, pd.DataFrame(), [], id="zero"),
        pytest.param(1000, pd.DataFrame(), [], id="nothing_to_turn_off"),
        pytest.param(-1000, pd.DataFrame(), [], id="nothing_to_turn_on"),
        pytest.param(
            1000,
            {"switch_item": "foo_bryter", "wattage": 1000},
            [{"OFF": "foo_bryter"}],
            id="turn_off_single_option",
        ),
        pytest.param(
            1000,
            [
                {"switch_item": "foo_bryter", "wattage": 100},
                {"switch_item": "bar_bryter", "wattage": 100},
            ],
            [{"OFF": "bar_bryter"}, {"OFF": "foo_bryter"}],
            id="turn_off_multiple",
        ),
        pytest.param(
            1000,
            [
                {
                    "switch_item": "foo_bryter",
                    "wattage": 100,
                    "lastchange": 10,
                    "is_on": "YES",
                },
                {"switch_item": "bar_bryter", "wattage": 100},
            ],
            [{"OFF": "foo_bryter"}, {"OFF": "bar_bryter"}],
            id="turning_off_on_long_enough",
        ),
        pytest.param(
            100,
            [
                {
                    "switch_item": "foo_bryter",
                    "wattage": 100,
                    "lastchange": 10,
                    "is_on": "YES",
                },
                {"switch_item": "bar_bryter", "wattage": 100},
            ],
            [{"OFF": "foo_bryter"}],
            id="prioritize_when_last_changed_exists_and_larger_than_5",
        ),
        pytest.param(
            100,
            [
                {
                    "switch_item": "foo_bryter",
                    "wattage": 100,
                    "lastchange": 10,
                    "is_on": "NO",
                },
                {"switch_item": "bar_bryter", "wattage": 100, "is_on": "YES"},
            ],
            [{"OFF": "bar_bryter"}],
            id="skip_off_switches",
        ),
        pytest.param(
            100,
            [
                {
                    "switch_item": "foo_bryter",
                    "wattage": 100,
                    "lastchange": 4.9,
                    "is_on": "YES",
                },
                {
                    "switch_item": "bar_bryter",
                    "wattage": 100,
                    "lastchange": 5.1,
                    "is_on": "YES",
                },
            ],
            [{"OFF": "bar_bryter"}],
            id="lastchange_5_minutes_matter",
        ),
    ],
)
def test_decide(overshoot, powerload_df, expected_actions):
    if isinstance(powerload_df, dict):
        powerload_df = pd.DataFrame([powerload_df])
    if isinstance(powerload_df, list):
        powerload_df = pd.DataFrame(powerload_df)
    actions = powercontroller._decide(overshoot, powerload_df)
    # Slice out only switch_item, for testing:
    sliced_actions = [
        {list(act.keys())[0]: list(act.values())[0]["switch_item"]} for act in actions
    ]
    assert sliced_actions == expected_actions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action, deviceinfo, expected_item, expected_value",
    [
        (
            "OFF",
            {"setpoint_item": "termostat", "meas_temp": 20, "setpoint_force": 3},
            "termostat",
            17,
        ),
        (
            "OFF",
            {"setpoint_item": "termostat", "meas_temp": 15, "setpoint_force": 8},
            "termostat",
            11,  # lower limit for heat-it thermostats
        ),
        (
            "ON",
            {"setpoint_item": "termostat", "meas_temp": 13, "setpoint_force": 3},
            "termostat",
            16,
        ),
        (
            "ON",
            {
                "setpoint_item": "termostat",
                "meas_temp": 13,
                "setpoint_force": 3,
                "switch_item": "termostat_status",
            },
            "termostat",
            16,
        ),
        ("ON", {"switch_item": "bryter"}, "bryter", "ON"),
        ("OFF", {"switch_item": "bryter"}, "bryter", "OFF"),
        ("ON", {"switch_item": "bryter", "inverted_switch": True}, "bryter", "OFF"),
        ("OFF", {"switch_item": "bryter", "inverted_switch": True}, "bryter", "ON"),
    ],
)
async def test_turn(action, deviceinfo, expected_item, expected_value, mocker):
    pers = persist.PyrotunPersistence()
    pers.openhab = AsyncMock()
    pers.openhab.set_item = AsyncMock()
    await powercontroller.turn(pers, action, deviceinfo)
    pers.openhab.set_item.assert_awaited_once_with(
        expected_item, expected_value, log=True
    )


baseline = 10000
step = 5000
safeguard = 50


@pytest.mark.parametrize(
    "hourmaxes, expected",
    [
        ([10], baseline - safeguard),
        ([10000], baseline - safeguard),
        ([12000], 12000 - safeguard),
        ([12000, 5000], baseline - 2000 - safeguard),
        ([12000, 13000], 13000 - safeguard),
        ([9000, 13000], 13000 - safeguard),
        ([9000, 11000], 11000 - safeguard),
        ([2000, 3000], baseline - safeguard),
        ([8000, 10000, 12000], 12000 - safeguard),
        ([12000, 10000, 8000], 8000 - safeguard),
        ([12000, 10000, 5000], 8000 - safeguard),
        ([1, 7000, 10000, 12000], 12000 - safeguard),
        ([1, 12000, 10000, 7000], 8000 - safeguard),
        ([1, 15000, 10000, 7000], baseline + step - safeguard),  # Up one step
        ([1, 24000, 10000, 8000], 11000 - safeguard),  # Up one step
        pytest.param([1, 24000, 15000, 8000], 20000 - safeguard, id="up-two_steps"),
        pytest.param([np.nan], baseline - safeguard, id="only-one-nan-hour"),
        pytest.param(
            [9000, 10500, 10100], 10100 - safeguard, id="keep-allowing-todays-max"
        ),
        pytest.param([9000, 10500, 10100, np.nan], 9950, id="lasthour-is-nan"),
        pytest.param(
            [10000, 12000, 11000],
            baseline + step - safeguard,
            id="not_offensive_at_one_step_up",
            # this scenario allows for one daymax at 22000, but we opt not to go there
        ),
    ],
)
def test_currentlimit_from_hourmaxes(hourmaxes, expected):
    assert powercontroller.currentlimit_from_hourmaxes(hourmaxes) == expected
