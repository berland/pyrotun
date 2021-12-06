import datetime

import pandas as pd
import pytest

from pyrotun import persist, powercontroller



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
    est = powercontroller._estimate_currenthourusage(pd.Series(dtype=float), 200)
    assert est == 200


@pytest.mark.asyncio
async def test_estimate_currenthourusage_for_real():
    pers = persist.PyrotunPersistence()
    await pers.ainit("influxdb")
    est = await powercontroller.estimate_currenthourusage(pers)
    print(est)
    assert 600 < est < 10000
    await pers.aclose()


@pytest.mark.parametrize(
    "overshoot, powerload_df, expected_actions",
    [
        (0, pd.DataFrame(), []),
        (1000, pd.DataFrame(), []),
        (-1000, pd.DataFrame(), []),
        (1000, {"switch_item": "foo_bryter", "wattage": 1000}, [{"OFF": "foo_bryter"}]),
        (
            1000,
            [
                {"switch_item": "foo_bryter", "wattage": 100},
                {"switch_item": "bar_bryter", "wattage": 100},
            ],
            [{"OFF": "bar_bryter"}, {"OFF": "foo_bryter"}],
        ),
        (
            # Prioritize assumedly ON appliances:
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
        ),
        (
            # Stop when overshoot is solved:
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
        ),
        (
            # Skip assumedly OFF
            100,
            [
                {
                    "switch_item": "foo_bryter",
                    "wattage": 100,
                    "lastchange": 10,
                    "is_on": "NO",
                },
                {"switch_item": "bar_bryter", "wattage": 100},
            ],
            [{"OFF": "bar_bryter"}],
        ),
        (
            # If needed, also message assumedly OFF devices
            500,
            [
                {
                    "switch_item": "foo_bryter",
                    "wattage": 100,
                    "lastchange": 10,
                    "is_on": "NO",
                },
                {"switch_item": "bar_bryter", "wattage": 100},
            ],
            [{"OFF": "bar_bryter"}, {"OFF": "foo_bryter"}],
        ),
        (
            # Prioritize on on_need, turn off less important first.
            500,
            [
                {"switch_item": "foo_bryter", "wattage": 100, "on_need": 5},
                {"switch_item": "bar_bryter", "wattage": 100, "on_need": 20},
            ],
            [{"OFF": "foo_bryter"}, {"OFF": "bar_bryter"}],
        ),
        (
            # Prioritize on on_need, reverse on undershoot, and skip third:
            -200,
            [
                {"switch_item": "foo_bryter", "wattage": 100, "on_need": 5},
                {"switch_item": "bar_bryter", "wattage": 100, "on_need": 20},
                {"switch_item": "com_bryter", "wattage": 100, "on_need": 2},
            ],
            [{"ON": "bar_bryter"}, {"ON": "foo_bryter"}],
        ),
        (
            # Prioritize on lastchange over on_need
            500,
            [
                {
                    "switch_item": "foo_bryter",
                    "wattage": 100,
                    "on_need": 5,
                    "lastchange": 4,
                },
                {
                    "switch_item": "bar_bryter",
                    "wattage": 100,
                    "on_need": 20,
                    "lastchange": 6,
                },
            ],
            [{"OFF": "bar_bryter"}, {"OFF": "foo_bryter"}],
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
