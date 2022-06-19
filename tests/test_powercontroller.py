import datetime

import pandas as pd
import pytest

from pyrotun import persist, powercontroller


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "year, month, expected_bkk",
    [
        (2022, 5, 10360),
        (2022, 4, 9550),
        (2022, 3, 8950),
        (2022, 2, 7970),
        (2022, 1, 10120),
        (2021, 12, 11520),
        (2021, 11, 9400),
        # (2021, 10, 9380),  # Local data hiatus!!!
        # (2021, 9, 7890),
        # (2021, 8, 6430),
        # (2021, 7, 5300),
        # (2021, 6, 7710),
    ],
)
async def test_nettleie_maanedseffekt_vs_bkk(year: int, month: int, expected_bkk: int):
    """Test that what we can calculate from local data resembles
    what BKK claims our nettleietrinn will end at.

    It seems we sometimes overestimate our own data, that is ok, then
    we at least keep on the right side of the cost."""
    pers = persist.PyrotunPersistence()
    await pers.ainit("influxdb")
    power = await powercontroller.nettleie_maanedseffekt(pers, year, month)
    assert power < expected_bkk + 240  # We allow overestimate our own data!
    assert power >= expected_bkk  # We never underestimate


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
        (1000, {"switch_item": "foo_bryter", "wattage": 1000}, [{"ON": "foo_bryter"}]),
        (
            1000,
            [
                {"switch_item": "foo_bryter", "wattage": 100},
                {"switch_item": "bar_bryter", "wattage": 100},
            ],
            [{"ON": "bar_bryter"}, {"ON": "foo_bryter"}],
        ),
        (
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
            [{"ON": "foo_bryter"}, {"ON": "bar_bryter"}],
        ),
        (
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
            [{"ON": "foo_bryter"}],
        ),
        (
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
            [{"ON": "bar_bryter"}],
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
