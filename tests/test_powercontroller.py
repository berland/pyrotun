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


@pytest.mark.asyncio
@pytest.mark.parametrize("hoursago", range(1, 100))
async def test_amspower_summation(hoursago: int):
    """Test summation (with infill) of AMSPower data up to the reported cumulatives.

    This proves that 2-second data from meter can be used to estimate within given hour."""
    pers = persist.PyrotunPersistence()
    await pers.ainit("influxdb")
    assert hoursago > 0
    start = datetime.datetime.utcnow().replace(
        second=0, minute=0, microsecond=0
    ) - datetime.timedelta(hours=hoursago)
    end = datetime.datetime.utcnow().replace(
        second=0, minute=0, microsecond=0
    ) - datetime.timedelta(hours=hoursago - 1)

    CURRENT_POWER_ITEM = "AMSpower"
    BACKUP_POWER_ITEM = "Smappee_avgW_5min"
    HOUR_ITEM = "AMS_cumulative_lasthour_KWh"
    HOURUSAGE_ESTIMATE_ITEM = "EstimatedKWh_thishour"
    query = (
        f"SELECT * FROM {CURRENT_POWER_ITEM} WHERE time > '{start}' AND time < '{end}'"
    )

    lasthour_df = await pers.influxdb.dframe_query(query)
    coverage = len(lasthour_df) / (60 * 60 / 2)
    print(f"coverage: {coverage}")
    fromseconddata = powercontroller._estimate_currenthourusage(lasthour_df["value"])
    print(f"from second-datas: {fromseconddata}")

    # Get the latest estimates in that hour:
    query = f"SELECT * FROM {HOURUSAGE_ESTIMATE_ITEM} WHERE time > '{end - datetime.timedelta(minutes=10)}' AND time < '{end}'"
    query_df = await pers.influxdb.dframe_query(query)
    estimates = query_df["value"].values
    # We should not have high variance in the end
    if estimates.std() > 50:
        print(f"high deviation, estimates={estimates}")
    estimate = float(estimates[-1])
    print(f"last estimate: {estimate}")
    # Get the correct hour-cumulative (inferred from the meter itself)
    query = f"SELECT * FROM {HOUR_ITEM} WHERE time > '{end}' AND time < '{end + datetime.timedelta(minutes=2)}'"
    query_df = await pers.influxdb.dframe_query(query)
    if isinstance(query_df, pd.DataFrame):
        fromcumulative = query_df["value"].values[0]
        if fromcumulative > 0:
            print(f"from cumulative data: {fromcumulative*1000}")
            miss = fromcumulative * 1000 - fromseconddata
            print(f"MISS: {miss}")
            estimatemiss = abs(estimate - fromcumulative * 1000)
            if coverage > 0.7:
                assert abs(miss) < 200
                assert estimatemiss < 100
    else:
        print("MISSING CUMULATIVE DATA")
