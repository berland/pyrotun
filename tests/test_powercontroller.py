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
