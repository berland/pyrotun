import datetime

import pandas as pd
import pytest

from pyrotun import persist, powerplanner


@pytest.mark.parametrize(
    "temp, expected_wattage",
    [
        (-25, 11500),  # extrapolation
        (-15, 6500),
        (0, 4000),  # interpolation
        (5, 3000),
        (10, 2500),  # actual interpolation
        (30, 2000),  # extrapolation
    ],
)
def test_avg_wattage_from_temperature(temp, expected_wattage):
    assert powerplanner.avg_wattage_from_temperature(temp) == expected_wattage


@pytest.mark.parametrize(
    "timestamp, expected_temp, kwargs",
    [
        (datetime.datetime(2021, 11, 28, 13, 0, 0), 25, {}),  # sunday
        (datetime.datetime(2021, 11, 29, 13, 0, 0), 20, {}),  # monday
        (datetime.datetime(2021, 11, 29, 18, 0, 0), 25, {}),  # monday
        (datetime.datetime(2021, 11, 29, 18, 0, 0), 26, {"comfort_temp": 26}),  # monday
        (datetime.datetime(2021, 11, 29, 18, 0, 0), 18, {"vacation": True}),  # monday
        (
            datetime.datetime(2021, 11, 29, 13, 0, 0),
            25,
            {"lowered_daytime": False},
        ),  # A monday
    ],
)
def test_master_heat_temp(timestamp, expected_temp, kwargs):
    assert (
        powerplanner._master_heat_temperature(timestamp=timestamp, **kwargs)
        == expected_temp
    )
