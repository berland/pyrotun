import datetime

import numpy as np
import pandas as pd
import pytest
import pytz

from pyrotun import heatreservoir


def test_prediction_dframe():
    prices = pd.DataFrame(
        index=[
            datetime.datetime(2020, 1, 1, 0, 0, 0),
            datetime.datetime(2020, 1, 1, 1, 0, 0),
            datetime.datetime(2020, 1, 1, 2, 0, 0),
        ],
        data=[0.2, 0.3, 0.25],
    )

    dframe = heatreservoir.prediction_dframe(
        datetime.datetime(2020, 1, 1, 0, 0, 0), prices=prices, freq="60min"
    )

    assert dframe.index[0] == pd.Timestamp("2020-01-01 00:00:00")
    # Always one hour after the last price information, since prices are valid one
    # hour into the future:
    assert dframe.index[-1] == pd.Timestamp("2020-01-01 03:00:00")
    assert len(dframe) == 4
    # Extrapolated last price, but price at this tstamp is probably not used by anything
    assert dframe["NOK/KWh"].values[-2] == 0.25
    assert dframe["NOK/KWh"].values[-1] == 0.25

    # starttime should be allowed to extend into the first hour, and get rounded down
    # in time.
    dframe = heatreservoir.prediction_dframe(
        datetime.datetime(2020, 1, 1, 0, 0, 1), prices=prices, freq="60min"
    )
    assert len(dframe) == 4

    assert (
        len(
            heatreservoir.prediction_dframe(
                datetime.datetime(2020, 1, 1, 0, 59, 0), prices=prices, freq="60min"
            )
        )
        == 4
    )

    assert (
        len(
            heatreservoir.prediction_dframe(
                datetime.datetime(2020, 1, 1, 0, 59, 59), prices=prices, freq="60min"
            )
        )
        == 4
    )

    assert (
        len(
            heatreservoir.prediction_dframe(
                datetime.datetime(2020, 1, 1, 1, 0, 0), prices=prices, freq="60min"
            )
        )
        == 3
    )

    # starttime before any prices:
    extrapolated = heatreservoir.prediction_dframe(
        datetime.datetime(2019, 12, 31, 22, 0, 0), prices=prices, freq="60min"
    )
    assert len(extrapolated) == 6
    assert (extrapolated["NOK/KWh"].values[0:3] == [0.2, 0.2, 0.2]).all()

    assert (
        len(
            heatreservoir.prediction_dframe(
                datetime.datetime(2020, 1, 1, 0, 0, 0), prices=prices, freq="10min"
            )
        )
        == 19
    )
    assert (
        len(
            heatreservoir.prediction_dframe(
                datetime.datetime(2020, 1, 1, 0, 9, 59), prices=prices, freq="10min"
            )
        )
        == 19
    )
    assert (
        len(
            heatreservoir.prediction_dframe(
                datetime.datetime(2020, 1, 1, 0, 10, 0), prices=prices, freq="10min"
            )
        )
        == 18
    )


@pytest.mark.parametrize(
    "price1, price2, starttemp, freq, on_now",
    [
        (0.3, 0.4, 19.5, "10min", False),
        (0.3, 0.4, 19.1, "10min", False),
        (0.3, 0.4, 19.02, "10min", False),
        (0.3, 0.4, 19.01, "10min", True),
        (0.3, 0.4, 19.0, "10min", True),
        #
        (0.3, 0.4, 19.5, "1min", False),
        (0.3, 0.4, 19.1, "1min", False),
        (0.3, 0.4, 19.01, "1min", False),
        (0.3, 0.4, 19.0, "1min", True),
        #
        (0.3, 0.4, 19.5, "60min", False),
        (0.3, 0.4, 19.1, "60min", True),
        (0.3, 0.4, 19.01, "60min", True),
        (0.3, 0.4, 19.0, "60min", True),
        #
        (0.5, 0.4, 19.5, "60min", False),
        (0.5, 0.4, 19.5, "60min", False),
        (0.5, 0.4, 19.5, "60min", False),
        (0.5, 0.4, 19.5, "60min", False),
        #
        (0.5, 0.4, 19.5, "10min", False),
        (0.5, 0.4, 19.5, "10min", False),
        (0.5, 0.4, 19.5, "10min", False),
        (0.5, 0.4, 19.5, "10min", False),
    ],
)
def test_two_hour_tspan(price1, price2, starttemp, freq, on_now):
    index = [
        datetime.datetime(2020, 1, 1, 0, 0, 0),
        datetime.datetime(2020, 1, 1, 1, 0, 0),
    ]
    prices = pd.Series(index=index, data=[price1, price2])
    min_temp = pd.Series(index=index, data=[19, 19])
    max_temp = pd.Series(index=index, data=[27, 27])

    def temp_predictor(temp, tstamp, t_delta_hours):
        return [
            {"temp": temp + 1 * t_delta_hours, "kwh": 0.5 * t_delta_hours},
            {"temp": temp - 0.1 * t_delta_hours, "kwh": 0},
        ]

    result = heatreservoir.optimize(
        starttime=index[0],
        starttemp=starttemp,
        prices=prices,
        min_temp=min_temp,
        max_temp=max_temp,
        temp_predictor=temp_predictor,
        freq=freq,
    )
    assert result["on_now"] == on_now


def test_cost_from_predictor():
    def temp_predictor(temp, tstamp, t_delta_hours):
        return [
            {"temp": temp + 1 * t_delta_hours, "kwh": 0.5 * t_delta_hours},
            {"temp": temp - 0.1 * t_delta_hours, "kwh": 0},
        ]

    assert heatreservoir.cost_from_predictor(temp_predictor, 20, 19.9, 1) == 0.0
    assert np.isclose(
        heatreservoir.cost_from_predictor(temp_predictor, 20, 20.1, 1), 1 / 11
    )
    assert heatreservoir.cost_from_predictor(temp_predictor, 20, 21, 1) == 0.5


@pytest.mark.parametrize(
    "price_low, price_high, starttemp, coolingrate, freq, on_in_minutes",
    [
        (0.3, 0.7, 40, 0.1, "60min", None),  # No need to turn on.
        (0.3, 0.7, 27, 0.1, "60min", None),  # No need to turn on.
        (0.3, 0.7, 26, 0.1, "60min", 10 * 60),
        (0.3, 0.7, 25, 0.1, "60min", 6 * 60),
        (0.3, 0.7, 24, 0.1, "60min", 5 * 60),
        (0.3, 0.7, 23, 0.1, "60min", 4 * 60),
        (0.3, 0.7, 22, 0.1, "60min", 3 * 60),
        (0.3, 0.7, 21, 0.1, "60min", 2 * 60),
        (0.3, 0.7, 20, 0.1, "60min", 1 * 60),
        (0.3, 0.7, 19, 0.1, "60min", 0),
        # Empty graph and unattainable target, should turn on:
        (0.3, 0.7, 10, 0.1, "60min", 0),
        ##
        (0.3, 0.7, 25, 0.0, "60min", None),  # No cooling..
        (0.3, 0.7, 25, 1, "60min", 2 * 60),
        (0.3, 0.7, 25, 2, "60min", 2 * 60),  # price effect
        (0.3, 0.3, 25, 2, "60min", 2 * 60),
        # Overshoots at 07:00, due to discretization:
        (0.3, 0.3, 25, 1, "60min", 3 * 60),
        # Touches 25 at 07:00 exactly.
        (0.3, 0.3, 24, 1, "60min", 3 * 60),
        # Finer time resolution allows touching 25 at 07:00:
        (0.3, 0.3, 25, 1, "30min", 3.5 * 60),
    ],
)
def test_night_day(price_low, price_high, starttemp, coolingrate, freq, on_in_minutes):

    index = [
        datetime.datetime(2020, 1, 1, 0, 0, 0),
        datetime.datetime(2020, 1, 1, 7, 0, 0),
        datetime.datetime(2020, 1, 1, 10, 0, 0),
    ]
    prices = pd.Series(index=index, data=[price_low, price_high, price_low])
    min_temp = pd.Series(index=index, data=[15, 25, 25])
    max_temp = pd.Series(index=index, data=[28, 28, 28])

    def temp_predictor(temp, tstamp, t_delta_hours):
        return [
            {"temp": temp + 1 * t_delta_hours, "kwh": 0.5 * t_delta_hours},
            {"temp": temp - coolingrate * t_delta_hours, "kwh": 0},
        ]

    result = heatreservoir.optimize(
        starttime=index[0],
        starttemp=starttemp,
        prices=prices,
        min_temp=min_temp,
        max_temp=max_temp,
        temp_predictor=temp_predictor,
        freq=freq,
    )
    print("Results:")
    print("------------------------------------")
    print(result)
    heatreservoir.plot_graph(result["graph"], path=result["path"], show=True)
    assert result["on_in_minutes"] == on_in_minutes


@pytest.mark.parametrize(
    "minutes",
    [(-15), (-11), (-10), (-9), (-4), (-1), (0), (1), (4), (9), (10), (11), (15)],
)
def test_starttime(minutes):
    """Test that we allow random start-times, +/- 1.5 * frequency should not alter
    the optimum point at which to turn on"""
    index = [
        datetime.datetime(2020, 1, 1, 0, 0, 0),
        datetime.datetime(2020, 1, 1, 5, 0, 0),
        datetime.datetime(2020, 1, 1, 9, 0, 0),
    ]
    prices = pd.Series(index=index, data=[0.5, 0.9, 0.9])
    min_temp = pd.Series(index=index, data=[15, 22, 22])
    max_temp = pd.Series(index=index, data=[28, 28, 28])

    starttime = index[0] + datetime.timedelta(minutes=minutes)

    def temp_predictor(temp, tstamp, t_delta_hours):
        return [
            {"temp": temp + 1 * t_delta_hours, "kwh": 0.5 * t_delta_hours},
            {"temp": temp - 0.5 * t_delta_hours, "kwh": 0},
        ]

    result = heatreservoir.optimize(
        starttime=starttime,
        starttemp=22,
        prices=prices,
        min_temp=min_temp,
        max_temp=max_temp,
        temp_predictor=temp_predictor,
        freq="30min",
    )
    # heatreservoir.plot_graph(result["graph"], path=result["path"], show=True)
    assert result["on_at"] == pd.Timestamp("2020-01-01 01:30:00")


def test_incomplete_graph():
    """If we start at a low, but acceptable night temperature, but are too
    low to reach the target requested (will result in graph not having nodes at
    the latest requested timestamp), ensure we turn on now

    This is due to min_temp varying within the timespan, which we
    assume can't happen for max_temp"""
    index = [
        datetime.datetime(2020, 1, 1, 0, 0, 0),
        datetime.datetime(2020, 1, 1, 4, 0, 0),
        datetime.datetime(2020, 1, 1, 9, 0, 0),
    ]
    prices = pd.Series(index=index, data=[0.5, 0.9, 0.9])
    min_temp = pd.Series(index=index, data=[15, 22, 22])
    max_temp = pd.Series(index=index, data=[28, 28, 28])

    starttime = index[0]
    assert starttime

    def temp_predictor(temp, tstamp, t_delta_hours):
        return [
            {"temp": temp + 1 * t_delta_hours, "kwh": 0.5 * t_delta_hours},
            {"temp": temp - 0.5 * t_delta_hours, "kwh": 0},
        ]

    result = heatreservoir.optimize(
        starttime=index[0],
        starttemp=17,
        prices=prices,
        min_temp=min_temp,
        max_temp=max_temp,
        temp_predictor=temp_predictor,
        freq="30min",
    )
    assert result["on_now"] is True


def test_max_temp():
    """Test that we don't spike in temperature when a spike path and
    an oscillating path has the same cost. There should be some tiny
    stabilty addons to the cost inside the code, that favours lower temperatures"""
    index = [
        datetime.datetime(2020, 1, 1, 0, 0, 0),
        datetime.datetime(2020, 1, 1, 10, 0, 0),
    ]
    prices = pd.Series(index=index, data=[0.5, 0.5])
    min_temp = pd.Series(index=index, data=[15, 15])
    max_temp = pd.Series(index=index, data=[28, 28])

    def temp_predictor(temp, tstamp, t_delta_hours):
        return [
            {"temp": temp + 1 * t_delta_hours, "kwh": 0.5 * t_delta_hours},
            {"temp": temp - 1 * t_delta_hours, "kwh": 0},
        ]

    result = heatreservoir.optimize(
        starttime=index[0],
        starttemp=15,
        prices=prices,
        min_temp=min_temp,
        max_temp=max_temp,
        temp_predictor=temp_predictor,
        freq="60min",
    )
    # heatreservoir.plot_graph(result["graph"], path=result["path"], show=True)
    assert result["cost"] == 1.5  # Ensure numerical stability cost is rounded away
    assert result["max_temp"] == 16


@pytest.mark.parametrize(
    "starttemp, expected_on_in_minutes, expected_max_temp",
    [
        (40, 0, 64.95),
        (45, 0, 70.17),
        (50, 120, 61.63),  # This is potentially suboptimal. Frequency issue?
        (55, 240, 60.27),
        (60, 300, 60.0),
        (65, 300, 65.0),  # Heats at 10, why? BUG
    ],
)
def test_waterheater(starttemp, expected_on_in_minutes, expected_max_temp):
    """Test with a more complex temp_predictor, where there is
    time-dependent temperature outtake"""

    index = [
        datetime.datetime(2020, 1, 1, 0, 0, 0),
        datetime.datetime(2020, 1, 1, 7, 0, 0),
        datetime.datetime(2020, 1, 1, 10, 0, 0),
    ]
    prices = pd.Series(index=index, data=[0.3, 0.5, 0.5])
    min_temp = pd.Series(index=index, data=[40, 50, 50])
    max_temp = pd.Series(index=index, data=[80, 80, 80])

    def temp_predictor(temp, tstamp, t_delta_hours):
        if tstamp.hour == 7:
            # morning shower
            outtake = 8.5
        else:
            # heat loss
            outtake = 2
        return [
            {
                "temp": temp + (8 - outtake - (temp - 20) / 20) * t_delta_hours,
                "kwh": 0.5 * t_delta_hours,
            },
            {"temp": temp - outtake * t_delta_hours, "kwh": 0},
        ]

    result = heatreservoir.optimize(
        starttime=index[0],
        starttemp=starttemp,
        prices=prices,
        min_temp=min_temp,
        max_temp=max_temp,
        temp_predictor=temp_predictor,
        freq="60min",
    )
    heatreservoir.plot_graph(result["graph"], path=result["path"], show=True)
    assert result["on_in_minutes"] == expected_on_in_minutes
    assert result["max_temp"] == expected_max_temp


def test_timezones():
    tz = pytz.timezone("Europe/Oslo")
    index = [
        datetime.datetime(2020, 1, 1, 0, 0, 0).astimezone(tz),
        datetime.datetime(2020, 1, 1, 7, 0, 0).astimezone(tz),
        datetime.datetime(2020, 1, 1, 10, 0, 0).astimezone(tz),
    ]
    prices = pd.Series(index=index, data=[0.3, 0.5, 0.5])
    min_temp = pd.Series(index=index, data=[40, 50, 50])
    max_temp = pd.Series(index=index, data=[80, 80, 80])

    def temp_predictor(temp, tstamp, t_delta_hours):
        return [
            {"temp": temp + 1 * t_delta_hours, "kwh": 0.5 * t_delta_hours},
            {"temp": temp - 0.1 * t_delta_hours, "kwh": 0},
        ]

    result = heatreservoir.optimize(
        starttime=index[0],
        starttemp=45,
        prices=prices,
        min_temp=min_temp,
        max_temp=max_temp,
        temp_predictor=temp_predictor,
        freq="60min",
    )
    # Visually checked that this makes sense..
    # heatreservoir.plot_graph(result["graph"], path=result["path"], show=True)
    assert result["max_temp"] == 50.9
