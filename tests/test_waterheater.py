import datetime

import networkx
import numpy as np
import pandas as pd
from matplotlib import pyplot

from pyrotun import waterheater


def test_waterheaterobject():
    w_heater = waterheater.WaterHeater()

    assert w_heater.waterusageprofile is None
    assert w_heater.meanwatertemp is None


def test_make_heatloss_diffusion_model():
    mock_temps = pd.DataFrame(
        index=pd.date_range(
            start=datetime.datetime.now().replace(minute=0, second=0, microsecond=0),
            freq="1h",
            periods=4,
        ),
        data=[60, 59.5, 59, 58.5],
        columns=["watertemp"],
    )

    # Linear data gives a constant model for the heatloss, only intercept:
    (intercept, coef) = waterheater.make_heatloss_diffusion_model(mock_temps)
    assert (intercept, coef) == (-0.5, 0)

    w_heater = waterheater.WaterHeater()
    w_heater.heatlossdiffusionmodel = (intercept, coef)

    # Diffusionloss is then constant for all temperatures:
    assert w_heater.diffusionloss(60) == 0.5
    assert w_heater.diffusionloss(50) == 0.5
    assert w_heater.diffusionloss(80) == 0.5

    # Another testdata where lower temperatures gives lower heatloss:
    mock_temps = pd.DataFrame(
        index=pd.date_range(
            start=datetime.datetime.now().replace(minute=0, second=0, microsecond=0),
            freq="1h",
            periods=4,
        ),
        data=[60, 59.8, 59.7, 59.65],
        columns=["watertemp"],
    )
    (intercept, coef) = waterheater.make_heatloss_diffusion_model(mock_temps)
    assert np.isclose((intercept, coef), (29.8, -0.5)).all()
    w_heater.heatlossdiffusionmodel = (intercept, coef)

    # Diffusionloss is then constant for all temperatures:
    assert np.isclose(w_heater.diffusionloss(60), 0.2)
    assert np.isclose(w_heater.diffusionloss(59.8), 0.1)
    assert np.isclose(w_heater.diffusionloss(59.7), 0.05)


def test_diffusionloss():
    w_heater = waterheater.WaterHeater()
    # Time period is implicit in the heatlossmodels coefficient.

    # This should mean one degree loss pr. hour if at 60 degrees:
    w_heater.heatlossdiffusionmodel = (0, -1 / 60)
    assert w_heater.diffusionloss(60) == 1
    # A little bit more if we are above 60 degrees:
    assert np.isclose(w_heater.diffusionloss(70), 1.16666667)
    # and vice versa:
    assert np.isclose(w_heater.diffusionloss(50), 0.8333333)


def test_make_graph():
    mock_prices = pd.DataFrame(
        index=pd.date_range(
            start=datetime.datetime.now().replace(minute=0, second=0, microsecond=0),
            freq="1h",
            periods=3,
        ),
        data=[0.0, 1, 1],
        columns=["NOK/KWh"],
    )
    print(mock_prices)

    w_heater = waterheater.WaterHeater()
    # Mock the object instead of running ainit()
    w_heater.waterusageprofile = pd.DataFrame(
        data=[dict(day=0, hour=0, minute=0, waterdiff=-1)]
    ).set_index(["day", "hour", "minute"])

    # This gives 0.1 degree temp loss every hour, indep of temp
    w_heater.heatlossdiffusionmodel = (-0.1, 0)
    assert w_heater.diffusionloss(40) == 0.1
    assert w_heater.diffusionloss(80) == 0.1
    w_heater.meanwatertemp = 65

    graph = w_heater.future_temp_cost_graph(
        starttemp=70,
        prices_df=mock_prices,
        vacation=False,
        starttime=mock_prices.index[0],
    )
    print(f"The graph looks like {graph.nodes}")
    _, ax = pyplot.subplots()
    waterheater.plot_graph(graph, ax=ax)
    path = networkx.shortest_path(
        graph,
        source=(mock_prices.index[0], 70),
        target=(mock_prices.index[-1], 69.4),
        weight="cost",
    )
    waterheater.plot_path(path, ax=ax)
    # pyplot.show()


def test_predict_tempincrease():
    assert waterheater.predict_tempincrease(pd.Timedelta(0)) == 0
    assert np.isclose(
        waterheater.predict_tempincrease(pd.Timedelta(15, unit="m")), 2.880787
    )
    assert 28.8 < waterheater.predict_tempincrease(pd.Timedelta(2.5, unit="h")) < 29


def test_waterheatercost():
    assumed_watts = 2600
    assumed_kwatts = assumed_watts / 1000
    assert assumed_watts == waterheater.WATTAGE

    # Price in NOK/KWh, gives result in NOK:
    assert waterheater.waterheatercost(1, pd.Timedelta(1, unit="h")) == (
        assumed_kwatts,
        assumed_kwatts,
    )
    assert waterheater.waterheatercost(0, pd.Timedelta(1, unit="h")) == (
        0,
        assumed_kwatts,
    )
    assert waterheater.waterheatercost(1, pd.Timedelta(30, unit="min")) == (
        assumed_kwatts / 2,
        assumed_kwatts / 2,
    )
