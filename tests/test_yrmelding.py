from pyrotun import yrmelding

# env variables must be sourced before this can run.

def test_yrmelding():
    weather =  yrmelding.get_weather()
    assert isinstance(weather, dict)
    assert "nexthour_description" in weather
    assert "temperatures" in weather
