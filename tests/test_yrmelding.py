from pyrotun import yrmelding

def test_yrmelding():
    weather =  yrmelding.get_weather()
    assert isinstance(weather, dict)
    assert "nexthour_description" in weather
    assert "temperatures" in weather
