from pyrotun import yrmelding, persist
import pytest

# env variables must be sourced before this can run.

@pytest.mark.asyncio
async def test_yrmelding():
    pers = persist.PyrotunPersistence()
    print(pers)
    weather = await yrmelding.get_weather(persistence=pers)
    assert isinstance(weather, dict)
    assert "nexthour_description" in weather
    assert "temperatures" in weather
