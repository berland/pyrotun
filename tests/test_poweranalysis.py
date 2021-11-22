from pyrotun import poweranalysis, persist
import pytest

# env variables must be sourced before this can run.


@pytest.mark.asyncio
async def test_make_heatingmodel():
    pers = persist.PyrotunPersistence()
    await pers.ainit(["influxdb"])
    print(pers)
    model = await poweranalysis.make_heatingmodel(pers)
    assert isinstance(model, dict)
    coeffs = model["powerneed"].coef_[0]
    print(coeffs)
    assert 2 < coeffs[0] < 3  # indoorderivative
    assert 0.1 < coeffs[1] < 0.3  # indoorvsoutdoor
    assert 0.1 < coeffs[2] < 0.2  # IrradiationProxy
