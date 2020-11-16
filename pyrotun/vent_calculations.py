import metpy.calc as mcalc
from metpy.units import units

import pyrotun
from pyrotun.connections import openhab

logger = pyrotun.getLogger(__name__)

# Sensor-names in HVAC-unit:
SENSORS = ["avkast", "fraluft", "tilluft", "inntak"]

# Notes:
# luftmengde innstilling lav er 18.2 liter pr. sek. Regnet ut
# fra snittentalpiøkning fra ettervarmer og tilhørende strømforbruk
# når varmeveksler var død.
#
# 1.269 kg/m3 lufttetthet, 313 W snitteffekt over 20 timer,
# 13.0 kJ/kg entalpiøkning. Feilmargin er temperturøkning fra
# stillestående varmeveksler (10% virkningsgrad)


# https://www.engineeringtoolbox.com/enthalpy-moist-air-d_683.html
# Specific heat for air
C_PA = 1.006  # kJ/KgC
# Specific heat for water vapor
C_PW = 1.84  # kJ/kgC
# Evaporation heat
H_WE = 2501  # kJ/kg


def enthalpy(temp, rh, pressure):
    """Temperatur er i Celsius, rh er i tall mellom 0 og 1,
    trykk er i millibar.

    Returenhet er i kJ/kg
    """
    assert 0 <= rh <= 1
    return C_PA * temp + hum_ratio(rh, temp, pressure) * h_w(temp)


def h_w(temp):
    """Specific enthalpy of water vapor - the latent heat"""
    return C_PW * temp + H_WE


def hum_ratio(rh, temp, pressure):
    """Mixing ratio.

    0 < RH < 1
    """
    assert 0 <= rh <= 1
    saturated_mixing_ratio = mcalc.saturation_mixing_ratio(
        pressure * units("mbar"), temp * units("degC")
    ).magnitude
    return rh * saturated_mixing_ratio


def main(connections=None):
    if connections is None:
        connections = {
            "openhab": openhab.OpenHABConnection(),
        }
    temps = {}
    rhs = {}
    enthalpies = {}
    mixs = {}

    pressure = float(connections["openhab"].get_item("Netatmo_stue_trykk"))

    for sensor in SENSORS:
        temps[sensor] = float(
            connections["openhab"].get_item("Sensor_" + sensor + "_temperatur")
        )
        #  RH in unit 0 to 1, not percent.
        rhs[sensor] = round(
            float(connections["openhab"].get_item("Sensor_" + sensor + "_fuktighet"))
            / 100,
            4,
        )
        enthalpies[sensor] = round(enthalpy(temps[sensor], rhs[sensor], pressure), 4)

        mixs[sensor] = round(hum_ratio(rhs[sensor], temps[sensor], pressure), 6)

        connections["openhab"].set_item(
            "Sensor_" + sensor + "_entalpi", enthalpies[sensor]
        )

        connections["openhab"].set_item(
            "Sensor_" + sensor + "_masseforhold", mixs[sensor] * 1000
        )
    logger.debug("temps: %s", temps)
    logger.debug("rhs: %s", rhs)
    logger.debug("enthalpies: %s", enthalpies)
    logger.debug("mixs: %s", mixs)

    # temp_virkningsgrad = round((temps['tilluft'] - temps['inntak'])/(temps['fraluft'] - temps['avkast']), 3)
    # https://www.sintef.no/globalassets/project/annex32/oppvarmingssystemer_tra6182_20061.pdf  ligning 5.1
    # og : https://www.engineeringtoolbox.com/heat-recovery-efficiency-d_201.html

    temp_virkningsgrad = round(
        (temps["fraluft"] - temps["avkast"]) / (temps["fraluft"] - temps["inntak"]), 3
    )

    # Beregner en alternativ virkningsgrad (som kan bli over 1 tror jeg,
    # ihvertfall med ettervarmer på)
    temp_virkningsgradopp = round(
        (temps["tilluft"] - temps["inntak"]) / (temps["fraluft"] - temps["inntak"]), 3
    )
    enthalpy_efficiency = round(
        (enthalpies["fraluft"] - enthalpies["avkast"])
        / (enthalpies["fraluft"] - enthalpies["inntak"]),
        3,
    )
    moisture_efficiency = round(
        (mixs["fraluft"] - mixs["avkast"]) / (mixs["fraluft"] - mixs["inntak"]), 3
    )

    connections["openhab"].set_item(
        "Ventilasjon_virkningsgrad_temperatur", temp_virkningsgrad, log=True
    )
    connections["openhab"].set_item(
        "Ventilasjon_virkningsgrad_temperaturoppvarming",
        temp_virkningsgradopp,
        log=True,
    )
    connections["openhab"].set_item(
        "Ventilasjon_virkningsgrad_entalpi", enthalpy_efficiency, log=True
    )
    connections["openhab"].set_item(
        "Ventilasjon_virkningsgrad_fukt", moisture_efficiency, log=True
    )

    fuktproduksjon = round((mixs["fraluft"] - mixs["tilluft"]) * 1000, 3)
    fuktfrahusnetto = round((mixs["avkast"] - mixs["inntak"]) * 1000, 3)
    fuktfravarmeveksler = round((mixs["tilluft"] - mixs["inntak"]) * 1000, 3)
    connections["openhab"].set_item("Fuktproduksjon_hus", fuktproduksjon, log=True)
    connections["openhab"].set_item("Ventilasjon_hustorking", fuktfrahusnetto)

    connections["openhab"].set_item(
        "Ventilasjon_fuktvekslingsmengde", fuktfravarmeveksler, log=True
    )


if __name__ == "__main__":
    main()
