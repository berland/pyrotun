import argparse
import asyncio
import datetime
import os
from functools import partial
from pathlib import Path
from typing import Optional, Union

import dotenv
import numpy as np
import pandas as pd
import pytz
import yaml
from matplotlib import pyplot

import pyrotun
from pyrotun import heatreservoir, persist  # noqa
from pyrotun.connections import localpowerprice

logger = pyrotun.getLogger(__name__)

# Positive number means colder house:
COLDER_FOR_POWERSAVING = 0


FLOORSFILE = "floors.yml"
# If the above file exists, it will overwrite the FLOORS variable.
FLOORS = {
    # This serves as an example:
    "Bad": {
        "sensor_item": "Termostat_Bad_SensorGulv",
        "setpoint_item": "Termostat_Bad_SetpointHeating",
        "setpoint_base": "temperature",  # or "target"
        "delta": 0,
        "heating_rate": 5,
        "cooling_rate": -0.1,
        "cooling_rate_winter": -0.6,
        "setpoint_force": 1,
        "wattage": 600,
        "maxtemp": 33,
        "backup_setpoint": 24,
    }
}

TIMEDELTA_MINUTES = 60  # minimum is 8 minutes!!
PD_TIMEDELTA = str(TIMEDELTA_MINUTES) + "min"
VACATION_ITEM = "Ferie"
WEATHER_COMPENSATION_ITEM = "Estimert_soloppvarming"
WEATHER_COMPENSATION_MULTIPLIER = 2
BACKUPSETPOINT = 22


def interpolate_summer_winter(
    day: Union[int, datetime.date], summer_value: float, winter_value: float
) -> float:
    if isinstance(day, datetime.date):
        day = (day - day.replace(month=1, day=1)).days
    assert 0 < day < 366
    interpolant = abs(day - 365 / 2) / (365 / 2)
    return (1 - interpolant) * summer_value + interpolant * winter_value


async def analyze_history(pers, selected_floors):
    daysago: int = 12 * 30
    skipbackwardsdays = 0 * 30
    minutegrouping = 10
    if (Path(__file__).parent / FLOORSFILE).exists():
        logger.info("Loading floor configuration from %s", FLOORSFILE)
        FLOORS = yaml.safe_load((Path(__file__).parent / FLOORSFILE).read_text())
    else:
        logger.warning("Did not find yml file with floor configuration, dummy run")
    for floor in selected_floors:
        logger.info("Analyzing heating and cooling rates for floor %s", floor)
        query = (
            f"SELECT difference(mean(value)) "
            f"FROM {FLOORS[floor]['sensor_item']} "
            f"where time > now() - {daysago}d "
            f"and time < now() - {skipbackwardsdays}d "
            f"group by time({minutegrouping}m) "
            "fill(linear) "
        )
        logger.info(query)
        resp = await pers.influxdb.dframe_query(query) * (60 / minutegrouping)

        # Crop away impossible derivatives:
        resp = resp[resp < 6]
        resp = resp[resp > -2]
        resp.hist(bins=30)
        pyplot.title(floor)
        pyplot.axvline(x=FLOORS[floor]["heating_rate"], color="red", linewidth=2)
        pyplot.axvline(x=0, color="black", linewidth=2)
        pyplot.axvline(x=FLOORS[floor]["cooling_rate"], color="red", linewidth=2)
        pyplot.axvline(
            x=FLOORS[floor]["cooling_rate_winter"], color="purple", linewidth=2
        )
        pyplot.show()


async def amain(
    pers=None,
    dryrun=False,
    plot=False,
    plotnodes=False,
    floors=None,
    freq="10min",
    vacation="auto",
    hoursago=0,
    minutesago=0,
    analyze=False,
):
    """Called from service or interactive"""
    assert hoursago >= 0
    assert minutesago >= 0

    floorsfile = Path(__file__).parent / FLOORSFILE
    if floorsfile.exists():
        logger.info("Loading floor configuration from %s", str(floorsfile))
        FLOORS = yaml.safe_load(floorsfile.read_text())
    else:
        # If pyrotun is installed into site_packags, floors.yml might
        # not have followed. Try looking relative to cwd (set by systemd)
        floorsfile = Path("pyrotun") / FLOORSFILE
        if floorsfile.exists():
            logger.info("Loading floor configuration from %s", str(floorsfile))
            FLOORS = yaml.safe_load(floorsfile.read_text())
        else:
            logger.error(
                f"Did not find yml file {str(floorsfile)} "
                "with floor configuration, dummy run"
            )

    closepers = False
    if pers is None:
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["tibber", "influxdb", "openhab"])
        closepers = True

    if floors is None:
        selected_floors = FLOORS.keys()
    else:
        selected_floors = floors

    if hoursago > 0 or minutesago > 0:
        assert dryrun is True, "Only dryrun when jumping back in history"

    if analyze:
        await analyze_history(pers, selected_floors)
        if closepers:
            await pers.aclose()
        return

    prices_df = await pers.tibber.get_prices()

    # Grid rental is time dependent:
    prices_df = prices_df.copy()
    prices_df["NOK/KWh"] += localpowerprice.get_gridrental(prices_df.index)

    if not vacation or vacation == "auto":
        vacation = await pers.openhab.get_item(VACATION_ITEM, datatype=bool)
    else:
        if vacation.lower() == "on":
            vacation = True
            logger.info("Vacation forced to on")
        else:
            logger.info("Vacation forced to off")
            vacation = False

    # Summer mode; truncate maxtemp to two degrees above setpoint:
    if 4 < datetime.datetime.now().month < 10:
        for floor in FLOORS:
            # Bad coding: overwriting a "constant" variable.
            FLOORS[floor]["maxtemp"] = 25 + 2 + FLOORS[floor].get("delta", 0)

    for floor in selected_floors:
        logger.info("Starting optimization for floor %s", floor)
        if hoursago > 0:
            logger.info("* Jumping back %s hours", str(hoursago))
        if minutesago > 0:
            logger.info("* Jumping back %s minutes", str(minutesago))
        currenttemp = await pers.influxdb.get_item(
            FLOORS[floor]["sensor_item"],
            datatype=float,
            ago=hoursago * 60 + minutesago,
            unit="m",
        )
        logger.debug(f"Current temperature is {currenttemp}")
        if (
            currenttemp is None
            or str(currenttemp) == "UNDEF"
            or not (12 < currenttemp < 40)
        ):
            logger.error(f"Currenttemp was {currenttemp}, can't be right, giving up")
            # Backup temperature:
            if not dryrun:
                await pers.openhab.set_item(
                    FLOORS[floor]["setpoint_item"],
                    str(FLOORS[floor]["backup_setpoint"]),
                    log=True,
                )
            continue

        if "delta" in FLOORS[floor]:
            delta = FLOORS[floor]["delta"]
        else:
            delta = 0

        # Deduct more in to correct for good weather:
        weathercompensation = (
            await pers.openhab.get_item(WEATHER_COMPENSATION_ITEM, datatype=float)
            * WEATHER_COMPENSATION_MULTIPLIER
        )
        delta = delta - round(weathercompensation * 2.0) / 2.0 - COLDER_FOR_POWERSAVING

        if currenttemp > FLOORS[floor]["maxtemp"]:
            logger.info("Floor is above allowed maxtemp, turning OFF")
            if not dryrun:
                min_temp_now = temp_requirement(
                    datetime.datetime.now(), vacation=vacation, delta=delta
                )
                await pers.openhab.set_item(
                    FLOORS[floor]["setpoint_item"],
                    str(max(10, min_temp_now - FLOORS[floor]["setpoint_force"])),
                    log=True,
                )
            continue

        tz = pytz.timezone(os.getenv("TIMEZONE"))
        starttime = (
            datetime.datetime.now()
            - datetime.timedelta(hours=hoursago)
            - datetime.timedelta(minutes=minutesago)
        ).astimezone(tz)

        if "cooling_rate_winter" in FLOORS[floor]:
            cooling_rate_interpolated = interpolate_summer_winter(
                datetime.datetime.today(),
                FLOORS[floor]["cooling_rate"],
                FLOORS[floor]["cooling_rate_winter"],
            )
        else:
            cooling_rate_interpolated = FLOORS[floor]["cooling_rate"]

        result = heatreservoir.optimize(
            starttime=starttime,
            starttemp=currenttemp,
            prices=prices_df["NOK/KWh"],
            min_temp=partial(temp_requirement, vacation=vacation, delta=delta),
            max_temp=FLOORS[floor]["maxtemp"],
            temp_predictor=partial(
                floortemp_predictor,
                wattage=FLOORS[floor]["wattage"],
                heating_rate=FLOORS[floor]["heating_rate"],
                cooling_rate=cooling_rate_interpolated,
            ),
            freq=freq,
        )
        if not result["graph"]:
            logger.warning(
                f"Temperature ({currenttemp}) below minimum, should force on"
            )
            if not dryrun:
                min_temp = temp_requirement(
                    datetime.datetime.now(), vacation=vacation, delta=delta
                )
                await pers.openhab.set_item(
                    FLOORS[floor]["setpoint_item"],
                    str(min_temp + FLOORS[floor]["setpoint_force"]),
                    log=True,
                )
            continue

        if plotnodes:
            heatreservoir.plot_graph(
                result["graph"], path=result["path"], ax=None, show=True
            )  # Plots all nodes in graph.

        logger.info(
            f"Cost for floor {floor} is {result['cost']:.2f}, "
            f"KWh is {result['cost']:.2f}"
        )

        # Calculate new setpoint
        if FLOORS[floor]["setpoint_base"] == "temperature":
            # This means we trust the sensor works fine together with the thermostat
            setpoint_base = result["path"][0][1]
        elif FLOORS[floor]["setpoint_base"] == "target":
            setpoint_base = temp_requirement(
                starttime,
                vacation=vacation,
                delta=delta,
            )
        else:
            logger.error("Wrong configuration for floor %s", floor)
            setpoint_base = 20

        if result["on_now"]:
            setpoint = setpoint_base + FLOORS[floor]["setpoint_force"]
            logger.info(f"Turning floor {floor} ON now")
        else:
            setpoint = setpoint_base - FLOORS[floor]["setpoint_force"]
            logger.info(f"Turning floor {floor} OFF now")

        setpoint = max(11, int(setpoint))  # This is Heat-it thermostat specific

        if not dryrun:
            await pers.openhab.set_item(
                FLOORS[floor]["setpoint_item"],
                str(setpoint),
                log=True,
            )
        if "on_at" in result and result["on_at"] is not None:
            logger.info(
                f"Will turn floor {floor} on at {result['on_at']}",
            )
        else:
            logger.info(
                f"Will not turn floor {floor} on in price-future",
            )

        if plot:
            _fig, ax = pyplot.subplots()
            heatreservoir.plot_path(result["path"], ax=ax, show=False)
            pyplot.title(floor)

            # Yesterdays temperatures shifted forward by 24h:
            hist_temps = await pers.influxdb.get_series(
                FLOORS[floor]["sensor_item"],
                since=datetime.datetime.now() - datetime.timedelta(hours=48),
            )
            # Smoothen curve:
            hist_temps = hist_temps.resample("15min").mean().interpolate(method="time")
            hist_temps.index = hist_temps.index.tz_convert(
                "Europe/Oslo"
            ) + datetime.timedelta(hours=24)
            ax.plot(
                hist_temps.index.to_pydatetime(),
                hist_temps[FLOORS[floor]["sensor_item"]],
                color="green",
                label="direct",
                alpha=0.7,
            )

            prices_df["mintemp"] = (
                prices_df.reset_index()["index"]
                .apply(temp_requirement, vacation=vacation, delta=delta)
                .values
            )
            ax.step(
                prices_df.index.to_pydatetime(),
                prices_df["mintemp"],
                where="post",
                color="blue",
                alpha=0.4,
            )

            # Prices on a secondary y-axis:
            ax2 = ax.twinx()
            ax2.step(
                prices_df.index.to_pydatetime(),
                prices_df["NOK/KWh"],
                where="post",
                alpha=0.2,
            )

            ax.set_xlim(
                left=prices_df.index.to_pydatetime().min(),
                right=prices_df.index.to_pydatetime().max(),
            )
            pyplot.show()

    if closepers:
        await pers.aclose()


def hightemp_penalty(temp, powerprice, t_delta_hours):
    # Add cost of 160W pr degree, spread on all heaters, say 10, so 16 watt pr degree.
    # Positive values for all values > 15 degrees.

    # Multiply the extra watts by this, as to some degree we are eroding
    # the job to be done by the heat pump which has a high COP
    extra_factor = 5

    # Important to not return a positive number to have a stable algoritm.
    overshoot_temp = max(temp - 15, 0.001)
    extra_kwh = overshoot_temp * 16 * extra_factor / 1000
    return powerprice * extra_kwh * t_delta_hours


def floortemp_predictor(
    temp, tstamp, t_delta_hours, wattage=0, heating_rate=0, cooling_rate=0
):
    # cooling_rate must be negative for cooling to occur
    return [
        {"temp": temp + heating_rate * t_delta_hours, "kwh": wattage * t_delta_hours},
        {"temp": temp + cooling_rate * t_delta_hours, "kwh": 0},
    ]


def temp_requirement(
    timestamp: pd.Timestamp,
    vacation: bool = False,
    master_correction: Optional[pd.Series] = None,
    delta: float = 0,
) -> float:
    """
    Args:
        timestamp
        vacation
        master_correction: A time-dependent correction added
            to the master temperature
        delta: Room-dependent (constant in time) correction added
            to master temperature. Positive value means warmer.
    """
    hour = timestamp.hour
    weekday = timestamp.weekday()  # Monday = 0, Sunday = 6
    friday = 4
    corona = False
    if vacation:
        # Ferie
        return 15 + delta
    if hour < 6 or hour > 21:
        # Natt:
        return 18 + delta
    if corona is False and (hour > 7 and hour < 13 and weekday <= friday):
        # Dagsenking:
        return 18 + delta
    if hour > 16 and hour < 22:
        # Ettermiddag
        return 22 + delta
    # Morgen og middagstid, 25 er egentlig komfort her!!
    return 23 + delta


def path_onoff(path):
    """Compute a pandas series with 1 or 0 whether the heater
    should be on or off along a path (computed assuming
    that if temperature increases, heater must be on)"""
    timestamps = [node[0] for node in path][:-1]  # skip the last one
    temps = [node[1] for node in path]
    onoff = pd.Series(temps).diff().shift(-1).dropna()
    onoff.index = timestamps
    return np.maximum(0, np.sign(onoff))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set", action="store_true", help="Actually submit setpoint to OpenHAB"
    )
    parser.add_argument("--plot", action="store_true", help="Make plots")
    parser.add_argument("--plotnodes", action="store_true", help="Make node plots")
    parser.add_argument("--floors", nargs="*", help="Floornames")
    parser.add_argument(
        "--freq", type=str, help="Time frequency, default 10min", default="10min"
    )
    parser.add_argument("--vacation", type=str, help="ON or OFF or auto", default="")
    parser.add_argument("--analyze", action="store_true", help="Analyze mode")
    parser.add_argument("--hoursago", type=int, help="Step back some hours", default=0)
    parser.add_argument(
        "--minutesago", type=int, help="Step back some minutes", default=0
    )

    return parser


if __name__ == "__main__":
    # Interactive testing:
    dotenv.load_dotenv()

    parser = get_parser()
    args = parser.parse_args()

    asyncio.run(
        amain(
            pers=None,
            dryrun=not args.set,
            plot=args.plot,
            plotnodes=args.plotnodes,
            floors=args.floors,
            freq=args.freq,
            vacation=args.vacation,
            hoursago=args.hoursago,
            minutesago=args.minutesago,
            analyze=args.analyze,
        )
    )
