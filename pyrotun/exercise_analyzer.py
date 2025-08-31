import argparse
import asyncio
import datetime
import glob
import sqlite3
from pathlib import Path

import activereader
import dotenv
import pandas as pd
import watchfiles

import pyrotun
import pyrotun.persist

dotenv.load_dotenv()

TUESDAY = 1
THURSDAY = 3
SATURDAY = 5

EXERCISE_DIR = Path.home() / "polar_dump"

logger = pyrotun.getLogger(__name__)


def speed_to_pace(speed_mps):
    if speed_mps <= 0:
        return float("inf")  # Avoid division by zero or negative speeds
    pace_min_per_km = (1000 / speed_mps) / 60
    minutes = int(pace_min_per_km)
    seconds = int((pace_min_per_km - minutes) * 60)
    return f"{minutes}:{seconds:02d}"


def lap_to_dict(lap: activereader.tcx.Lap, explicit_distance=None) -> dict:
    if explicit_distance is not None:
        # Meters per second
        pace = explicit_distance / lap.total_time_s
    else:
        # Meters per second
        pace = lap.avg_speed_ms
    expected_hr_avg = None
    if pace > 3.0:
        expected_hr_avg = 130 + (pace - 3.854) * (170 - 130) / (5.04 - 3.854)
    return {
        "date": str(lap.start_time.date()),
        "epoch": int(lap.start_time.timestamp()),
        "distance": round(lap.distance_m, 1),
        "speed_ms": round(lap.avg_speed_ms, 3),
        "hr_avg": lap.hr_avg if lap.hr_avg and 130 < lap.hr_avg < 180 else None,
        "expected_hr_avg": round(expected_hr_avg, 1) if expected_hr_avg else None,
        "hr_max": lap.hr_max if lap.hr_max and 140 < lap.hr_max < 190 else None,
        "time": lap.total_time_s,
        "gps_pace": speed_to_pace(lap.avg_speed_ms),
        "pace": speed_to_pace(pace),
    }


async def analyze_tirsdag(directory: Path) -> pd.DataFrame:
    reader = activereader.Tcx.from_file((directory / "tcx").read_text(encoding="utf-8"))
    records: list[dict] = []
    order1000 = 0
    order500 = 0
    order200 = 0
    for lap in reader.laps:
        if 900 < lap.distance_m < 1100 and 180 < lap.total_time_s < 270:
            order1000 = order1000 + 1
            record = {
                "order": order1000,
                "category": "tirsdag1000",
                **lap_to_dict(lap, explicit_distance=1000),
            }
            records.append(record)
        if 400 < lap.distance_m < 600 and 60 < lap.total_time_s < 120:
            order500 = order500 + 1
            record = {
                "order": order500,
                "category": "tirsdag500",
                **lap_to_dict(lap, explicit_distance=500),
            }
            records.append(record)
        if 150 < lap.distance_m < 250 and 25 < lap.total_time_s < 45:
            order200 = order200 + 1
            record = {
                "order": order200,
                "category": "tirsdag200",
                **lap_to_dict(lap, explicit_distance=200),
            }
            records.append(record)
    return pd.DataFrame.from_records(records)


async def analyze_torsdag(directory: Path) -> pd.DataFrame:
    reader = activereader.Tcx.from_file((directory / "tcx").read_text(encoding="utf-8"))
    records: list[dict] = []
    order40s = 0
    order200 = 0
    found3000 = False
    found200 = False
    for lap in reader.laps:
        if (
            lap.total_time_s == 40
            and 100 < lap.distance_m < 250
            and not found3000
            and not found200
        ):
            order40s = order40s + 1
            record = {"order": order40s, "category": "torsdag40s", **lap_to_dict(lap)}
            records.append(record)
        elif 2800 < lap.distance_m < 3200 and 180 * 3 < lap.total_time_s < 260 * 3:
            record = {
                "order": 1,
                "category": "torsdag3000",
                **lap_to_dict(lap, explicit_distance=3000),
            }
            records.append(record)
            found3000 = True
        elif 186 < lap.distance_m < 214 and 28 < lap.total_time_s < 42:
            # WOWOW, plukker opp 40s-dragene,
            order200 = order200 + 1
            record = {
                "order": order200,
                "category": "torsdag200",
                **lap_to_dict(lap, explicit_distance=200),
            }
            records.append(record)
            found200 = True
    return pd.DataFrame.from_records(records)


async def analyze_lordag(directory: Path) -> pd.DataFrame:
    reader = activereader.Tcx.from_file((directory / "tcx").read_text(encoding="utf-8"))
    records: list[dict] = []
    ordersiljulang = 0
    order400 = 0
    order200 = 0
    for lap in reader.laps:
        if 7 * 60 + 45 < lap.total_time_s < 10 * 60 and "2024-09-21" not in str(
            lap.start_time.date()
        ):
            # 2023-11-25 er BFG i tunnellen og treffes nesten
            # 2024-09-21 er oslo maraton og treffes :(
            # 2023-11-11 har en delt langrunde som kan detekteres ved summasjon
            ordersiljulang = ordersiljulang + 1
            record = {
                "order": ordersiljulang,
                "category": "siljulang",
                **lap_to_dict(lap),
            }
            records.append(record)
        if 350 < lap.distance_m < 450 and 50 < lap.total_time_s < 90:
            order400 = order400 + 1
            record = {
                "order": order400,
                "category": "silju400",
                **lap_to_dict(lap, explicit_distance=400),
            }
            records.append(record)
        if 150 < lap.distance_m < 250 and 25 < lap.total_time_s < 45:
            order200 = order200 + 1
            record = {
                "order": order200,
                "category": "silju200",
                **lap_to_dict(lap, explicit_distance=200),
            }
            records.append(record)
    return pd.DataFrame.from_records(records)


async def analyze_all():
    dfs = []
    dirs = sorted(glob.glob(str(EXERCISE_DIR / "20*")))
    for _dir in dirs:
        await asyncio.sleep(0.1)
        d = datetime.datetime.fromisoformat(Path(_dir).name)
        if d.weekday() == TUESDAY and d.hour == 18:
            logger.info(f"Analyzing tirsdag {_dir}")
            dfs.append(await analyze_tirsdag(Path(_dir)))
        if d.weekday() == THURSDAY and d.hour == 18:
            logger.info(f"Analyzing torsdag {_dir}")
            dfs.append(await analyze_torsdag(Path(_dir)))
        if d.weekday() == SATURDAY and d.hour == 9:
            logger.info(f"Analyzing siljustøl {_dir}")
            dfs.append(await analyze_lordag(Path(_dir)))

    data = pd.concat(dfs)
    data.to_csv(EXERCISE_DIR / "intervaller.csv")
    data.to_sql(
        "intervaller",
        sqlite3.connect(str(EXERCISE_DIR / "intervaller.db")),
        index=False,
        if_exists="replace",
    )


async def main(pers=None, dryrun=False):
    if pers is None:
        dotenv.load_dotenv()
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["websession"])
    assert pers.websession is not None

    logger.info(f"Starting watching {EXERCISE_DIR} for exercise to analyze")
    async for changes in watchfiles.awatch(EXERCISE_DIR):
        # changes is set of tuples
        logger.info(f"Detected filesystem change: {changes}")
        dirnames = set([Path(change[1]) for change in changes])
        logger.info(f"Will process directories {dirnames}")
        # dirnames are timestamps
        interval_session_found = False
        for _dirname in dirnames:
            dirname = Path(_dirname)
            if not dirname.is_dir():
                dirname = dirname.parent
            try:
                date = datetime.datetime.fromisoformat(dirname.name)
            except ValueError:
                logger.warning(f"Skipping strange-looking directory {dirname}")
                continue

            if date.weekday() in {TUESDAY, THURSDAY} and date.hour == 18:
                interval_session_found = True
            if date.weekday() == SATURDAY and date.hour == 9:
                interval_session_found = True
        if interval_session_found:
            logger.info("Analyzing tcx for all intervals")
            await analyze_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # asyncio.run(main())
    asyncio.run(analyze_all())
