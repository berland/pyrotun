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
        pace = explicit_distance / lap.total_time_s
    else:
        pace = lap.avg_speed_ms
    return {
        "date": datetime.datetime.combine(
            lap.start_time.date(), datetime.datetime.min.time()
        ),
        "distance": lap.distance_m,
        "speed_ms": lap.avg_speed_ms,
        "hr_avg": lap.hr_avg,
        "hr_max": lap.hr_max,
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
        if 900 < lap.distance_m < 1100:
            order1000 = order1000 + 1
            record = {
                "order": order1000,
                "category": "tirsdag1000",
                **lap_to_dict(lap, explicit_distance=1000),
            }
            records.append(record)
        if 400 < lap.distance_m < 600:
            order500 = order500 + 1
            record = {
                "order": order500,
                "category": "tirsdag500",
                **lap_to_dict(lap, explicit_distance=500),
            }
            records.append(record)
        if 150 < lap.distance_m < 250:
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
    for lap in reader.laps:
        if lap.total_time_s == 40:
            order40s = order40s + 1
            record = {"order": order40s, "category": "torsdag40s", **lap_to_dict(lap)}
            records.append(record)
        if 2800 < lap.distance_m < 3200:
            record = {
                "order": 1,
                "category": "torsdag3000",
                **lap_to_dict(lap, explicit_distance=3000),
            }
            records.append(record)
        if 150 < lap.distance_m < 250:
            order200 = order200 + 1
            record = {
                "order": order200,
                "category": "torsdag200",
                **lap_to_dict(lap, explicit_distance=200),
            }
            records.append(record)
    return pd.DataFrame.from_records(records)


async def analyze_lordag(directory: Path) -> pd.DataFrame:
    reader = activereader.Tcx.from_file((directory / "tcx").read_text(encoding="utf-8"))
    records: list[dict] = []
    ordersiljulang = 0
    order400 = 0
    order200 = 0
    for lap in reader.laps:
        if 7 * 60 < lap.total_time_s < 10 * 60:
            ordersiljulang = ordersiljulang + 1
            record = {
                "order": ordersiljulang,
                "category": "siljulang",
                **lap_to_dict(lap),
            }
            records.append(record)
        if 350 < lap.distance_m < 450:
            order400 = order400 + 1
            record = {
                "order": order400,
                "category": "silju400",
                **lap_to_dict(lap, explicit_distance=400),
            }
            records.append(record)
        if 150 < lap.distance_m < 250:
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
    dirs = sorted(glob.glob(str(EXERCISE_DIR / "2025*")))
    for _dir in dirs:
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
    asyncio.run(main())
    # asyncio.run(analyze_all())
