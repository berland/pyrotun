import argparse
import asyncio
import datetime
import glob
import sqlite3
from pathlib import Path

import activereader
import dotenv
import numpy as np
import pandas as pd
import watchfiles
from geopy import distance

import pyrotun
import pyrotun.persist

dotenv.load_dotenv()

TUESDAY = 1
THURSDAY = 3
SATURDAY = 5

EXERCISE_DIR = Path.home() / "polar_dump"

logger = pyrotun.getLogger(__name__)

# midpoints = pd.read_csv("intervaller.csv").groupby("category").agg("mean")[["lon", "lat"]]
# midpoints.to_dict(orient="index")
LAP_CENTROIDS = {
    "torsdag40s": {"lon": 5.282329871212935, "lat": 60.29923441179828},
    "torsdag3000": {"lon": 5.2924457330164, "lat": 60.29050007675862},
    "siljulang": {"lon": 5.304600518840247, "lat": 60.29737223157386},
    "silju200": {"lon": 5.317422952368899, "lat": 60.291241335829895},
    "silju400": {"lon": 5.317486788439951, "lat": 60.29053396667061},
    "torsdag200": {"lon": 5.318714879288788, "lat": 60.29041950444813},
    "tirsdag200": {"lon": 5.319548818252679, "lat": 60.29029686700665},
    "tirsdag500": {"lon": 5.341696926437765, "lat": 60.27289127429241},
    "tirsdag1000": {"lon": 5.3658868288692, "lat": 60.27033820109348},
}


def speed_to_pace(speed_mps):
    if speed_mps <= 0:
        return float("inf")  # Avoid division by zero or negative speeds
    pace_min_per_km = (1000 / speed_mps) / 60
    minutes = int(pace_min_per_km)
    seconds = int((pace_min_per_km - minutes) * 60)
    return f"{minutes}:{seconds:02d}"


def seconds_pr_km_to_pace(seconds):
    minutes = int(seconds / 60)
    seconds = int(seconds - minutes * 60)
    return f"{minutes}:{seconds:02d}"


def lap_to_dict(lap: activereader.tcx.Lap, explicit_distance=None) -> dict:
    lats = [p.lat for p in lap.trackpoints if p.lat is not None]
    lons = [p.lon for p in lap.trackpoints if p.lon is not None]
    if explicit_distance is not None:
        # Meters per second
        pace = explicit_distance / lap.total_time_s
    else:
        # Meters per second
        pace = lap.avg_speed_ms
    expected_hr_avg = None
    if pace is not None and pace > 3.0:
        expected_hr_avg = 130 + (pace - 3.854) * (170 - 130) / (5.04 - 3.854)
    return {
        "date": str(lap.start_time.date()),
        "epoch": int(lap.start_time.timestamp()),
        "distance": round(lap.distance_m, 1),
        "speed_ms": round(lap.avg_speed_ms or 0, 3),
        "hr_avg": lap.hr_avg if lap.hr_avg and 140 < lap.hr_avg < 180 else None,
        "expected_hr_avg": round(expected_hr_avg, 1) if expected_hr_avg else None,
        "hr_max": lap.hr_max if lap.hr_max and 140 < lap.hr_max < 190 else None,
        "time": lap.total_time_s,
        "gps_pace": speed_to_pace(lap.avg_speed_ms or 0),
        "pace": speed_to_pace(pace or 0),
        "lon": np.nanmean(lons) if lons else None,
        "lat": np.nanmean(lats) if lats else None,
    }


def lap_centroid_dist(category: str, centroid_dict: dict) -> float:
    return distance.distance(
        tuple(LAP_CENTROIDS[category].values()),
        (centroid_dict["lon"], centroid_dict["lat"]),
    ).meters


async def analyze_tirsdag(directory: Path) -> pd.DataFrame:
    reader = activereader.Tcx.from_file((directory / "tcx").read_text(encoding="utf-8"))
    records: list[dict] = []
    order1000 = 0
    order500 = 0
    order200 = 0
    for lap in reader.laps:
        centroid = lap_to_dict(lap)
        if (
            900 < lap.distance_m < 1100
            and 180 < lap.total_time_s < 270
            and lap_centroid_dist("tirsdag1000", centroid) < 4500
        ):
            order1000 = order1000 + 1
            record = {
                "order": order1000,
                "category": "tirsdag1000",
                **lap_to_dict(lap, explicit_distance=1000),
            }
            record["cat_centroid_dist"] = lap_centroid_dist("tirsdag1000", centroid)
            records.append(record)
        elif (
            400 < lap.distance_m < 600
            and 60 < lap.total_time_s < 120
            and lap_centroid_dist("tirsdag500", centroid) < 4000
        ):
            order500 = order500 + 1
            record = {
                "order": order500,
                "category": "tirsdag500",
                **lap_to_dict(lap, explicit_distance=500),
            }
            record["cat_centroid_dist"] = lap_centroid_dist("tirsdag500", centroid)
            records.append(record)
        if 150 < lap.distance_m < 250 and 25 < lap.total_time_s < 45:
            order200 = order200 + 1
            record = {
                "order": order200,
                "category": "tirsdag200",
                **lap_to_dict(lap, explicit_distance=200),
            }
            record["cat_centroid_dist"] = lap_centroid_dist("tirsdag200", centroid)
            records.append(record)
    return pd.DataFrame.from_records(records)


async def make_description(directory: Path) -> dict[str, str]:
    d = datetime.datetime.fromisoformat(directory.name)
    if d.weekday() == TUESDAY and d.hour == 18:
        data = await analyze_tirsdag(directory)
        if data.empty:
            return {}
        rows_1000 = data["category"] == "tirsdag1000"
        rows_500 = data["category"] == "tirsdag500"
        rows_200 = data["category"] == "tirsdag200"
        hr_cost = round(
            (data[rows_1000]["hr_avg"] - data[rows_1000]["expected_hr_avg"]).mean(),
            1,
        )
        if sum(rows_1000):
            startfart1000 = seconds_pr_km_to_pace(
                data[rows_1000]["time"].head(1).values[0]
            )
            sluttfart1000 = seconds_pr_km_to_pace(
                data[rows_1000]["time"].tail(1).values[0]
            )
            desc_1000 = (
                f"{startfart1000}->{sluttfart1000} på 1000 (pulskost {hr_cost}), "
            )
        else:
            desc_1000 = ""
        if sum(rows_500):
            startfart500 = seconds_pr_km_to_pace(
                data[rows_500]["time"].head(1).values[0] * 2
            )
            sluttfart500 = seconds_pr_km_to_pace(
                data[rows_500]["time"].tail(1).values[0] * 2
            )
            desc_500 = f"{startfart500}->{sluttfart500} på 500, "
        else:
            desc_500 = ""
        if sum(rows_200):
            startfart200 = int(data[rows_200]["time"].head(1).values[0])
            sluttfart200 = int(data[rows_200]["time"].tail(1).values[0])
            desc_200 = f"{startfart200}->{sluttfart200} på 200"
        else:
            desc_200 = ""
        return {
            "title": f"BFG {sum(rows_1000)}x1000m, {sum(rows_500)}x500m, {sum(rows_200)}x200m, 6x60m",
            "desc": f"{desc_1000}{desc_500}{desc_200}.",
        }
    if d.weekday() == THURSDAY and d.hour == 18:
        data = await analyze_torsdag(directory)
        if data.empty:
            return {}
        rows_40s = data["category"] == "torsdag40s"
        rows_3000 = data["category"] == "torsdag3000"
        rows_200 = data["category"] == "torsdag200"
        if sum(rows_40s):
            meanlength40s = int(data[rows_40s]["distance"].mean())
            desc_40s = f"{meanlength40s}m pr 40s drag, "
        else:
            # (hvis noen andre har ropt f.eks., som 2025-02-06)
            desc_40s = ""
        if sum(rows_3000):
            hr_cost = round(
                (data[rows_3000]["hr_avg"] - data[rows_3000]["expected_hr_avg"]).mean(),
                1,
            )
            desc_3000 = (
                seconds_pr_km_to_pace(data[rows_3000]["time"].values[0] / 3)
                + f" på 3000m (pulskost{hr_cost}), "
            )
        else:
            desc_3000 = ""
        if sum(rows_200):
            startfart200 = int(data[rows_200]["time"].head(1).values[0])
            sluttfart200 = int(data[rows_200]["time"].tail(1).values[0])
            desc_200 = f"{startfart200}->{sluttfart200} på 200"
        else:
            desc_200 = ""
        return {
            "title": (
                f"BFG {sum(rows_40s)}x40s, {'3000m, ' if sum(rows_3000) else ''}"
                f"{sum(rows_200)}x200m, 6x60m"
            ),
            "desc": f"{desc_40s}{desc_3000}{desc_200}",
        }
    if d.weekday() == SATURDAY and d.hour == 9:
        data = await analyze_lordag(directory)
        if data.empty:
            return {}
        rows_lang = data["category"] == "siljulang"
        rows_400 = data["category"] == "silju400"
        rows_200 = data["category"] == "silju200"
        if sum(rows_lang):
            # _startfart_lang = seconds_pr_km_to_pace(
            #    data[rows_lang]["time"].head(1).values[0]
            # )
            # _sluttfart_lang = seconds_pr_km_to_pace(
            #    data[rows_lang]["time"].tail(1).values[0]
            # )
            langefarter = "-".join(
                [seconds_pr_km_to_pace(t) for t in data[rows_lang]["time"].values]
            )
            hr_cost_lang = round(
                (data[rows_lang]["hr_avg"] - data[rows_lang]["expected_hr_avg"]).mean(),
                1,
            )
            desc_lang = (
                f"{sum(rows_lang)} lange på {langefarter} (pulskost {hr_cost_lang}), "
            )
        else:
            desc_lang = ""
        if sum(rows_400):
            startfart400 = int(data[rows_400]["time"].head(1).values[0])
            sluttfart400 = int(data[rows_400]["time"].tail(1).values[0])
            hr_cost_400 = round(
                (data[rows_400]["hr_avg"] - data[rows_400]["expected_hr_avg"]).mean(),
                1,
            )
            desc_400 = (
                f"{startfart400}->{sluttfart400} på 400 (pulskost {hr_cost_400}). "
            )
        else:
            desc_400 = ""
        if sum(rows_200):
            startfart200 = int(data[rows_200]["time"].head(1).values[0])
            sluttfart200 = int(data[rows_200]["time"].tail(1).values[0])
            desc_200 = f"{startfart200}->{sluttfart200} på 200, "
        else:
            desc_200 = ""
        return {
            "title": "BFG Siljustøl" if sum(rows_400) else "BFG-lørdag",
            "desc": f"{desc_lang}{desc_400}{desc_200}6x60m.",
        }


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
        elif 180 < lap.distance_m < 214 and 28 < lap.total_time_s < 42:
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
                **lap_to_dict(lap, explicit_distance=2170),
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


async def describe():
    dirs = sorted(glob.glob(str(EXERCISE_DIR / "202*")))
    for _dir in dirs:
        desc = await make_description(Path(_dir))
        if desc:
            print(f"{desc['title']}:    {Path(_dir).name}\n\t{desc['desc']}")


async def analyze_all():
    dfs = []
    dirs = sorted(glob.glob(str(EXERCISE_DIR / "202*")))
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
    asyncio.run(describe())
