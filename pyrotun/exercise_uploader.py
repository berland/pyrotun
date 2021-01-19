import os
from pathlib import Path
import dateutil
import asyncio
import json
import time

import isodate
import pandas as pd
import watchgod
import dotenv
import gpxpy
from geopy import distance
import argparse

import pyrotun
import pyrotun.persist

dotenv.load_dotenv()

logger = pyrotun.getLogger(__name__)

EXERCISE_DIR = Path.home() / "polar_dump"

DONE_FILE = "done"

URL = os.getenv("EXERCISE_URL")


def gpx2df(gpxfile):
    gpx = gpxpy.parse(Path(gpxfile).read_text())
    data = gpx.tracks[0].segments[0].points
    gpx_df = pd.DataFrame(
        [
            {
                "lat": point.latitude,
                "lon": point.longitude,
                "elev": point.elevation,
                "datetime": point.time,
            }
            for point in data
        ]
    )

    return gpx_df


def diffgpxdf(gpxdf):
    ddf = pd.concat(
        [
            gpxdf.rename(lambda x: x + "1", axis=1),
            gpxdf.shift(-1).rename(lambda x: x + "2", axis=1),
        ],
        axis=1,
    ).dropna()
    ddf["dist"] = ddf.apply(
        lambda row: distance.distance(
            (row["lat1"], row["lon1"]), (row["lat2"], row["lon2"])
        ).m,
        axis=1,
    )
    ddf["t_delta"] = (ddf["datetime2"] - ddf["datetime1"]).dt.total_seconds()
    ddf["moving"] = (ddf["dist"] / ddf["t_delta"]) > 0.8
    return ddf


def timedeltaformatter(t_delta):
    # Produce HH:MM from timedelta objects or numerical seconds
    # Later round to nearest 5 min.
    if not isinstance(t_delta, (int, float)):
        ts = t_delta.total_seconds()
    else:
        ts = t_delta
    hours, remainder = divmod(ts, 3600)
    minutes, seconds = divmod(remainder, 60)
    return ("{}:{:02d}").format(int(hours), int(minutes))


assert timedeltaformatter(0) == "0:00"
assert timedeltaformatter(1) == "0:00"
assert timedeltaformatter(60) == "0:01"
assert timedeltaformatter(60 * 60) == "1:00"


def make_http_post_data(dirname):
    exercise_datetime = dateutil.parser.isoparse(dirname.name)
    exercise_summary = json.loads((Path(dirname) / "exercise_summary").read_text())
    heart_rate_zones = json.loads((Path(dirname) / "heart_rate_zones").read_text())
    zonedata_df = pd.DataFrame(heart_rate_zones["zone"])

    # Map 5 zones into an alternative categorization:
    threezone_map = {0: "lav", 1: "lav", 2: "moderat", 3: "hoy", 4: "hoy"}
    zonedata_df["threezones"] = zonedata_df["index"].replace(threezone_map)

    # Compute times to seconds (from iso deltastrings)
    zonedata_df["in-zone"] = pd.to_timedelta(
        zonedata_df["in-zone"].map(isodate.parse_duration)
    ).dt.total_seconds()

    threezones_df = zonedata_df.groupby("threezones").sum()
    threezones_df["hh:mm"] = threezones_df["in-zone"].apply(timedeltaformatter)

    gpxfile = Path(dirname) / "gpx"
    if gpxfile.is_file():
        gpxdf = gpx2df(Path(dirname) / "gpx")
        ddf = diffgpxdf(gpxdf)
        dist = sum(ddf["dist"])
        move_time = sum(ddf[ddf.moving].t_delta)
        moving_speed = prettyprintseconds(move_time / (dist / 1000.0))

    map_sport_info = {
        "RUNNING": "Løp",
        "HIKING": "Fjelltur",
        "INDOOR_CYCLING": "Sykkelrulle",
        "BACKCOUNTRY_SKIING": "Skitur",
        "CROSS-COUNTRY_SKIING": "Langrenn",
    }

    details = ""
    if "distance" in exercise_summary:
        distance = exercise_summary["distance"] / 1000
        details += f"{distance:.1f} km. "

    if "duration" in exercise_summary:
        duration = isodate.parse_duration(exercise_summary["duration"]).seconds
        if duration < 120:
            logger.warning("Skipping too short exercise %s", dirname)
            return
    if "duration" in exercise_summary and "distance" in exercise_summary:
        if gpxfile.is_file():
            details += f"{moving_speed} min/km. "
            details += f"Tid: {prettyprintseconds(move_time)}. "

    if "heart-rate" in exercise_summary:
        avg_beat = exercise_summary["heart-rate"]["average"]
        max_beat = exercise_summary["heart-rate"]["maximum"]
        details += f"Puls {avg_beat} / {max_beat}."

    post_data = {
        "legginn": "ja",
        "navn": os.getenv("EXERCISE_NAME"),
        "hva": map_sport_info.get(exercise_summary["detailed-sport-info"], "polar v"),
        "dato": exercise_datetime.strftime("%Y-%m-%d"),
        "hoyintensitet": threezones_df.loc["hoy"]["hh:mm"],
        "modintensitet": threezones_df.loc["moderat"]["hh:mm"],
        "lavintensitet": threezones_df.loc["lav"]["hh:mm"],
        "styrke": "0:00",
        "toying": "0:00",
        "detaljer": details,
    }

    return post_data


async def process_dir(dirname, pers=None, dryrun=False, force=False):
    dirname = Path(dirname)
    logger.info("in process_dir()")
    try:
        dateutil.parser.isoparse(dirname.name)
    except ValueError:
        logger.warning("Skipping strange-looking directory %s", str(dirname))
        return
    if (dirname / "exercise_summary").is_file() and (
        dirname / "heart_rate_zones"
    ).is_file():
        logger.info("passed initial test")
        postdata = None
        if force or not (dirname / DONE_FILE).is_file():
            logger.info("making post data")
            postdata = make_http_post_data(dirname)
            logger.info("postdata is %s", str(postdata))
        if postdata is None:
            return

        if not dryrun and pers is not None:
            logger.info("Submitting exercise data %s", str(postdata))
            await pers.websession.post(url=os.getenv("EXERCISE_URL"), data=postdata)
            Path(dirname / DONE_FILE).touch()
        else:
            logger.info("DRY-run, would have submitted %s", str(postdata))


def prettyprintseconds(seconds):
    seconds = round(seconds)
    if seconds < 60 * 60:
        return time.strftime("%-M:%S", time.gmtime(seconds))
    return time.strftime("%-H:%M:%S", time.gmtime(seconds))


assert prettyprintseconds(0) == "0:00"
assert prettyprintseconds(60) == "1:00"
assert prettyprintseconds(61) == "1:01"


async def main(pers=None, dryrun=False):
    if pers is None:
        dotenv.load_dotenv()
        pers = pyrotun.persist.PyrotunPersistence()
        await pers.ainit(["websession"])
    assert pers.websession is not None

    logger.info("Starting watching %s for exercises", EXERCISE_DIR)
    async for changes in watchgod.awatch(EXERCISE_DIR):
        logger.info("Detected filesystem change: %s", str(changes))
        dirnames = set([Path(change[1]).parent for change in changes])
        logger.info("Will process directories %s", str(dirnames))
        # dirnames are timestamps
        for dirname in dirnames:
            logger.info("Processing dir %s", str(dirname))
            await process_dir(dirname, pers, dryrun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", type=str, help="GPX file to analyze")
    parser.add_argument("--processdir", type=str, help="Dry run a directory")
    args = parser.parse_args()

    if args.analyze:
        gpxdf = gpx2df(args.analyze)
        ddf = diffgpxdf(gpxdf)
        print(ddf.head())
        dist = sum(ddf["dist"])
        print(f"Distance {dist}")
        move_time = sum(ddf[ddf.moving].t_delta)
        print(f"Moving time: {prettyprintseconds(move_time)}")
        speed = prettyprintseconds(move_time / (dist / 1000.0))
        print(f"Avg speed {speed}")
    if args.processdir:
        asyncio.run(process_dir(args.processdir, dryrun=True, force=True))
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(dryrun=True))
