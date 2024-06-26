import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict

import dateutil  # type: ignore
import dotenv
import gpxpy
import isodate
import pandas as pd
import watchfiles
from geopy import distance

import pyrotun
import pyrotun.persist

dotenv.load_dotenv()

logger = pyrotun.getLogger(__name__)

MAP_SPORT_INFO = {
    "BACKCOUNTRY_SKIING": "Skitur",
    "CROSS-COUNTRY_SKIING": "Langrenn",
    "HIKING": "Fjelltur",
    "INDOOR_CYCLING": "Sykkelrulle",
    "RUNNING": "Løp",
    "TRAIL_RUNNING": "Løp, terreng",
    "TREADMILL_RUNNING": "Løp, mølle",
    "OTHER": "Styrke innendørs",
}

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
    ts = t_delta if isinstance(t_delta, (int, float)) else t_delta.total_seconds()
    hours, remainder = divmod(ts, 3600)
    minutes, _ = divmod(remainder, 60)
    return ("{}:{:02d}").format(int(hours), int(minutes))


assert timedeltaformatter(0) == "0:00"
assert timedeltaformatter(1) == "0:00"
assert timedeltaformatter(60) == "0:01"
assert timedeltaformatter(60 * 60) == "1:00"


def make_http_post_data(dirname: Path) -> Dict[str, str]:
    exercise_datetime = dateutil.parser.isoparse(dirname.name)
    exercise_summary = json.loads((Path(dirname) / "exercise_summary").read_text())
    heart_rate_zones = json.loads((Path(dirname) / "heart_rate_zones").read_text())
    zonedata_df = pd.DataFrame(heart_rate_zones["zone"])

    # Map 5 zones into an alternative categorization:
    threezone_map = {0: "lav", 1: "lav", 2: "moderat", 3: "hoy", 4: "hoy"}
    if zonedata_df.empty:
        print("Empty zonedata, assigning exercise duration to low")
        zonedata_df = pd.DataFrame(
            [
                {"index": 1, "in-zone": exercise_summary["duration"]},
                {"index": 2, "in-zone": "P0M"},
                {"index": 3, "in-zone": "P0M"},
            ]
        )
        print("Empty zonedata, putting everything to low")
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

    details = ""
    if "distance" in exercise_summary:
        distance = exercise_summary["distance"] / 1000
        details += f"{distance:.1f} km. "

    if "duration" in exercise_summary:
        duration = isodate.parse_duration(exercise_summary["duration"]).seconds
        if duration < 120:
            logger.warning("Skipping too short exercise %s", dirname)
            return
    if (
        "duration" in exercise_summary
        and "distance" in exercise_summary
        and gpxfile.is_file()
    ):
        details += f"{moving_speed} min/km. "
        details += f"Tid: {prettyprintseconds(move_time)}. "

    if "heart-rate" in exercise_summary:
        avg_beat = "-"
        max_beat = "-"
        if "average" in exercise_summary["heart-rate"]:
            avg_beat = exercise_summary["heart-rate"]["average"]
        if "maximum" in exercise_summary["heart-rate"]:
            max_beat = exercise_summary["heart-rate"]["maximum"]
        details += f"Puls {avg_beat} / {max_beat}."

    return {
        "legginn": "ja",
        "navn": os.getenv("EXERCISE_NAME"),
        "hva": MAP_SPORT_INFO.get(exercise_summary["detailed-sport-info"], "polar v"),
        "dato": exercise_datetime.strftime("%Y-%m-%d"),
        "hoyintensitet": threezones_df.loc["hoy"]["hh:mm"],
        "modintensitet": threezones_df.loc["moderat"]["hh:mm"],
        "lavintensitet": threezones_df.loc["lav"]["hh:mm"],
        "styrke": "0:00",
        "toying": "0:00",
        "detaljer": details,
    }


async def process_dir(dirname: Path, pers=None, dryrun=False, force=False):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname = dirname.parent
    try:
        dateutil.parser.isoparse(dirname.name)
    except ValueError:
        logger.warning(f"Skipping strange-looking directory {dirname}")
        return
    if (dirname / "exercise_summary").is_file() and (
        dirname / "heart_rate_zones"
    ).is_file():
        postdata = None
        if force or not (dirname / DONE_FILE).is_file():
            logger.info("making post data")
            postdata = make_http_post_data(dirname)
            logger.info("postdata is %s", str(postdata))
        if postdata is None:
            return

        if not dryrun and pers is not None:
            logger.info("Submitting exercise data %s", str(postdata))
            async with pers.websession.post(
                url=os.getenv("EXERCISE_URL"), params=postdata
            ) as response:
                logger.info(str(response))
                content = await response.content.read()
                logger.info(content)
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

    logger.info(f"Starting watching {EXERCISE_DIR} for exercises")
    async for changes in watchfiles.awatch(EXERCISE_DIR):
        # changes is set of tuples
        logger.info(f"Detected filesystem change: {changes}")
        dirnames = set([Path(change[1]) for change in changes])
        logger.info(f"Will process directories {dirnames}")
        # dirnames are timestamps
        for dirname in dirnames:
            logger.info(f"Processing dir {dirname}")
            await process_dir(dirname, pers, dryrun)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submit", action="store_true", help="If we should do real submits"
    )
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
        asyncio.run(process_dir(args.processdir, dryrun=not args.submit, force=True))
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(dryrun=not args.submit))
