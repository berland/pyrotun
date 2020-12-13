import os
from pathlib import Path
from dateutil import parser
import asyncio
import json
import time

import isodate
import pandas as pd
import watchgod
import dotenv

import pyrotun
import pyrotun.persist

dotenv.load_dotenv()

logger = pyrotun.getLogger(__name__)

EXERCISE_DIR = Path.home() / "polar_dump"

DONE_FILE = "done"

URL = os.getenv("EXERCISE_URL")


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
    exercise_datetime = parser.isoparse(dirname.name)
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

    map_sport_info = {"RUNNING": "Løp"}

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
        speed_sec_pr_km = duration / distance
        speed_min_pr_km = prettyprintseconds(speed_sec_pr_km)
        details += f"{speed_min_pr_km} min/km. "


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


async def process_dir(dirname, pers, dryrun=False):
    dirname = Path(dirname)
    try:
        parser.isoparse(dirname.name)
    except ValueError:
        logger.warning("Skipping strange-looking directory %s", str(dirname))
        return
    if (
        (dirname / "exercise_summary").is_file()
        and (dirname / "heart_rate_zones").is_file()
        and not (dirname / DONE_FILE).is_file()
    ):
        postdata = make_http_post_data(dirname)

        if postdata is None:
            # Warning already emitted.
            return

        if not dryrun:
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
        dirnames = set([Path(change[1]).parent for change in changes])
        # dirnames are timestamps
        for dirname in dirnames:
            await process_dir(dirname, pers, dryrun)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(dryrun=True))
