#!/usr/bin/env pythonn3
import argparse
import asyncio
import datetime as dt
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import dotenv
import pandas as pd

BASE_URL = "https://www.polaraccesslink.com/v3"

logger = logging.getLogger(__name__)


def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def flatten(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "date": rec.get("date"),
        "heart_rate_avg": rec.get("heart_rate_avg"),
        "heart_rate_variability_avg": rec.get("heart_rate_variability_avg"),
        "breathing_rate_avg": rec.get("breathing_rate_avg"),
        "beat_to_beat_avg": rec.get("beat_to_beat_avg"),
        "ans_charge": rec.get("ans_charge"),
        "ans_status": rec.get("ans_status"),
        "sleep_charge": rec.get("sleep_charge"),
        "sleep_status": rec.get("sleep_status"),
        "nightly_recharge": rec.get("nightly_recharge"),
        "nightly_recharge_status": rec.get("nightly_recharge_status"),
        "polar_user": rec.get("polar_user"),
    }


async def get_json(
    session: aiohttp.ClientSession,
    path: str,
) -> Optional[Dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    timeout = aiohttp.ClientTimeout(total=30)

    async with session.get(url, timeout=timeout) as resp:
        if resp.status in (204, 404):
            return None
        if resp.status == 401:
            raise RuntimeError(f"401 Unauthorized for {url}")
        if resp.status == 403:
            raise RuntimeError(f"403 Forbidden for {url}")
        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status} for {url}: {text}")

        text = await resp.text()
        if not text.strip():
            return None

        payload = await resp.json()
        return payload if isinstance(payload, dict) else None


async def fetch_recent_rows(
    token: str,
    days_back: int = 7,
    sleep_s: float = 0.1,
) -> List[Dict[str, Any]]:
    today = dt.date.today()
    end = today - dt.timedelta(days=1)
    start = end - dt.timedelta(days=days_back - 1)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    rows: List[Dict[str, Any]] = []

    async with aiohttp.ClientSession(headers=headers) as session:
        for d in daterange(start, end):
            ds = d.isoformat()
            logger.info(f"Fetching nightly data for {d} from polar")
            payload = await get_json(session, f"/users/nightly-recharge/{ds}")
            if payload and payload.get("date"):
                rows.append(flatten(payload))
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

    return rows


def empty_base_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "heart_rate_avg",
            "heart_rate_variability_avg",
            "breathing_rate_avg",
            "beat_to_beat_avg",
            "ans_charge",
            "ans_status",
            "sleep_charge",
            "sleep_status",
            "nightly_recharge",
            "nightly_recharge_status",
            "polar_user",
        ]
    )


def read_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_base_df()
    return pd.read_csv(path)


def add_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" not in df.columns:
        raise ValueError("Input dataframe is missing 'date' column")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    numeric_cols = [
        "heart_rate_avg",
        "heart_rate_variability_avg",
        "breathing_rate_avg",
        "beat_to_beat_avg",
        "ans_charge",
        "sleep_charge",
        "nightly_recharge",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    # Unix epoch seconds at midnight
    df["date_epoch"] = (df["date"].astype("int64") // 10**9).astype("int64")

    # Rolling means
    df["hr_7d"] = df["heart_rate_avg"].rolling(7, min_periods=4).mean()
    df["hrv_7d"] = df["heart_rate_variability_avg"].rolling(7, min_periods=4).mean()

    df["hr_14d"] = df["heart_rate_avg"].rolling(14, min_periods=7).mean()
    df["hrv_14d"] = df["heart_rate_variability_avg"].rolling(14, min_periods=7).mean()

    df["hr_21d"] = df["heart_rate_avg"].rolling(21, min_periods=10).mean()
    df["hrv_21d"] = df["heart_rate_variability_avg"].rolling(21, min_periods=10).mean()

    df["hr_28d"] = df["heart_rate_avg"].rolling(28, min_periods=14).mean()
    df["hrv_28d"] = df["heart_rate_variability_avg"].rolling(28, min_periods=14).mean()

    df["hr_42d"] = df["heart_rate_avg"].rolling(42, min_periods=21).mean()
    df["hrv_42d"] = df["heart_rate_variability_avg"].rolling(42, min_periods=21).mean()

    # Rolling std
    df["hr_42d_std"] = df["heart_rate_avg"].rolling(42, min_periods=21).std()
    df["hrv_42d_std"] = (
        df["heart_rate_variability_avg"].rolling(42, min_periods=21).std()
    )

    # Deviations
    df["hr_delta_42d"] = df["heart_rate_avg"] - df["hr_42d"]
    df["hrv_delta_42d"] = df["heart_rate_variability_avg"] - df["hrv_42d"]

    # Relative values
    df["hr_pct_of_42d"] = 100.0 * df["heart_rate_avg"] / df["hr_42d"]
    df["hrv_pct_of_42d"] = 100.0 * df["heart_rate_variability_avg"] / df["hrv_42d"]

    # Z-scores
    df["hr_z_42d"] = df["hr_delta_42d"] / df["hr_42d_std"]
    df["hrv_z_42d"] = df["hrv_delta_42d"] / df["hrv_42d_std"]

    # Combined strain
    df["strain_score"] = df["hr_z_42d"] - df["hrv_z_42d"]

    # Rolling correlation
    df["hr_hrv_corr_30d"] = (
        df["heart_rate_avg"]
        .rolling(30, min_periods=15)
        .corr(df["heart_rate_variability_avg"])
    )

    # Flags
    df["flag_hr_high"] = df["hr_z_42d"] >= 1.0
    df["flag_hrv_low"] = df["hrv_z_42d"] <= -1.0
    df["flag_both_bad"] = df["flag_hr_high"] & df["flag_hrv_low"]

    bad = df["flag_both_bad"].fillna(False).to_numpy()
    streak = []
    s = 0
    for x in bad:
        if x:
            s += 1
        else:
            s = 0
        streak.append(s)
    df["bad_streak"] = streak

    def readiness_label(row):
        hrz = row.get("hr_z_42d")
        hrvz = row.get("hrv_z_42d")
        if pd.isna(hrz) or pd.isna(hrvz):
            return "unknown"
        if hrz <= 0.3 and hrvz >= -0.3:
            return "good"
        if hrz >= 1.5 and hrvz <= -1.5:
            return "very_poor"
        if hrz >= 1.0 and hrvz <= -1.0:
            return "poor"
        return "normal"

    df["readiness_label"] = df.apply(readiness_label, axis=1)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def write_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def load_events_csv(events_csv: Optional[Path]) -> pd.DataFrame:
    if not events_csv or not events_csv.exists():
        return pd.DataFrame(
            columns=["date", "date_epoch", "type", "label", "notes", "priority"]
        )

    df = pd.read_csv(events_csv)
    if "date" not in df.columns:
        raise ValueError("events.csv must contain a 'date' column")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "type" not in df.columns:
        df["type"] = "event"
    if "label" not in df.columns:
        df["label"] = df["type"]
    if "notes" not in df.columns:
        df["notes"] = ""
    if "priority" not in df.columns:
        df["priority"] = 2

    df["date_epoch"] = (df["date"].astype("int64") // 10**9).astype("int64")
    df = (
        df.sort_values(["date", "type", "label"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    cols = ["date", "date_epoch", "type", "label", "notes", "priority"]
    return df[cols]


def load_periods_csv(periods_csv: Optional[Path]) -> pd.DataFrame:
    if not periods_csv or not periods_csv.exists():
        return pd.DataFrame(
            columns=[
                "start_date",
                "start_date_epoch",
                "end_date",
                "end_date_epoch",
                "type",
                "label",
                "notes",
                "color",
            ]
        )

    df = pd.read_csv(periods_csv)
    if "start_date" not in df.columns or "end_date" not in df.columns:
        raise ValueError("periods.csv must contain 'start_date' and 'end_date' columns")

    df["start_date"] = pd.to_datetime(df["start_date"]).dt.normalize()
    df["end_date"] = pd.to_datetime(df["end_date"]).dt.normalize()

    if "type" not in df.columns:
        df["type"] = "period"
    if "label" not in df.columns:
        df["label"] = df["type"]
    if "notes" not in df.columns:
        df["notes"] = ""
    if "color" not in df.columns:
        df["color"] = "#dfe7f2"

    df["start_date_epoch"] = (df["start_date"].astype("int64") // 10**9).astype("int64")
    df["end_date_epoch"] = (df["end_date"].astype("int64") // 10**9).astype("int64")

    df = (
        df.sort_values(["start_date", "end_date", "type", "label"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    df["start_date"] = df["start_date"].dt.strftime("%Y-%m-%d")
    df["end_date"] = df["end_date"].dt.strftime("%Y-%m-%d")

    cols = [
        "start_date",
        "start_date_epoch",
        "end_date",
        "end_date_epoch",
        "type",
        "label",
        "notes",
        "color",
    ]
    return df[cols]


def write_sqlite_tables(
    nightly_df: pd.DataFrame,
    sqlite_path: Path,
    table_name: str,
    events_df: Optional[pd.DataFrame] = None,
    periods_df: Optional[pd.DataFrame] = None,
) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(sqlite_path)
    try:
        nightly_df.to_sql(table_name, conn, if_exists="replace", index=False)

        cur = conn.cursor()
        cur.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON "{table_name}" (date)'
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date_epoch "
            f'ON "{table_name}" (date_epoch)'
        )

        if events_df is not None:
            events_df.to_sql("events", conn, if_exists="replace", index=False)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events (date)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_date_epoch "
                "ON events (date_epoch)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events (type)")

        if periods_df is not None:
            periods_df.to_sql("periods", conn, if_exists="replace", index=False)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_periods_start_date "
                "ON periods (start_date)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_periods_end_date ON periods (end_date)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_periods_type ON periods (type)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_periods_start_date_epoch "
                "ON periods (start_date_epoch)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_periods_end_date_epoch "
                "ON periods (end_date_epoch)"
            )

        conn.commit()
    finally:
        conn.close()


async def update_polar_nightly_store(
    token: str,
    csv_path: str | Path,
    sqlite_path: str | Path,
    table_name: str = "polar_nightly",
    days_back: int = 7,
    sleep_s: float = 0.1,
    events_csv: Optional[str | Path] = None,
    periods_csv: Optional[str | Path] = None,
) -> Dict[str, Any]:
    csv_path = Path(csv_path)
    sqlite_path = Path(sqlite_path)
    events_csv_path = Path(events_csv) if events_csv else None
    periods_csv_path = Path(periods_csv) if periods_csv else None
    recent_rows = await fetch_recent_rows(
        token=token,
        days_back=days_back,
        sleep_s=sleep_s,
    )
    recent_df = pd.DataFrame(recent_rows)

    existing_df = read_existing_csv(csv_path)

    frames = [df for df in (existing_df, recent_df) if not df.empty]
    if not frames:
        raise RuntimeError("No recent Polar data fetched and no existing CSV found.")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if "date" not in combined.columns:
        raise RuntimeError("Combined dataset is missing 'date' column.")

    enriched = add_enrichment(combined)

    write_csv(enriched, csv_path)

    events_df = load_events_csv(events_csv_path)
    periods_df = load_periods_csv(periods_csv_path)

    write_sqlite_tables(
        nightly_df=enriched,
        sqlite_path=sqlite_path,
        table_name=table_name,
        events_df=events_df,
        periods_df=periods_df,
    )

    return {
        "recent_rows_fetched": len(recent_df),
        "total_rows_written": len(enriched),
        "csv_path": str(csv_path),
        "sqlite_path": str(sqlite_path),
        "table_name": table_name,
        "events_rows_written": len(events_df),
        "periods_rows_written": len(periods_df),
    }


async def async_main() -> int:
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description="Async Polar nightly updater: fetch recent recharge data, "
        "update CSV, and mirror nightly/events/periods into SQLite."
    )
    polar_files = Path("/home/berland/polar_dump")
    parser.add_argument(
        "--token", default=os.getenv("POLAR_ACCESS_TOKEN"), help="Polar access token"
    )
    parser.add_argument(
        "--csv",
        default=polar_files / "polar_nightly.csv",
        help="Path to master nightly CSV",
    )
    parser.add_argument(
        "--sqlite",
        default=polar_files / "polar_nightly.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--table", default="polar_nightly", help="SQLite table name for nightly data"
    )
    parser.add_argument(
        "--days-back", type=int, default=7, help="How many recent days to refetch"
    )
    parser.add_argument(
        "--sleep", type=float, default=0.1, help="Delay between API calls in seconds"
    )
    parser.add_argument(
        "--events", default=polar_files / "events.csv", help="Optional events.csv"
    )
    parser.add_argument(
        "--periods", default=polar_files / "periods.csv", help="Optional periods.csv"
    )
    args = parser.parse_args()

    result = await update_polar_nightly_store(
        token=args.token,
        csv_path=args.csv,
        sqlite_path=args.sqlite,
        table_name=args.table,
        days_back=args.days_back,
        sleep_s=args.sleep,
        events_csv=args.events,
        periods_csv=args.periods,
    )

    print("Update complete")
    for k, v in result.items():
        print(f"{k}: {v}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
