#!/usr/bin/env python3
import argparse
import asyncio
import datetime as dt
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
import pandas as pd
from connections.polar_token_manager import PolarTokenManager

BASE_URL = "https://www.polaraccesslink.com/v4/data"

logger = logging.getLogger(__name__)


def flatten(rec: Dict[str, Any]) -> Dict[str, Any]:
    mean_rri = rec.get("meanNightlyRecoveryRri")
    mean_rmssd = rec.get("meanNightlyRecoveryRmssd")
    mean_resp_interval = rec.get("meanNightlyRecoveryRespirationInterval")

    try:
        mean_rri_num = float(mean_rri) if mean_rri is not None else None
    except (TypeError, ValueError):
        mean_rri_num = None

    heart_rate_avg = None
    if mean_rri_num and mean_rri_num > 0:
        heart_rate_avg = 60000.0 / mean_rri_num

    return {
        "date": rec.get("sleepResultDate"),
        # backward-compatible legacy columns
        "heart_rate_avg": heart_rate_avg,
        "heart_rate_variability_avg": mean_rmssd,
        "breathing_rate_avg": None,
        "beat_to_beat_avg": mean_rri_num,
        "ans_charge": None,
        "ans_status": rec.get("ansStatus"),
        "sleep_charge": None,
        "sleep_status": None,
        "nightly_recharge": rec.get("ansRate"),
        "nightly_recharge_status": rec.get("recoveryIndicator"),
        "polar_user": None,
        # v4-native columns
        "ans_rate": rec.get("ansRate"),
        "recovery_indicator": rec.get("recoveryIndicator"),
        "recovery_indicator_sub_level": rec.get("recoveryIndicatorSubLevel"),
        "mean_nightly_recovery_rri": mean_rri_num,
        "mean_nightly_recovery_rmssd": mean_rmssd,
        "mean_nightly_recovery_respiration_interval": mean_resp_interval,
        "baseline_mean_rri": rec.get("baselineMeanRri"),
        "baseline_sd_rri": rec.get("baselineSdRri"),
        "baseline_mean_rmssd": rec.get("baselineMeanRmssd"),
        "baseline_sd_rmssd": rec.get("baselineSdRmssd"),
        "baseline_mean_respiration_interval": rec.get(
            "baselineMeanRespirationInterval"
        ),
        "baseline_sd_respiration_interval": rec.get("baselineSdRespirationInterval"),
    }


async def fetch_rows_for_range(
    manager: PolarTokenManager,
    start: dt.date,
    end: dt.date,
) -> List[Dict[str, Any]]:
    if end < start:
        return []

    logger.info("Fetching v4 nightly recharge data from %s to %s", start, end)

    payload = await manager.authorized_get(
        f"{BASE_URL}/nightly-recharge-results",
        params={
            "from": start.isoformat(),
            "to": end.isoformat(),
        },
        accept="application/json",
    )

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("nightlyRechargeResults"), list):
            records = payload["nightlyRechargeResults"]
        elif isinstance(payload.get("data"), list):
            records = payload["data"]
        else:
            records = [payload]
    else:
        records = []

    return [flatten(rec) for rec in records if rec.get("sleepResultDate")]


async def fetch_historical_rows_chunked(
    manager: PolarTokenManager,
    start: dt.date,
    end: dt.date,
    chunk_days: int = 28,
    sleep_s: float = 0.2,
) -> List[Dict[str, Any]]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    if end < start:
        return []

    all_rows: List[Dict[str, Any]] = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(chunk_start + dt.timedelta(days=chunk_days - 1), end)

        rows = await fetch_rows_for_range(
            manager=manager,
            start=chunk_start,
            end=chunk_end,
        )
        logger.info(
            "Fetched %s rows for chunk %s -> %s",
            len(rows),
            chunk_start,
            chunk_end,
        )
        all_rows.extend(rows)

        chunk_start = chunk_end + dt.timedelta(days=1)
        if chunk_start <= end and sleep_s > 0:
            await asyncio.sleep(sleep_s)

    return all_rows


def empty_base_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ans_status",
            "ans_rate",
            "recovery_indicator",
            "recovery_indicator_sub_level",
            "mean_nightly_recovery_rri",
            "mean_nightly_recovery_rmssd",
            "mean_nightly_recovery_respiration_interval",
            "baseline_mean_rri",
            "baseline_sd_rri",
            "baseline_mean_rmssd",
            "baseline_sd_rmssd",
            "baseline_mean_respiration_interval",
            "baseline_sd_respiration_interval",
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
        "ans_rate",
        "mean_nightly_recovery_rri",
        "mean_nightly_recovery_rmssd",
        "mean_nightly_recovery_respiration_interval",
        "baseline_mean_rri",
        "baseline_sd_rri",
        "baseline_mean_rmssd",
        "baseline_sd_rmssd",
        "baseline_mean_respiration_interval",
        "baseline_sd_respiration_interval",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    df["date_epoch"] = (df["date"].astype("int64") // 10**9).astype("int64")

    # Rolling means for v4 core signals
    df["rri_7d"] = df["mean_nightly_recovery_rri"].rolling(7, min_periods=4).mean()
    df["rmssd_7d"] = df["mean_nightly_recovery_rmssd"].rolling(7, min_periods=4).mean()
    df["resp_7d"] = (
        df["mean_nightly_recovery_respiration_interval"]
        .rolling(7, min_periods=4)
        .mean()
    )

    df["rri_42d"] = df["mean_nightly_recovery_rri"].rolling(42, min_periods=21).mean()
    df["rmssd_42d"] = (
        df["mean_nightly_recovery_rmssd"].rolling(42, min_periods=21).mean()
    )
    df["resp_42d"] = (
        df["mean_nightly_recovery_respiration_interval"]
        .rolling(42, min_periods=21)
        .mean()
    )

    df["rri_42d_std"] = (
        df["mean_nightly_recovery_rri"].rolling(42, min_periods=21).std()
    )
    df["rmssd_42d_std"] = (
        df["mean_nightly_recovery_rmssd"].rolling(42, min_periods=21).std()
    )
    df["resp_42d_std"] = (
        df["mean_nightly_recovery_respiration_interval"]
        .rolling(42, min_periods=21)
        .std()
    )

    df["rri_delta_42d"] = df["mean_nightly_recovery_rri"] - df["rri_42d"]
    df["rmssd_delta_42d"] = df["mean_nightly_recovery_rmssd"] - df["rmssd_42d"]
    df["resp_delta_42d"] = (
        df["mean_nightly_recovery_respiration_interval"] - df["resp_42d"]
    )

    df["rri_z_42d"] = df["rri_delta_42d"] / df["rri_42d_std"]
    df["rmssd_z_42d"] = df["rmssd_delta_42d"] / df["rmssd_42d_std"]
    df["resp_z_42d"] = df["resp_delta_42d"] / df["resp_42d_std"]

    df["flag_rmssd_low"] = df["rmssd_z_42d"] <= -1.0
    df["flag_resp_high"] = df["resp_z_42d"] >= 1.0
    df["flag_recovery_poor"] = df["flag_rmssd_low"] & df["flag_resp_high"]

    def readiness_label(row):
        rmssdz = row.get("rmssd_z_42d")
        respz = row.get("resp_z_42d")
        if pd.isna(rmssdz) or pd.isna(respz):
            return "unknown"
        if rmssdz >= -0.3 and respz <= 0.3:
            return "good"
        if rmssdz <= -1.5 and respz >= 1.5:
            return "very_poor"
        if rmssdz <= -1.0 and respz >= 1.0:
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
    manager: PolarTokenManager,
    csv_path: str | Path,
    sqlite_path: str | Path,
    table_name: str = "polar_nightly",
    days_back: int = 7,
    events_csv: Optional[str | Path] = None,
    periods_csv: Optional[str | Path] = None,
    start_date: Optional[dt.date] = None,
    end_date: Optional[dt.date] = None,
    chunk_days: int = 28,
    sleep_s: float = 0.2,
) -> Dict[str, Any]:
    csv_path = Path(csv_path)
    sqlite_path = Path(sqlite_path)
    events_csv_path = Path(events_csv) if events_csv else None
    periods_csv_path = Path(periods_csv) if periods_csv else None

    if start_date is not None or end_date is not None:
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date must be provided together")
        recent_rows = await fetch_historical_rows_chunked(
            manager=manager,
            start=start_date,
            end=end_date,
            chunk_days=chunk_days,
            sleep_s=sleep_s,
        )
    else:
        today = dt.date.today()
        recent_end = today - dt.timedelta(days=1)
        recent_start = recent_end - dt.timedelta(days=days_back - 1)
        recent_rows = await fetch_historical_rows_chunked(
            manager=manager,
            start=recent_start,
            end=recent_end,
            chunk_days=chunk_days,
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
        "start_date": str(start_date) if start_date else None,
        "end_date": str(end_date) if end_date else None,
        "chunk_days": chunk_days,
    }


async def async_main() -> int:
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description="Async Polar nightly updater: fetch recent v4 nightly recharge data, "
        "update CSV, and mirror nightly/events/periods into SQLite."
    )
    polar_files = Path("/home/berland/polar_dump")

    parser.add_argument(
        "--client-id",
        default=os.getenv("POLAR_V4_CLIENT_ID"),
        help="Polar OAuth client id",
    )
    parser.add_argument(
        "--client-secret",
        default=os.getenv("POLAR_V4_CLIENT_SECRET"),
        help="Polar OAuth client secret",
    )
    parser.add_argument(
        "--token-file",
        default=os.getenv("POLAR_TOKEN_FILE", str(polar_files / "polar_tokens.json")),
        help="Path to token JSON file",
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
        "--events", default=polar_files / "events.csv", help="Optional events.csv"
    )
    parser.add_argument(
        "--periods", default=polar_files / "periods.csv", help="Optional periods.csv"
    )
    parser.add_argument(
        "--start-date",
        help="Historical backfill start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        help="Historical backfill end date YYYY-MM-DD",
    )
    parser.add_argument(
        "--bootstrap-days",
        type=int,
        help="Convenience option: fetch from N days ago through yesterday",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=28,
        help="Maximum days per API request window",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between API chunk requests in seconds",
    )
    args = parser.parse_args()

    if not args.client_id or not args.client_secret:
        raise RuntimeError(
            "Missing Polar credentials. Set POLAR_CLIENT_ID and POLAR_CLIENT_SECRET."
        )

    start_date = None
    end_date = None

    if args.bootstrap_days is not None:
        end_date = dt.date.today() - dt.timedelta(days=1)
        start_date = end_date - dt.timedelta(days=args.bootstrap_days - 1)
    elif args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            raise RuntimeError("--start-date and --end-date must be used together")
        start_date = dt.date.fromisoformat(args.start_date)
        end_date = dt.date.fromisoformat(args.end_date)

    manager = PolarTokenManager(
        client_id=args.client_id,
        client_secret=args.client_secret,
        token_file=args.token_file,
    )

    result = await update_polar_nightly_store(
        manager=manager,
        csv_path=args.csv,
        sqlite_path=args.sqlite,
        table_name=args.table,
        days_back=args.days_back,
        events_csv=args.events,
        periods_csv=args.periods,
        start_date=start_date,
        end_date=end_date,
        chunk_days=args.chunk_days,
        sleep_s=args.sleep,
    )
    print("Update complete")
    for k, v in result.items():
        print(f"{k}: {v}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
