#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import dotenv
import pandas as pd


def to_epoch_ms(series: pd.Series) -> pd.Series:
    return (pd.to_datetime(series).astype("int64") // 10**6).astype("int64")


def normalize_scope_value(value):
    if value in ("", None):
        return None
    return value


def normalize_tags(tags) -> Tuple[str, ...]:
    if not tags:
        return tuple()
    if isinstance(tags, list):
        return tuple(sorted(str(t) for t in tags))
    return (str(tags),)


def make_text(label: str, notes: str) -> str:
    label = str(label)
    notes = str(notes or "").strip()
    return f"{label} — {notes}" if notes else label


def load_events(events_csv: Path) -> pd.DataFrame:
    if not events_csv.exists():
        return pd.DataFrame(columns=["date", "type", "label", "notes", "time"])

    df = pd.read_csv(events_csv)
    if "date" not in df.columns:
        raise ValueError("events.csv must contain 'date'")
    if "type" not in df.columns:
        df["type"] = "event"
    if "label" not in df.columns:
        df["label"] = df["type"]
    if "notes" not in df.columns:
        df["notes"] = ""

    df["time"] = to_epoch_ms(df["date"])
    return df


def load_periods(periods_csv: Path) -> pd.DataFrame:
    if not periods_csv.exists():
        return pd.DataFrame(
            columns=[
                "start_date",
                "end_date",
                "type",
                "label",
                "notes",
                "time",
                "timeEnd",
            ]
        )

    df = pd.read_csv(periods_csv)
    print(df)
    if "start_date" not in df.columns or "end_date" not in df.columns:
        raise ValueError("periods.csv must contain 'start_date' and 'end_date'")
    if "type" not in df.columns:
        df["type"] = "period"
    if "label" not in df.columns:
        df["label"] = df["type"]
    if "notes" not in df.columns:
        df["notes"] = ""

    df["time"] = to_epoch_ms(df["start_date"])
    df["timeEnd"] = to_epoch_ms(df["end_date"])
    return df


def make_event_payload(row, dashboard_uid=None, panel_id=None) -> dict:
    payload = {
        "time": int(row["time"]),
        "timeEnd": int(row["time"]),
        "text": make_text(row["label"], row.get("notes", "")),
        "tags": [str(row["type"])],
    }
    if dashboard_uid:
        payload["dashboardUID"] = dashboard_uid
    if panel_id is not None:
        payload["panelId"] = panel_id
    return payload


def make_period_payload(row, dashboard_uid=None, panel_id=None) -> dict:
    payload = {
        "time": int(row["time"]),
        "timeEnd": int(row["timeEnd"]),
        "text": make_text(row["label"], row.get("notes", "")),
        "tags": [str(row["type"])],
    }
    if dashboard_uid:
        payload["dashboardUID"] = dashboard_uid
    if panel_id is not None:
        payload["panelId"] = panel_id
    return payload


def annotation_key(ann: dict) -> tuple:
    dashboard_uid = normalize_scope_value(
        ann.get("dashboardUID") or ann.get("dashboardUid")
    )
    panel_id = normalize_scope_value(ann.get("panelId"))

    time_val = int(ann.get("time")) if ann.get("time") is not None else None
    time_end_val = (
        int(ann.get("timeEnd")) if ann.get("timeEnd") is not None else time_val
    )

    return (
        dashboard_uid,
        panel_id,
        time_val,
        time_end_val,
        ann.get("text", ""),
        normalize_tags(ann.get("tags")),
    )


def logical_key_without_text(ann: dict) -> tuple:
    dashboard_uid = normalize_scope_value(
        ann.get("dashboardUID") or ann.get("dashboardUid")
    )
    panel_id = normalize_scope_value(ann.get("panelId"))

    time_val = int(ann.get("time")) if ann.get("time") is not None else None
    time_end_val = (
        int(ann.get("timeEnd")) if ann.get("timeEnd") is not None else time_val
    )

    return (
        dashboard_uid,
        panel_id,
        time_val,
        time_end_val,
        normalize_tags(ann.get("tags")),
    )


async def fetch_existing_annotations(
    session: aiohttp.ClientSession,
    grafana_url: str,
    from_ms: int,
    to_ms: int,
    dashboard_uid: Optional[str] = None,
    panel_id: Optional[int] = None,
    limit: int = 1000,
) -> List[dict]:
    url = grafana_url.rstrip("/") + "/api/annotations"
    params = {
        "from": from_ms,
        "to": to_ms,
        "limit": limit,
    }
    if dashboard_uid:
        params["dashboardUID"] = dashboard_uid
    if panel_id is not None:
        params["panelId"] = panel_id

    async with session.get(url, params=params) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"GET {url} failed: HTTP {resp.status}: {text}")
        return json.loads(text)


async def create_annotation(
    session: aiohttp.ClientSession,
    grafana_url: str,
    payload: dict,
) -> dict:
    url = grafana_url.rstrip("/") + "/api/annotations"
    async with session.post(url, json=payload) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"POST {url} failed: HTTP {resp.status}: {text}")
        return json.loads(text)


async def update_annotation(
    session: aiohttp.ClientSession,
    grafana_url: str,
    ann_id: int,
    payload: dict,
) -> dict:
    url = grafana_url.rstrip("/") + f"/api/annotations/{ann_id}"
    async with session.put(url, json=payload) as resp:
        text = await resp.text()
        if resp.status >= 400:
            raise RuntimeError(f"PUT {url} failed: HTTP {resp.status}: {text}")
        return json.loads(text)


def build_existing_indexes(existing_annotations: List[dict]):
    exact = {}
    logical = {}
    for ann in existing_annotations:
        exact[annotation_key(ann)] = ann
        logical.setdefault(logical_key_without_text(ann), []).append(ann)
    return exact, logical


async def sync_payloads(
    session: aiohttp.ClientSession,
    grafana_url: str,
    payloads: List[dict],
    existing_annotations: List[dict],
    dry_run: bool = False,
) -> Dict[str, int]:
    exact_idx, logical_idx = build_existing_indexes(existing_annotations)

    created = 0
    updated = 0
    skipped = 0

    for payload in payloads:
        exact_match = exact_idx.get(annotation_key(payload))
        if exact_match:
            skipped += 1
            print(f"skip   : {payload['text']}", flush=True)
            continue

        candidates = logical_idx.get(logical_key_without_text(payload), [])
        if candidates:
            candidate = candidates[0]
            ann_id = candidate["id"]
            if dry_run:
                print(f"update : {payload['text']} (id={ann_id})", flush=True)
            else:
                await update_annotation(session, grafana_url, ann_id, payload)
                print(f"update : {payload['text']} (id={ann_id})", flush=True)
            updated += 1

            candidate_updated = dict(candidate)
            candidate_updated.update(payload)
            exact_idx[annotation_key(candidate_updated)] = candidate_updated
            continue

        if dry_run:
            print(f"create : {payload['text']}", flush=True)
        else:
            await create_annotation(session, grafana_url, payload)
            print(f"create : {payload['text']}", flush=True)
        created += 1

    return {
        "created": created,
        "updated": updated,
        "skipped": skipped,
    }


def build_grafana_session_kwargs(
    token: Optional[str] = None,
    basic_user: Optional[str] = None,
    basic_password: Optional[str] = None,
    x_grafana_token: Optional[str] = None,
) -> dict:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    auth = None

    if basic_user is None:
        headers["Authorization"] = f"Bearer {token}"
    else:
        headers["X-Grafana-Token"] = token

    if basic_user is not None:
        auth = aiohttp.BasicAuth(basic_user, basic_password or "", encoding="utf-8")

    return {
        "headers": headers,
        "auth": auth,
    }


async def sync_grafana_annotations(
    grafana_url: str,
    token: str,
    basic_user: str | None = None,
    basic_password: str | None = None,
    events_csv: str | Path | None = None,
    periods_csv: str | Path | None = None,
    dashboard_uid: str | None = None,
    panel_id: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    payloads: list[dict] = []

    if events_csv:
        print("loading events")
        events_df = load_events(Path(events_csv))
        for _, row in events_df.iterrows():
            payloads.append(make_event_payload(row, dashboard_uid, panel_id))

    if periods_csv:
        print("loading periods")
        periods_df = load_periods(Path(periods_csv))
        for _, row in periods_df.iterrows():
            print(row)
            payloads.append(make_period_payload(row, dashboard_uid, panel_id))

    if not payloads:
        return {
            "created": 0,
            "updated": 0,
            "skipped": 0,
        }

    min_time = min(p["time"] for p in payloads)
    max_time = max(p.get("timeEnd", p["time"]) for p in payloads)
    print(grafana_url)
    print(
        build_grafana_session_kwargs(
            token=token,
            basic_user=basic_user,
            basic_password=basic_password,
        )
    )

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60),
        **build_grafana_session_kwargs(
            token=token,
            basic_user=basic_user,
            basic_password=basic_password,
        ),
    ) as session:
        existing = await fetch_existing_annotations(
            session=session,
            grafana_url=grafana_url,
            from_ms=min_time - 24 * 3600 * 1000,
            to_ms=max_time + 24 * 3600 * 1000,
            dashboard_uid=dashboard_uid or "",
            panel_id=panel_id or "",
            limit=5000,
        )

        result = await sync_payloads(
            session=session,
            grafana_url=grafana_url,
            payloads=payloads,
            existing_annotations=existing,
            dry_run=dry_run,
        )
        return result


async def async_main() -> int:
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Idempotently sync events.csv and periods.csv to Grafana annotations."
        )
    )
    polar_dir = Path("/home/berland/polar_dump")
    parser.add_argument(
        "--grafana-url",
        default="http://localhost:3000",
        help="Base URL, e.g. http://localhost:3000",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GRAFANA_TOKEN", ""),
        help="Grafana service account token",
    )
    parser.add_argument(
        "--events", default=polar_dir / "events.csv", help="Path to events.csv"
    )
    parser.add_argument(
        "--periods", default=polar_dir / "periods.csv", help="Path to periods.csv"
    )
    parser.add_argument("--dashboard-uid", default=None, help="Optional dashboard UID")
    parser.add_argument("--panel-id", type=int, default=None, help="Optional panel ID")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show actions without writing"
    )
    args = parser.parse_args()

    result = await sync_grafana_annotations(
        grafana_url=args.grafana_url,
        token=args.token,
        events_csv=args.events,
        periods_csv=args.periods,
        dashboard_uid=args.dashboard_uid,
        panel_id=args.panel_id,
        dry_run=args.dry_run,
    )

    if os.getenv("BOBSERV_GRAFANA_URL"):
        result = await sync_grafana_annotations(
            grafana_url=os.getenv("BOBSERV_GRAFANA_URL"),
            token=os.getenv("BOBSERV_GRAFANA_API_KEY"),
            basic_user=os.getenv("RAASERVNO_HTTP_BASIC_AUTH_USER"),
            basic_password=os.getenv("RAASERVNO_HTTP_BASIC_AUTH_PASSWORD"),
            events_csv=args.events,
            periods_csv=args.periods,
            dashboard_uid=args.dashboard_uid,
            panel_id=args.panel_id,
            dry_run=args.dry_run,
        )

    print(json.dumps(result, indent=2), flush=True)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
