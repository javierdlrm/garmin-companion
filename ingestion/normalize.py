"""Normalize canonical raw snapshots into typed, feature-group-ready DataFrames.

Both Garmin sources (live and demo) emit the same *canonical raw* schema, so this is
the single place where raw → tabular mapping happens. Here we also:

* attach ``user_id`` (the source is account-scoped; the entity model is multi-user
  from day one — draft §7),
* derive ``event_time`` (the timestamp Hopsworks uses for point-in-time joins),
* stamp ``source`` and ``ingested_at`` for lineage,
* add an explicit **HRV missingness indicator** (DESIGN_DECISIONS T4 — HRV is MNAR,
  so missingness is itself a signal and must never be silently mean-imputed),
* fill ``activity_end_time`` from start + duration when absent.

Column names match the feature-group schemas created in the feature pipeline notebook.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _stamp(df: pd.DataFrame, user_id: str, source: str) -> pd.DataFrame:
    df.insert(0, "user_id", user_id)
    df["source"] = source
    df["ingested_at"] = pd.Timestamp.utcnow().tz_localize(None)
    return df


def to_daily_summary_df(snaps: list[dict[str, Any]], user_id: str, source: str) -> pd.DataFrame:
    df = pd.DataFrame(snaps)
    df["summary_date"] = pd.to_datetime(df["summary_date"]).dt.date
    # event_time anchors PIT joins. Daily summaries are known end-of-day; we use local
    # midday of the Garmin calendar day (see T2 day-boundary note in DESIGN_DECISIONS).
    df["event_time"] = pd.to_datetime(df["summary_date"]) + pd.Timedelta(hours=12)
    num_cols = [c for c in df.columns if c not in {"summary_date", "event_time"}]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="ignore")
    return _stamp(df, user_id, source)


def to_sleep_df(snaps: list[dict[str, Any]], user_id: str, source: str) -> pd.DataFrame:
    df = pd.DataFrame(snaps)
    df["sleep_date"] = pd.to_datetime(df["sleep_date"]).dt.date
    for c in ("sleep_start_time", "wakeup_time"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    # PIT event_time = when the night's data becomes available, i.e. wake-up.
    df["event_time"] = df["wakeup_time"]
    # MNAR missingness indicator (T4). Keep raw HRV as-is (NaN preserved, not imputed).
    df["hrv_missing"] = df["hrv_avg_sleep"].isna().astype(int)
    return _stamp(df, user_id, source)


def to_activity_df(snaps: list[dict[str, Any]], user_id: str, source: str) -> pd.DataFrame:
    df = pd.DataFrame(snaps)
    df["activity_id"] = df["activity_id"].astype(str)
    df["activity_start_time"] = pd.to_datetime(df["activity_start_time"], errors="coerce")
    df["activity_end_time"] = pd.to_datetime(df["activity_end_time"], errors="coerce")
    # Derive end time where the source didn't provide it.
    missing_end = df["activity_end_time"].isna() & df["duration_min"].notna()
    df.loc[missing_end, "activity_end_time"] = df.loc[missing_end, "activity_start_time"] + pd.to_timedelta(
        df.loc[missing_end, "duration_min"], unit="m"
    )
    df["event_time"] = df["activity_start_time"]
    return _stamp(df, user_id, source)


def to_epoch_df(snaps: list[dict[str, Any]], user_id: str, source: str) -> pd.DataFrame:
    df = pd.DataFrame(snaps)
    df["epoch_start_time"] = pd.to_datetime(df["epoch_start_time"], errors="coerce")
    df["epoch_end_time"] = pd.to_datetime(df["epoch_end_time"], errors="coerce")
    df["event_time"] = df["epoch_start_time"]
    return _stamp(df, user_id, source)
