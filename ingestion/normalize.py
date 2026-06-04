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


def _to_dt(s: pd.Series) -> pd.Series:
    """Parse a timestamp column that may be epoch-milliseconds (live Garmin returns
    ints like 1780272180000) or ISO strings (demo). Plain ``pd.to_datetime`` would
    read the ints as *nanoseconds* and land everything in 1970, which silently breaks
    hour-of-day baselines and point-in-time ordering.
    """
    s = pd.Series(s)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():  # numeric -> epoch milliseconds
        out = pd.to_datetime(num, unit="ms", errors="coerce")
        missing = out.isna() & s.notna()  # any non-numeric entries (mixed/demo)
        if missing.any():
            out.loc[missing] = pd.to_datetime(s[missing], errors="coerce")
        return out
    return pd.to_datetime(s, errors="coerce")


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Force the listed columns to float64 (NaN for missing).

    Real Garmin data can leave a whole column empty (e.g. a device with no HRV, or
    epoch records that only carry stress). pandas types an all-None column as
    ``object``, which then breaks float math and Hopsworks feature-group type
    inference; coercing to numeric makes them clean nullable doubles.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
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
    # Drop nights with no recorded sleep (watch not worn): zero duration, null score,
    # and null sleep timestamps. They carry no sleep signal and their NaT timestamps
    # cannot be serialized as feature values. The date still exists in the daily grain.
    _dur = pd.to_numeric(df["sleep_duration_min"], errors="coerce").fillna(0)
    df = df[_dur > 0].reset_index(drop=True)
    for c in ("sleep_start_time", "wakeup_time"):
        df[c] = _to_dt(df[c])
    # PIT event_time = when the night's data becomes available, i.e. wake-up.
    # Some nights have no recorded wake-up time (NaT); fall back to the morning of the
    # sleep date so event_time is never null (Hopsworks requires a valid event_time).
    fallback = pd.to_datetime(df["sleep_date"]) + pd.Timedelta(hours=8)
    df["event_time"] = pd.to_datetime(df["wakeup_time"], errors="coerce").fillna(fallback)
    df = _coerce_numeric(df, [
        "sleep_duration_min", "deep_sleep_min", "rem_sleep_min", "light_sleep_min",
        "awake_min", "sleep_score", "sleep_efficiency", "avg_sleep_hr",
        "avg_sleep_respiration", "avg_sleep_spo2",
        "hrv_avg_sleep", "hrv_lowest_sleep", "hrv_highest_sleep",
    ])
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
    df = _coerce_numeric(df, [
        "duration_min", "moving_duration_min", "distance_m", "avg_hr", "max_hr",
        "calories", "avg_speed", "max_speed", "elevation_gain_m",
        "training_effect_aerobic", "training_effect_anaerobic",
        "time_in_zone_1_min", "time_in_zone_2_min", "time_in_zone_3_min",
        "time_in_zone_4_min", "time_in_zone_5_min",
    ])
    return _stamp(df, user_id, source)


def to_epoch_df(snaps: list[dict[str, Any]], user_id: str, source: str) -> pd.DataFrame:
    df = pd.DataFrame(snaps)
    df["epoch_start_time"] = _to_dt(df["epoch_start_time"])
    # epoch_end_time is absent on the live path (point samples). Fill it with the start
    # so the timestamp column has no NaT (Hopsworks can't avro-serialize NaT).
    df["epoch_end_time"] = _to_dt(df["epoch_end_time"]).fillna(df["epoch_start_time"])
    df["event_time"] = df["epoch_start_time"]
    # Drop unusable / duplicate samples: Garmin stress arrays overlap at day
    # boundaries, producing duplicate (user_id, epoch_start_time) primary keys.
    df = df.dropna(subset=["epoch_start_time"])
    df = df.drop_duplicates(subset=["epoch_start_time"], keep="last")
    df = _coerce_numeric(df, [
        "steps", "active_calories", "avg_hr", "stress_level",
        "body_battery", "respiration_rate", "spo2",
    ])
    return _stamp(df, user_id, source)
