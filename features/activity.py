"""Per-activity load features.

We compute training load **ourselves** from HR-zone time rather than trusting a
device-derived training-load / EPOC metric, which is itself a model output and would
make the load feature partly circular (DESIGN_DECISIONS T8). The base unit is the
zone-weighted HR load from draft §8.2.2, optionally scaled by an activity-type
multiplier.
"""

from __future__ import annotations

import pandas as pd

#: Columns of the ``fg_garmin_activity_raw`` feature group (draft §8.1.3).
RAW_COLUMNS = [
    "user_id", "activity_id", "event_time",
    "activity_start_time", "activity_end_time", "activity_type",
    "duration_min", "moving_duration_min", "distance_m", "avg_hr", "max_hr", "calories",
    "avg_speed", "max_speed", "elevation_gain_m",
    "training_effect_aerobic", "training_effect_anaerobic",
    "time_in_zone_1_min", "time_in_zone_2_min", "time_in_zone_3_min",
    "time_in_zone_4_min", "time_in_zone_5_min",
    "source", "ingested_at",
]

#: Activity-type multipliers (draft §8.2.2). Default 1.0 for unlisted types.
ACTIVITY_TYPE_MULTIPLIER = {
    "walking": 0.5,
    "indoor_cardio": 1.0,
    "cycling": 1.0,
    "running": 1.0,
    "strength_training": 1.2,
    "intervals": 1.4,
    "karate": 1.3,
}

_ZONE_WEIGHTS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def add_load(activity_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``zone_weighted_load``, ``activity_load`` and ``zone4_5_minutes`` columns."""
    df = activity_df.copy()
    zone_cols = {i: f"time_in_zone_{i}_min" for i in range(1, 6)}
    for c in zone_cols.values():
        if c not in df:
            df[c] = 0.0
    df[list(zone_cols.values())] = df[list(zone_cols.values())].fillna(0.0)

    df["zone_weighted_load"] = sum(_ZONE_WEIGHTS[i] * df[zone_cols[i]] for i in range(1, 6))
    mult = df["activity_type"].map(ACTIVITY_TYPE_MULTIPLIER).fillna(1.0)
    df["activity_load"] = df["zone_weighted_load"] * mult
    df["zone4_5_minutes"] = df[zone_cols[4]] + df[zone_cols[5]]
    df["max_hr_pct"] = (df["max_hr"] / 190.0 * 100).round(1)  # crude; replace with HRmax if known
    return df


def daily_load(activity_df_with_load: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-activity load to a per-(user, date) daily load series.

    Days with no activity are intentionally **not** filled here — the caller
    reindexes against the full calendar so rest days become explicit zeros.
    """
    df = activity_df_with_load.copy()
    df["date"] = pd.to_datetime(df["activity_start_time"]).dt.date
    agg = (
        df.groupby(["user_id", "date"])
        .agg(
            load=("activity_load", "sum"),
            zone4_5_minutes=("zone4_5_minutes", "sum"),
            n_sessions=("activity_id", "count"),
            n_strength=("activity_type", lambda s: (s == "strength_training").sum()),
        )
        .reset_index()
    )
    return agg
