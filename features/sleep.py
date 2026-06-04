"""Light derivations on the raw sleep frame.

HRV is treated as optional throughout (device/route dependent and MNAR — see the
``hrv_missing`` indicator added in :mod:`ingestion.normalize`). We never mean-impute
HRV here; downstream models receive both the (possibly NaN) value and the indicator.
"""

from __future__ import annotations

import pandas as pd

#: Columns of the ``fg_garmin_sleep_raw`` feature group (draft §8.1.2) + hrv_missing.
RAW_COLUMNS = [
    "user_id", "sleep_date", "event_time",
    "sleep_start_time", "wakeup_time",
    "sleep_duration_min", "deep_sleep_min", "rem_sleep_min", "light_sleep_min", "awake_min",
    "sleep_score", "sleep_efficiency", "avg_sleep_hr", "avg_sleep_respiration", "avg_sleep_spo2",
    "hrv_avg_sleep", "hrv_lowest_sleep", "hrv_highest_sleep", "hrv_missing",
    "source", "ingested_at",
]

#: A "personal target" used only to express sleep debt as a derived feature.
TARGET_SLEEP_MIN = 450.0


def add_derived(sleep_df: pd.DataFrame) -> pd.DataFrame:
    """Add same-row sleep derivations."""
    df = sleep_df.copy()
    df["sleep_debt_min"] = (TARGET_SLEEP_MIN - df["sleep_duration_min"]).clip(lower=0)
    total = df[["deep_sleep_min", "rem_sleep_min", "light_sleep_min"]].sum(axis=1).replace(0, pd.NA)
    df["deep_sleep_pct"] = (df["deep_sleep_min"] / total * 100).astype(float)
    return df
