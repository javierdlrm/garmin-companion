"""Light derivations on the raw daily-summary frame.

Kept deliberately small — the heavy lifting (rolling baselines, load) lives in the
dedicated modules. These derived columns feed the baselines and the readiness view.
"""

from __future__ import annotations

import pandas as pd

#: Columns of the ``fg_garmin_daily_summary_raw`` feature group (draft §8.1.1).
RAW_COLUMNS = [
    "user_id", "summary_date", "event_time",
    "steps", "intensity_minutes", "active_calories", "bmr_calories",
    "resting_hr", "min_hr", "max_hr", "avg_stress", "max_stress",
    "body_battery_high", "body_battery_low", "body_battery_start", "body_battery_end",
    "spo2_avg", "respiration_avg", "source", "ingested_at",
]


def add_derived(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add cheap same-row derivations (no cross-row leakage)."""
    df = daily_df.copy()
    # body_battery_start is unreliable on the live Garmin path (the unofficial API
    # exposes no clean start-of-day value), so fall back to the day's high — Body
    # Battery peaks in the morning after the overnight recharge, making it a good
    # morning proxy. Demo data carries body_battery_start, so it is preferred there.
    morning = df["body_battery_start"].where(df["body_battery_start"].notna(), df["body_battery_high"])
    # Overnight recharge: how much Body Battery climbed from the day's low to morning.
    df["body_battery_recharge"] = (morning - df["body_battery_low"]).clip(lower=0)
    # Morning Body Battery proxy used by the readiness view.
    df["body_battery_morning"] = morning
    return df
