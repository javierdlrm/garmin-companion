"""Intra-day / realtime state — drives the stress-anomaly model (draft §8.2.3).

Panel guidance for the stress-anomaly path:

* **Robust statistics:** anomaly z-scores use **median / MAD**, not mean / SD — the SD
  is inflated and masked by the very spikes we hunt.
* **Time-of-day baselines:** stress at 03:00 means something different than at 14:00,
  so baselines are computed per hour-of-day bucket.
* **Context-conditional (exclude activity):** an ``is_active`` flag lets the model
  condition on / exclude active periods, otherwise "anomaly" just means "exercised".
* **Keep offline event-time history (B5):** this feature group is append-only and
  event-time keyed so the anomaly model has a training corpus; the *online* copy can
  be TTL-bounded (A5).

The realtime z-score features are reconstructed identically at serving time from the
request payload via the on-demand transforms in :mod:`transformations`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_STEPS_THRESHOLD = 200  # steps within an epoch above which we call it "active"


def _mad(s: pd.Series) -> float:
    med = s.median()
    return float((s - med).abs().median())


def time_of_day_stats(epoch_df: pd.DataFrame) -> pd.DataFrame:
    """Robust per-(user, hour) baselines for stress and HR over history."""
    df = epoch_df.copy()
    df["hour"] = pd.to_datetime(df["epoch_start_time"]).dt.hour
    rows = []
    for (user_id, hour), g in df.groupby(["user_id", "hour"]):
        rows.append(
            {
                "user_id": user_id,
                "hour": hour,
                "stress_median": g["stress_level"].median(),
                "stress_mad": _mad(g["stress_level"]) or 1.0,
                "hr_median": g["avg_hr"].median(),
                "hr_mad": _mad(g["avg_hr"]) or 1.0,
            }
        )
    return pd.DataFrame(rows)


def build_stress_state(epoch_df: pd.DataFrame, tod_stats: pd.DataFrame) -> pd.DataFrame:
    """Per-epoch realtime-state features for the stress-anomaly training data."""
    df = epoch_df.sort_values(["user_id", "epoch_start_time"]).copy()
    ts = pd.to_datetime(df["epoch_start_time"])
    df["hour"] = ts.dt.hour
    df["minutes_since_wake"] = (ts.dt.hour - 7).clip(lower=0) * 60  # crude circadian proxy
    df["is_active"] = (df["steps"].fillna(0) >= ACTIVE_STEPS_THRESHOLD).astype(int)

    df = df.merge(tod_stats, on=["user_id", "hour"], how="left")
    df["stress_zscore_vs_time_of_day"] = (df["stress_level"] - df["stress_median"]) / df["stress_mad"]
    df["hr_zscore_vs_time_of_day"] = (df["avg_hr"] - df["hr_median"]) / df["hr_mad"]

    # 2-hour Body Battery slope (per-user, ordered by time).
    def slope(g: pd.Series) -> pd.Series:
        return g.diff()

    df["body_battery_slope_2h"] = df.groupby("user_id")["body_battery"].transform(slope)

    keep = [
        "user_id", "epoch_start_time", "event_time", "hour", "minutes_since_wake", "is_active",
        "stress_level", "avg_hr", "body_battery", "respiration_rate",
        "stress_zscore_vs_time_of_day", "hr_zscore_vs_time_of_day", "body_battery_slope_2h",
    ]
    return df[[c for c in keep if c in df.columns]].replace([np.inf, -np.inf], np.nan)


def dedupe_events(flagged: pd.DataFrame, time_col: str = "epoch_start_time", gap_hours: float = 3.0) -> pd.DataFrame:
    """Collapse consecutive flagged epochs into events (stress-eval: count events, not samples)."""
    df = flagged.sort_values(["user_id", time_col]).copy()
    t = pd.to_datetime(df[time_col])
    new_event = (t.groupby(df["user_id"]).diff() > pd.Timedelta(hours=gap_hours)) | t.groupby(df["user_id"]).diff().isna()
    df["event_id"] = new_event.groupby(df["user_id"]).cumsum()
    return df.groupby(["user_id", "event_id"]).first().reset_index()
