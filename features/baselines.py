"""Personal recovery baselines — ``fg_recovery_baselines_daily`` (draft §8.2.1).

Two correctness rules from the panel are enforced here:

* **As-of, no same-day leakage (T3):** every rolling mean is computed over a trailing
  window and then ``shift(1)`` so the baseline for date *D* reflects strictly *< D*.
  Normalization/baseline statistics must never see the value they will be compared to.
* **Store raw baselines only (A1):** this feature group holds rolling *means* only.
  The *delta vs. baseline* (e.g. ``hrv_delta_28d_pct``) is computed at request time as
  an **on-demand transformation** (see :mod:`transformations`), because a trailing
  28-day baseline cannot be captured by a feature-view ``TransformationStatistics``
  (which holds a single training-set-fixed statistic).
"""

from __future__ import annotations

import pandas as pd


def _rolling_asof(group: pd.DataFrame, col: str, window: int, *, how: str = "mean") -> pd.Series:
    """Trailing rolling stat over ``window`` days, shifted to exclude the current day."""
    s = group[col]
    roll = s.rolling(window=window, min_periods=max(2, window // 4))
    out = getattr(roll, how)()
    return out.shift(1)


def build_baselines(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Build the daily baseline feature group.

    Args:
        daily_features: one row per (user_id, date) with at least ``resting_hr``,
            ``hrv_avg_sleep``, ``sleep_duration_min``, ``sleep_debt_min``,
            ``avg_stress``, ``body_battery_morning``, ``body_battery_recharge``,
            ``respiration_avg``, ``spo2_avg`` and a ``date`` column.
    """
    df = daily_features.sort_values(["user_id", "date"]).copy()
    out = df[["user_id", "date"]].copy()

    specs = [
        ("rhr_7d_mean", "resting_hr", 7, "mean"),
        ("rhr_28d_mean", "resting_hr", 28, "mean"),
        ("hrv_7d_mean", "hrv_avg_sleep", 7, "mean"),
        ("hrv_28d_mean", "hrv_avg_sleep", 28, "mean"),
        ("sleep_duration_7d_mean", "sleep_duration_min", 7, "mean"),
        ("sleep_debt_7d_min", "sleep_debt_min", 7, "sum"),
        ("stress_7d_mean", "avg_stress", 7, "mean"),
        ("body_battery_morning_7d_mean", "body_battery_morning", 7, "mean"),
        ("body_battery_recharge_7d_mean", "body_battery_recharge", 7, "mean"),
        ("respiration_28d_mean", "respiration_avg", 28, "mean"),
        ("spo2_28d_mean", "spo2_avg", 28, "mean"),
    ]
    grouped = df.groupby("user_id", group_keys=False)
    for out_col, src_col, window, how in specs:
        if src_col in df:
            out[out_col] = grouped.apply(lambda g, c=src_col, w=window, h=how: _rolling_asof(g, c, w, how=h)).values

    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["event_time"] = pd.to_datetime(out["date"]) + pd.Timedelta(hours=8)  # morning, when used
    return out
