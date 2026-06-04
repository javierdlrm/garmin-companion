"""Post-workout recovery episodes — ``fg_recovery_episodes`` (draft §8.2.4).

This is the label source for the recovery-time model. Two panel rules are critical:

* **Frozen, pre-activity baseline (B2/T7):** "recovered" is defined relative to the
  baseline *as it stood the day before the workout*. Recomputing the baseline as
  recovery progresses is both leakage and a degenerate (self-converging) target.
* **Censoring (Survival decision):** if a new hard session starts before recovery
  completes — or the data ends first — the episode is right-censored. We record
  ``censored`` and leave horizon labels ``None`` when the horizon has not elapsed
  uninterrupted, rather than silently calling them "not recovered".

The model itself (regression + binary classification with censored-episode handling)
lives in notebook 2; an optional Kaplan–Meier illustration is suggested there.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

# Recovery thresholds (draft §8.2.4) — a proxy for development, not a medical metric.
HRV_RECOVERED_FRAC = 0.95
RHR_RECOVERED_DELTA = 3.0
SLEEP_SCORE_FLOOR = 50.0
BB_RECOVERED_DELTA = 10.0
HARD_ZONE45_MIN = 15.0


def _morning_lookup(morning_daily: pd.DataFrame) -> dict:
    """{(user_id, date) -> dict of morning metrics}."""
    out = {}
    for _, r in morning_daily.iterrows():
        out[(r["user_id"], r["date"])] = {
            "resting_hr": r.get("resting_hr"),
            "hrv_avg_sleep": r.get("hrv_avg_sleep"),
            "sleep_score": r.get("sleep_score"),
            "body_battery_morning": r.get("body_battery_morning"),
        }
    return out


def _baseline_lookup(baselines: pd.DataFrame) -> dict:
    out = {}
    for _, r in baselines.iterrows():
        out[(r["user_id"], r["date"])] = {
            "rhr": r.get("rhr_28d_mean"),
            "hrv": r.get("hrv_28d_mean"),
            "bb": r.get("body_battery_morning_7d_mean"),
        }
    return out


def _has(v) -> bool:
    """A signal is usable only if it is present AND not NaN (a NaN baseline/morning
    value would otherwise sneak past an ``is not None`` test and make every
    comparison False — censoring every episode)."""
    return v is not None and pd.notna(v)


def _recovered(morning: dict, base: dict) -> bool:
    checks = []
    if _has(base.get("hrv")) and _has(morning.get("hrv_avg_sleep")):
        checks.append(morning["hrv_avg_sleep"] >= HRV_RECOVERED_FRAC * base["hrv"])
    if _has(base.get("rhr")) and _has(morning.get("resting_hr")):
        checks.append(morning["resting_hr"] <= base["rhr"] + RHR_RECOVERED_DELTA)
    if _has(morning.get("sleep_score")):
        checks.append(morning["sleep_score"] >= SLEEP_SCORE_FLOOR)
    if _has(base.get("bb")) and _has(morning.get("body_battery_morning")):
        checks.append(morning["body_battery_morning"] >= base["bb"] - BB_RECOVERED_DELTA)
    # Require at least one comparable signal and all available checks to pass.
    return len(checks) > 0 and all(checks)


def build_recovery_episodes(
    activities: pd.DataFrame,
    morning_daily: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """Build the recovery-episode feature group with censored labels."""
    morning = _morning_lookup(morning_daily)
    base = _baseline_lookup(baselines)

    rows = []
    for user_id, acts in activities.sort_values("activity_start_time").groupby("user_id"):
        acts = acts.reset_index(drop=True)
        all_dates = sorted({d for (u, d) in morning if u == user_id})
        last_date = all_dates[-1] if all_dates else None
        # Hard-session start dates for interruption detection.
        hard_dates = sorted(
            pd.to_datetime(acts.loc[acts["zone4_5_minutes"] >= HARD_ZONE45_MIN, "activity_start_time"]).dt.date.tolist()
        )

        for _, a in acts.iterrows():
            a_start = pd.to_datetime(a["activity_start_time"])
            a_end = pd.to_datetime(a["activity_end_time"])
            a_date = a_start.date()
            b = base.get((user_id, a_date), {})

            recovered_at = None
            censored = 0
            interrupted = False
            scan = [d for d in all_dates if d > a_date]
            for m_date in scan:
                # Interruption: a *different* hard session started after this workout.
                if any(a_date < hd <= m_date for hd in hard_dates):
                    interrupted = True
                    censored = 1
                    break
                if _recovered(morning.get((user_id, m_date), {}), b):
                    recovered_at = m_date
                    break
            if recovered_at is None and not interrupted:
                censored = 1  # right-censored at end of data

            recovery_hours = None
            if recovered_at is not None:
                m_time = dt.datetime.combine(recovered_at, dt.time(7, 0))
                recovery_hours = round((m_time - a_end.to_pydatetime()).total_seconds() / 3600.0, 1)

            rows.append(
                {
                    "user_id": user_id,
                    "activity_id": str(a["activity_id"]),
                    "activity_type": a.get("activity_type"),
                    "activity_end_time": a_end,
                    "event_time": a_end,
                    "activity_load": a.get("activity_load"),
                    "max_hr_pct": a.get("max_hr_pct"),
                    "zone4_5_minutes": a.get("zone4_5_minutes"),
                    "rhr_pre_baseline": b.get("rhr"),
                    "hrv_pre_baseline": b.get("hrv"),
                    "recovered_at": recovered_at,
                    "recovery_hours_label": recovery_hours,
                    "recovered_within_24h_label": _horizon_label(recovery_hours, censored, 24, a_end, last_date),
                    "recovered_within_48h_label": _horizon_label(recovery_hours, censored, 48, a_end, last_date),
                    "censored": censored,
                }
            )
    return pd.DataFrame(rows)


def _horizon_label(recovery_hours, censored, horizon_h, a_end, last_date):
    """1 if recovered within horizon, 0 if definitively not, None if unknown/censored.

    Returns None when the horizon hasn't elapsed in the available data or the episode
    was censored before the horizon — we must not call those "not recovered".
    """
    if recovery_hours is not None:
        return int(recovery_hours <= horizon_h)
    if censored:
        return None
    if last_date is None:
        return None
    horizon_elapsed = (dt.datetime.combine(last_date, dt.time(7, 0)) - a_end.to_pydatetime()).total_seconds() / 3600.0
    return 0 if horizon_elapsed >= horizon_h else None
