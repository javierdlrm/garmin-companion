"""DEMO_MODE sample data — illustrative Garmin-shaped snapshots.

This is **not** a physiology simulator and **not** the recommended data path (the
user-facing path is ``python-garminconnect`` via :class:`~ingestion.garmin_client.
GarminConnectSource`). It exists only so the notebooks execute end-to-end without a
Garmin account, for learning. Numbers are plausible but fabricated — do not draw any
health conclusions from them.

It emits the same **canonical raw** schema as the live source, so
:mod:`ingestion.normalize` has a single code path. There is one deliberate signal in
the data — training load on a day raises next-morning resting HR and lowers HRV — so
the downstream readiness/recovery features have *something* to learn. A few nights
have missing HRV to exercise the MNAR handling (DESIGN_DECISIONS T4).
"""

from __future__ import annotations

import datetime as dt
import math
import random
from typing import Any

# Garmin activityType.typeKey values we sample from, with a rough load multiplier
# (DESIGN_DECISIONS B3/T8 — we compute load ourselves rather than trusting a
# device-derived training-load metric).
_ACTIVITY_TYPES = [
    ("running", 1.0),
    ("cycling", 0.9),
    ("strength_training", 1.2),
    ("walking", 0.4),
    ("indoor_cardio", 1.1),
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _daterange(start_date: dt.date, end_date: dt.date):
    day = start_date
    while day <= end_date:
        yield day
        day += dt.timedelta(days=1)


def _latent_load_series(start_date, end_date, rng) -> dict[dt.date, float]:
    """A per-day training-load latent that drives the rest of the signals."""
    load = {}
    for day in _daterange(start_date, end_date):
        # ~3-4 training days a week, weekend long sessions a bit harder.
        base = rng.random()
        if base < 0.45:
            day_load = 0.0
        else:
            day_load = rng.uniform(20, 90)
            if day.weekday() >= 5:
                day_load *= 1.3
        load[day] = round(day_load, 1)
    return load


def generate_daily_summaries(start_date: dt.date, end_date: dt.date, seed: int = 42) -> list[dict[str, Any]]:
    rng = _rng(seed)
    load = _latent_load_series(start_date, end_date, rng)
    prev_load = 0.0
    out = []
    for day in _daterange(start_date, end_date):
        # Yesterday's load elevates today's resting HR and stress.
        fatigue = prev_load / 90.0
        resting_hr = round(48 + 8 * fatigue + rng.gauss(0, 1.5), 1)
        avg_stress = round(max(5, min(95, 30 + 25 * fatigue + rng.gauss(0, 6))), 1)
        bb_low = round(max(5, 35 - 20 * fatigue + rng.gauss(0, 5)), 1)
        bb_high = round(min(100, 80 + 15 * (1 - fatigue) + rng.gauss(0, 5)), 1)
        out.append(
            {
                "summary_date": day.isoformat(),
                "steps": int(rng.gauss(9000, 2500)),
                "intensity_minutes": int(max(0, load[day] / 2 + rng.gauss(10, 8))),
                "active_calories": round(300 + load[day] * 6 + rng.gauss(0, 80), 1),
                "bmr_calories": round(1650 + rng.gauss(0, 40), 1),
                "resting_hr": resting_hr,
                "min_hr": round(resting_hr - rng.uniform(2, 6), 1),
                "max_hr": round(150 + rng.gauss(0, 12), 1),
                "avg_stress": avg_stress,
                "max_stress": round(min(100, avg_stress + rng.uniform(20, 45)), 1),
                "body_battery_high": bb_high,
                "body_battery_low": bb_low,
                "body_battery_start": round(min(100, bb_high - rng.uniform(0, 10)), 1),
                "body_battery_end": round(max(5, bb_low + rng.uniform(0, 15)), 1),
                "spo2_avg": round(min(100, 96 + rng.gauss(0, 1)), 1),
                "respiration_avg": round(14 + rng.gauss(0, 1), 1),
            }
        )
        prev_load = load[day]
    return out


def generate_sleep(start_date: dt.date, end_date: dt.date, seed: int = 42) -> list[dict[str, Any]]:
    rng = _rng(seed + 1)
    load = _latent_load_series(start_date, end_date, _rng(seed))  # same latent as daily
    prev_load = 0.0
    out = []
    for day in _daterange(start_date, end_date):
        fatigue = prev_load / 90.0
        total = max(240, rng.gauss(430 - 20 * fatigue, 45))  # minutes
        deep = total * rng.uniform(0.12, 0.20)
        rem = total * rng.uniform(0.18, 0.26)
        awake = total * rng.uniform(0.03, 0.08)
        light = max(0.0, total - deep - rem - awake)
        start_ts = dt.datetime.combine(day - dt.timedelta(days=1), dt.time(23, 0)) + dt.timedelta(
            minutes=rng.uniform(-40, 60)
        )
        wake_ts = start_ts + dt.timedelta(minutes=total + awake)
        # HRV drops with fatigue; ~8% of nights have no HRV reading (MNAR — T4).
        hrv_avg = None if rng.random() < 0.08 else round(max(20, 65 - 18 * fatigue + rng.gauss(0, 4)), 1)
        out.append(
            {
                "sleep_date": day.isoformat(),
                "sleep_start_time": start_ts.isoformat(),
                "wakeup_time": wake_ts.isoformat(),
                "sleep_duration_min": round(total, 1),
                "deep_sleep_min": round(deep, 1),
                "rem_sleep_min": round(rem, 1),
                "light_sleep_min": round(light, 1),
                "awake_min": round(awake, 1),
                "sleep_score": round(max(20, min(100, 82 - 25 * fatigue + rng.gauss(0, 6))), 1),
                "sleep_efficiency": round(min(100, 88 + rng.gauss(0, 4)), 1),
                "avg_sleep_hr": round(52 + 6 * fatigue + rng.gauss(0, 2), 1),
                "avg_sleep_respiration": round(14 + rng.gauss(0, 0.8), 1),
                "avg_sleep_spo2": round(min(100, 95 + rng.gauss(0, 1)), 1),
                "hrv_avg_sleep": hrv_avg,
                "hrv_lowest_sleep": None if hrv_avg is None else round(hrv_avg - rng.uniform(5, 12), 1),
                "hrv_highest_sleep": None if hrv_avg is None else round(hrv_avg + rng.uniform(5, 12), 1),
            }
        )
        prev_load = load[day]
    return out


def generate_activities(start_date: dt.date, end_date: dt.date, seed: int = 42) -> list[dict[str, Any]]:
    rng = _rng(seed + 2)
    load = _latent_load_series(start_date, end_date, _rng(seed))
    out = []
    counter = 0
    for day in _daterange(start_date, end_date):
        if load[day] <= 0:
            continue
        counter += 1
        act_type, mult = rng.choice(_ACTIVITY_TYPES)
        duration_min = max(15.0, load[day] / mult)
        start_ts = dt.datetime.combine(day, dt.time(18, 0)) + dt.timedelta(minutes=rng.uniform(-180, 120))
        end_ts = start_ts + dt.timedelta(minutes=duration_min)
        intensity = min(1.0, load[day] / 90.0)
        # Distribute minutes across HR zones, weighted toward higher zones on hard days.
        z = [0.30, 0.30, 0.20, 0.13, 0.07]
        if intensity > 0.6:
            z = [0.15, 0.25, 0.25, 0.22, 0.13]
        zones = {f"time_in_zone_{i + 1}_min": round(duration_min * z[i], 1) for i in range(5)}
        out.append(
            {
                "activity_id": f"{day.isoformat()}-{counter:04d}",
                "activity_start_time": start_ts.isoformat(),
                "activity_end_time": end_ts.isoformat(),
                "activity_type": act_type,
                "duration_min": round(duration_min, 1),
                "moving_duration_min": round(duration_min * rng.uniform(0.9, 1.0), 1),
                "distance_m": None if act_type == "strength_training" else round(duration_min * rng.uniform(150, 260), 1),
                "avg_hr": round(120 + 30 * intensity + rng.gauss(0, 5), 1),
                "max_hr": round(150 + 25 * intensity + rng.gauss(0, 5), 1),
                "calories": round(duration_min * (6 + 4 * intensity), 1),
                "avg_speed": None if act_type == "strength_training" else round(2.5 + rng.gauss(0, 0.4), 2),
                "max_speed": None if act_type == "strength_training" else round(3.5 + rng.gauss(0, 0.5), 2),
                "elevation_gain_m": round(max(0, rng.gauss(60, 40)), 1),
                "training_effect_aerobic": round(min(5, 2 + 2.5 * intensity + rng.gauss(0, 0.3)), 2),
                "training_effect_anaerobic": round(min(5, 0.5 + 2.5 * intensity + rng.gauss(0, 0.3)), 2),
                **zones,
            }
        )
    return out


def generate_epochs(start_date: dt.date, end_date: dt.date, seed: int = 42, step_hours: int = 2) -> list[dict[str, Any]]:
    """Intra-day epochs at ``step_hours`` granularity (modest volume for the demo)."""
    rng = _rng(seed + 3)
    out = []
    for day in _daterange(start_date, end_date):
        for hour in range(0, 24, step_hours):
            ts = dt.datetime.combine(day, dt.time(hour, 0))
            # Circadian-ish HR/stress: low overnight, higher midday.
            circ = math.sin((hour - 6) / 24 * 2 * math.pi)
            asleep = hour < 7
            out.append(
                {
                    "epoch_start_time": ts.isoformat(),
                    "epoch_end_time": (ts + dt.timedelta(hours=step_hours)).isoformat(),
                    "steps": 0 if asleep else int(max(0, rng.gauss(700, 400))),
                    "active_calories": 0.0 if asleep else round(max(0, rng.gauss(40, 25)), 1),
                    "avg_hr": round((48 if asleep else 70) + 12 * max(0, circ) + rng.gauss(0, 4), 1),
                    "stress_level": round(max(0, (12 if asleep else 35) + 20 * max(0, circ) + rng.gauss(0, 8)), 1),
                    "body_battery": round(min(100, max(5, 60 + 25 * circ + rng.gauss(0, 6))), 1),
                    "respiration_rate": round((13 if asleep else 15) + rng.gauss(0, 1), 1),
                    "spo2": round(min(100, 96 + rng.gauss(0, 1)), 1),
                }
            )
    return out
