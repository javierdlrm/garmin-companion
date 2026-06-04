"""Swappable Garmin data source.

The rest of the tutorial never imports ``garminconnect`` directly — it talks to a
:class:`GarminSource`. That keeps the unofficial client isolated (it may break or
change), makes ``DEMO_MODE`` trivial, and leaves a clean seam to later drop in the
official Garmin Health/Activity API.

Every ``fetch_*`` method returns a list of *raw* JSON-like ``dict`` snapshots, one
per day / activity. Raw snapshots are persisted to ``raw_dir`` (draft §3.3 step 2:
"store raw Garmin JSON snapshots") so the normalize step and any reprocessing are
reproducible and decoupled from the live API.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from abc import ABC, abstractmethod
from typing import Any


def _date_range(start_date: dt.date, end_date: dt.date):
    day = start_date
    while day <= end_date:
        yield day
        day += dt.timedelta(days=1)


class GarminSource(ABC):
    """Abstract Garmin data source.

    Implementations return raw, lightly-shaped Garmin snapshots. Normalization to
    the feature-group schemas happens later in :mod:`ingestion.normalize`, so both
    the live and demo backends share a single normalize path (avoids skew between
    "what I tested on" and "what runs in production").
    """

    def __init__(self, raw_dir: str = "data/raw", source_name: str = "garmin") -> None:
        self.raw_dir = raw_dir
        self.source_name = source_name

    @abstractmethod
    def fetch_daily_summaries(self, start_date: dt.date, end_date: dt.date) -> list[dict[str, Any]]:
        """All-day wellness summaries (steps, RHR, stress, Body Battery, ...)."""

    @abstractmethod
    def fetch_sleep(self, start_date: dt.date, end_date: dt.date) -> list[dict[str, Any]]:
        """Per-night sleep + overnight HRV / respiration / SpO2."""

    @abstractmethod
    def fetch_activities(self, start_date: dt.date, end_date: dt.date) -> list[dict[str, Any]]:
        """Completed workouts with HR-zone breakdown."""

    @abstractmethod
    def fetch_epochs(self, start_date: dt.date, end_date: dt.date) -> list[dict[str, Any]]:
        """Intra-day epoch summaries (the near-real-time signal)."""

    def persist_raw(self, kind: str, snapshots: list[dict[str, Any]]) -> str:
        """Write raw snapshots to ``{raw_dir}/{kind}.json`` and return the path."""
        os.makedirs(self.raw_dir, exist_ok=True)
        path = os.path.join(self.raw_dir, f"{kind}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snapshots, fh, indent=2, default=str)
        return path


class GarminConnectSource(GarminSource):
    """Recommended path — wraps the unofficial ``python-garminconnect`` client.

    Requires Garmin Connect credentials (env ``GARMIN_EMAIL`` / ``GARMIN_PASSWORD``
    or passed explicitly). Install with ``pip install garminconnect``.

    Note on day boundaries (DESIGN_DECISIONS T2): Garmin buckets daily summaries by
    *device-local* midnight. We keep Garmin's own ``calendarDate`` as ``summary_date``
    rather than re-bucketing in UTC, so trailing windows and as-of joins stay aligned
    with how Garmin assigned the data.
    """

    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        raw_dir: str = "data/raw",
    ) -> None:
        super().__init__(raw_dir=raw_dir, source_name="garminconnect")
        self._email = email or os.environ.get("GARMIN_EMAIL")
        self._password = password or os.environ.get("GARMIN_PASSWORD")
        self._api = None

    def _client(self):
        if self._api is None:
            try:
                from garminconnect import Garmin  # imported lazily — optional dep
            except ImportError as exc:  # pragma: no cover - depends on user env
                raise ImportError(
                    "python-garminconnect is required for the live path. "
                    "Install it with `pip install garminconnect`, or set DEMO_MODE=True."
                ) from exc
            if not self._email or not self._password:
                raise ValueError(
                    "Garmin credentials missing. Set GARMIN_EMAIL / GARMIN_PASSWORD "
                    "(or pass them to GarminConnectSource), or use DEMO_MODE=True."
                )
            api = Garmin(self._email, self._password)
            api.login()
            self._api = api
        return self._api

    def fetch_daily_summaries(self, start_date, end_date):
        api = self._client()
        out = []
        for day in _date_range(start_date, end_date):
            iso = day.isoformat()
            stats = api.get_stats(iso) or {}
            bb = api.get_body_battery(iso, iso) or [{}]
            bb0 = bb[0] if isinstance(bb, list) and bb else {}
            out.append(self._canon_daily(iso, stats, bb0))
        return self._persist_and_return("daily_summary", out)

    @staticmethod
    def _canon_daily(iso, stats, bb):
        """Adapt garminconnect's wellness dicts into the canonical raw schema.

        Field names in the unofficial API drift between versions/devices — these
        ``.get(...)`` lookups are best-effort and may need tweaking against your data.
        """
        return {
            "summary_date": iso,
            "steps": stats.get("totalSteps"),
            "intensity_minutes": (stats.get("moderateIntensityMinutes", 0) or 0)
            + (stats.get("vigorousIntensityMinutes", 0) or 0),
            "active_calories": stats.get("activeKilocalories"),
            "bmr_calories": stats.get("bmrKilocalories"),
            "resting_hr": stats.get("restingHeartRate"),
            "min_hr": stats.get("minHeartRate"),
            "max_hr": stats.get("maxHeartRate"),
            "avg_stress": stats.get("averageStressLevel"),
            "max_stress": stats.get("maxStressLevel"),
            "body_battery_high": stats.get("bodyBatteryHighestValue"),
            "body_battery_low": stats.get("bodyBatteryLowestValue"),
            "body_battery_start": (bb.get("startTimestampGmt") and bb.get("charged")),
            "body_battery_end": stats.get("bodyBatteryMostRecentValue"),
            "spo2_avg": stats.get("averageSpo2"),
            "respiration_avg": stats.get("avgWakingRespirationValue"),
        }

    def fetch_sleep(self, start_date, end_date):
        api = self._client()
        out = []
        for day in _date_range(start_date, end_date):
            iso = day.isoformat()
            sleep = api.get_sleep_data(iso) or {}
            # HRV is captured overnight; not all devices/days have it (MNAR — T4).
            try:
                hrv = api.get_hrv_data(iso)
            except Exception:  # pragma: no cover - device/route dependent
                hrv = None
            out.append(self._canon_sleep(iso, sleep, hrv))
        return self._persist_and_return("sleep", out)

    @staticmethod
    def _canon_sleep(iso, sleep, hrv):
        dto = (sleep or {}).get("dailySleepDTO", {}) or {}
        hrv_summary = (hrv or {}).get("hrvSummary", {}) if hrv else {}
        sec = lambda k: (dto.get(k) or 0) / 60.0  # noqa: E731 - seconds -> minutes
        return {
            "sleep_date": iso,
            "sleep_start_time": dto.get("sleepStartTimestampGMT"),
            "wakeup_time": dto.get("sleepEndTimestampGMT"),
            "sleep_duration_min": sec("sleepTimeSeconds"),
            "deep_sleep_min": sec("deepSleepSeconds"),
            "rem_sleep_min": sec("remSleepSeconds"),
            "light_sleep_min": sec("lightSleepSeconds"),
            "awake_min": sec("awakeSleepSeconds"),
            "sleep_score": (dto.get("sleepScores", {}) or {}).get("overall", {}).get("value"),
            "sleep_efficiency": dto.get("sleepEfficiency"),
            "avg_sleep_hr": (sleep or {}).get("avgSleepHeartRate"),
            "avg_sleep_respiration": (sleep or {}).get("avgRespirationValue"),
            "avg_sleep_spo2": (sleep or {}).get("averageSpO2Value"),
            # HRV may be absent (MNAR) — keep None so normalize can flag missingness.
            "hrv_avg_sleep": hrv_summary.get("lastNightAvg"),
            "hrv_lowest_sleep": hrv_summary.get("lowestHrv"),
            "hrv_highest_sleep": hrv_summary.get("highestHrv"),
        }

    def fetch_activities(self, start_date, end_date):
        api = self._client()
        acts = api.get_activities_by_date(start_date.isoformat(), end_date.isoformat()) or []
        return self._persist_and_return("activity", [self._canon_activity(a) for a in acts])

    @staticmethod
    def _canon_activity(a):
        zones = {f"time_in_zone_{i}_min": None for i in range(1, 6)}
        dur_min = (a.get("duration") or 0) / 60.0
        start = a.get("startTimeGMT")
        return {
            "activity_id": str(a.get("activityId")),
            "activity_start_time": start,
            "activity_end_time": None,  # derived in normalize from start + duration
            "activity_type": (a.get("activityType", {}) or {}).get("typeKey"),
            "duration_min": dur_min,
            "moving_duration_min": (a.get("movingDuration") or 0) / 60.0,
            "distance_m": a.get("distance"),
            "avg_hr": a.get("averageHR"),
            "max_hr": a.get("maxHR"),
            "calories": a.get("calories"),
            "avg_speed": a.get("averageSpeed"),
            "max_speed": a.get("maxSpeed"),
            "elevation_gain_m": a.get("elevationGain"),
            "training_effect_aerobic": a.get("aerobicTrainingEffect"),
            "training_effect_anaerobic": a.get("anaerobicTrainingEffect"),
            **zones,
        }

    def fetch_epochs(self, start_date, end_date):
        api = self._client()
        out = []
        for day in _date_range(start_date, end_date):
            iso = day.isoformat()
            # Intraday HR / stress samples — flattened to one canonical record per
            # sample. The real API returns dense arrays; we down-sample in practice.
            stress = api.get_stress_data(iso) or {}
            for ts, level in (stress.get("stressValuesArray") or []):
                out.append(
                    {
                        "epoch_start_time": ts,
                        "epoch_end_time": None,
                        "steps": None,
                        "active_calories": None,
                        "avg_hr": None,
                        "stress_level": level,
                        "body_battery": None,
                        "respiration_rate": None,
                        "spo2": None,
                    }
                )
        return self._persist_and_return("epoch", out)

    def _persist_and_return(self, kind, snapshots):
        self.persist_raw(kind, snapshots)
        return snapshots


class SampleDataSource(GarminSource):
    """DEMO_MODE backend — delegates to :mod:`ingestion.sample_data`.

    Emits compact, deterministic, Garmin-shaped snapshots so the notebooks run with
    no Garmin account. Illustrative only; do not draw physiological conclusions.
    """

    def __init__(self, raw_dir: str = "data/raw", seed: int = 42) -> None:
        super().__init__(raw_dir=raw_dir, source_name="sample")
        self.seed = seed

    def _gen(self):
        from ingestion import sample_data
        return sample_data

    def fetch_daily_summaries(self, start_date, end_date):
        snaps = self._gen().generate_daily_summaries(start_date, end_date, seed=self.seed)
        self.persist_raw("daily_summary", snaps)
        return snaps

    def fetch_sleep(self, start_date, end_date):
        snaps = self._gen().generate_sleep(start_date, end_date, seed=self.seed)
        self.persist_raw("sleep", snaps)
        return snaps

    def fetch_activities(self, start_date, end_date):
        snaps = self._gen().generate_activities(start_date, end_date, seed=self.seed)
        self.persist_raw("activity", snaps)
        return snaps

    def fetch_epochs(self, start_date, end_date):
        snaps = self._gen().generate_epochs(start_date, end_date, seed=self.seed)
        self.persist_raw("epoch", snaps)
        return snaps


def get_source(demo_mode: bool = True, **kwargs) -> GarminSource:
    """Return the configured Garmin source.

    Args:
        demo_mode: if True, return :class:`SampleDataSource` (no account needed).
            If False, return :class:`GarminConnectSource` (requires credentials).
        **kwargs: forwarded to the chosen backend (e.g. ``email``, ``password``,
            ``raw_dir``, ``seed``).
    """
    if demo_mode:
        return SampleDataSource(**{k: v for k, v in kwargs.items() if k in {"raw_dir", "seed"}})
    return GarminConnectSource(**{k: v for k, v in kwargs.items() if k in {"email", "password", "raw_dir"}})
