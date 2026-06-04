"""Garmin feature pipeline — scheduled daily job.

Production counterpart of ``1_garmin_feature_pipeline.ipynb``: pulls this account's
own Garmin history (credentials from Hopsworks secrets), normalizes it, and upserts the
raw + derived feature groups. Designed to run daily on a rolling 120-day window so the
trailing baselines / EWMA load stay stable and the online store stays fresh.

Reuses the same ``ingestion`` / ``features`` modules as the notebook (single source of
truth). Great Expectations validation is intentionally omitted here to keep the job
environment minimal; the schema/range checks live in the notebook.
"""

import os
import sys
import datetime as dt

# Make the repo's ingestion/ and features/ packages importable when run as a job.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import hopsworks

from ingestion import get_source, normalize
from features import daily_summary, sleep, activity, baselines, training_load, epoch

USER_ID = "javier"
END_DATE = dt.date.today() - dt.timedelta(days=1)        # last complete day
START_DATE = END_DATE - dt.timedelta(days=120)           # ~4 months trailing


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Garmin credentials from Hopsworks secrets (never hard-coded).
    sec = hopsworks.get_secrets_api()
    os.environ["GARMIN_EMAIL"] = sec.get_secret("GARMIN_EMAIL").value
    os.environ["GARMIN_PASSWORD"] = sec.get_secret("GARMIN_PASSWORD").value

    source = get_source(demo_mode=False)
    print(f"Ingesting Garmin {START_DATE} -> {END_DATE}")
    raw_daily = source.fetch_daily_summaries(START_DATE, END_DATE)
    raw_sleep = source.fetch_sleep(START_DATE, END_DATE)
    raw_activities = source.fetch_activities(START_DATE, END_DATE)
    raw_epochs = source.fetch_epochs(START_DATE, END_DATE)
    print(f"daily={len(raw_daily)} sleep={len(raw_sleep)} "
          f"activities={len(raw_activities)} epochs={len(raw_epochs)}")

    daily_df = daily_summary.add_derived(normalize.to_daily_summary_df(raw_daily, USER_ID, source.source_name))
    sleep_df = sleep.add_derived(normalize.to_sleep_df(raw_sleep, USER_ID, source.source_name))
    activity_df = activity.add_load(normalize.to_activity_df(raw_activities, USER_ID, source.source_name))
    epoch_df = normalize.to_epoch_df(raw_epochs, USER_ID, source.source_name)

    # --- Raw feature groups (PK upsert keeps daily reruns idempotent) ---
    daily_fg = fs.get_or_create_feature_group(
        name="fg_garmin_daily_summary_raw", version=1,
        description="All-day Garmin wellness summary",
        primary_key=["user_id", "summary_date"], event_time="event_time",
        online_enabled=True,
        statistics_config={"enabled": True, "histograms": True, "correlations": True},
    )
    daily_fg.insert(daily_df)

    sleep_fg = fs.get_or_create_feature_group(
        name="fg_garmin_sleep_raw", version=1, description="Sleep + overnight HRV",
        primary_key=["user_id", "sleep_date"], event_time="event_time", online_enabled=True,
    )
    sleep_fg.insert(sleep_df)

    activity_fg = fs.get_or_create_feature_group(
        name="fg_garmin_activity_raw", version=1, description="Workouts with HR-zone load",
        primary_key=["user_id", "activity_id"], event_time="event_time", online_enabled=True,
    )
    activity_fg.insert(activity_df)

    epoch_fg = fs.get_or_create_feature_group(
        name="fg_garmin_epoch_raw", version=1, description="Intra-day epochs (near-real-time)",
        primary_key=["user_id", "epoch_start_time"], event_time="event_time",
        online_enabled=True, ttl_enabled=True, ttl=dt.timedelta(days=2),
    )
    epoch_fg.insert(epoch_df)

    # --- Derived feature groups ---
    merged = (
        daily_df.rename(columns={"summary_date": "date"})
        .merge(sleep_df.rename(columns={"sleep_date": "date"})[
            ["user_id", "date", "hrv_avg_sleep", "sleep_duration_min", "sleep_debt_min", "sleep_score"]],
            on=["user_id", "date"], how="left")
    )
    baselines_df = baselines.build_baselines(merged)
    baselines_fg = fs.get_or_create_feature_group(
        name="fg_recovery_baselines_daily", version=1, description="Personal recovery baselines",
        primary_key=["user_id", "date"], event_time="event_time", online_enabled=True,
    )
    baselines_fg.insert(baselines_df)

    daily_load = activity.daily_load(activity_df)
    calendar = merged[["user_id", "date"]].drop_duplicates()
    daily_load_full = (
        calendar.merge(daily_load, on=["user_id", "date"], how="left")
        .fillna({"load": 0.0, "zone4_5_minutes": 0.0, "n_sessions": 0, "n_strength": 0})
    )
    load_df = training_load.build_training_load(daily_load_full)
    load_fg = fs.get_or_create_feature_group(
        name="fg_training_load_daily", version=1, description="EWMA acute/chronic load + strain",
        primary_key=["user_id", "date"], event_time="event_time", online_enabled=True,
    )
    load_fg.insert(load_df)

    tod_stats = epoch.time_of_day_stats(epoch_df)
    stress_state_df = epoch.build_stress_state(epoch_df, tod_stats)
    tod_fg = fs.get_or_create_feature_group(
        name="fg_time_of_day_stats", version=1, description="Robust per-hour stress/HR baselines",
        primary_key=["user_id", "hour"], online_enabled=True,
    )
    tod_fg.insert(tod_stats)

    stress_state_fg = fs.get_or_create_feature_group(
        name="fg_stress_state", version=1,
        description="Realtime stress state; online=latest, offline=event-time history",
        primary_key=["user_id"], event_time="event_time",
        online_enabled=True, ttl_enabled=True, ttl=dt.timedelta(days=1),
    )
    stress_state_fg.insert(stress_state_df)

    print("Feature pipeline complete.")


if __name__ == "__main__":
    main()
