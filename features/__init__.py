"""Derived feature builders for the Garmin companion system.

The *raw* feature groups (daily summary, sleep, activity, epoch) come straight from
:mod:`ingestion.normalize`. The modules here build the *derived* feature groups that
the models consume, with point-in-time correctness baked in:

* :mod:`features.daily_summary` / :mod:`features.sleep` — light derivations on the raw
  daily/sleep frames (overnight recharge, sleep debt) used by the baselines.
* :mod:`features.activity` — per-activity zone-weighted load.
* :mod:`features.epoch` — time-of-day baselines + request-time (on-demand) realtime
  features for the stress-anomaly model.
* :mod:`features.baselines` — trailing 7d/28d **raw** baselines, resolved strictly
  as-of (no same-day leakage). Deltas vs. these live in :mod:`transformations`.
* :mod:`features.training_load` — EWMA acute/chronic load, monotony, strain.
* :mod:`features.recovery_episodes` — post-workout episodes with a **frozen
  pre-activity** baseline and censored recovery labels.
* :mod:`features.weak_labels` — readiness weak-label rule + demo manual feedback.
"""
