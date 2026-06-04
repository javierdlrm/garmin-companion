# Garmin + Hopsworks Real-time Recovery, Stress Anomaly, and Post-workout Recovery ML System

**Draft report**  
**Date:** 2026-06-02  
**Author:** Javier de la Rúa Martínez / ChatGPT draft

---

## 1. Executive summary

This report proposes a **personal real-time training intelligence system** that combines three related prediction problems:

1. **Training readiness & injury-risk proxy**  
   Predicts whether today is suitable for hard training, light training, or recovery.

2. **Stress anomaly detection**  
   Detects unusual physiological stress patterns compared with the user’s own baseline.

3. **Post-workout recovery time prediction**  
   Predicts how long it may take until recovery markers return to baseline after a workout.

The system should not be framed as a medical diagnosis, injury prediction system, or clinical risk model. A safer and more accurate framing is:

> **A Garmin-powered personal training readiness and recovery monitoring system.**

The system uses Garmin-derived data such as sleep, HRV, resting heart rate, stress, Body Battery, activity load, heart-rate zones, and manual feedback. Hopsworks is used as the feature store, model registry, and online serving platform.

The main user-facing output is a combined recommendation:

```text
Status: Yellow

Recommendation:
Do mobility, light technical karate practice, or easy zone 2 today.
Avoid max-effort intervals or hard lower-body work.

Why:
- HRV is below your personal baseline
- Resting heart rate is elevated
- Post-workout recovery model estimates 14h recovery remaining
- Stress anomaly score is normal, so this is mostly training-load related
```

---

## 2. Goals

### 2.1 Product goals

The system should answer:

- Am I ready to train hard today?
- Is my stress pattern unusual today?
- How long should I wait before another hard session?
- What kind of training should I do today?
- Which factors explain the recommendation?

### 2.2 ML system goals

The system should demonstrate:

- Real-time or near-real-time wearable data ingestion.
- Offline and online feature storage.
- Point-in-time correct training datasets.
- Multiple feature views for different prediction problems.
- Model registry and versioned deployments.
- Online feature retrieval for inference.
- Prediction logging and feedback loops.
- Continuous model improvement using manual labels.

---

## 3. Garmin data availability

### 3.1 Official Garmin Health API

Garmin’s Health API exposes all-day health and wellness data. According to Garmin’s Health API documentation, available categories include:

| Garmin Health API category | Use in this project |
|---|---|
| Steps | Daily activity load, low-intensity movement, baseline activity |
| Intensity minutes | Daily strain proxy |
| Sleep | Recovery, sleep debt, sleep quality features |
| Calories | Energy expenditure proxy |
| Heart rate | Resting HR, daily HR trends, elevated HR anomalies |
| Stress | Stress anomaly detection, readiness features |
| Pulse Ox | Overnight recovery context |
| Body Battery | Core recovery, depletion, and recharge signal |
| Body composition | Optional long-term context |
| Respiration | Overnight recovery and anomaly context |
| Blood pressure | Optional, if available |
| Beat-to-beat interval | Advanced HRV-related analysis where available |

Garmin’s Health API documentation says that health data is provided in JSON format and can include heart rate, steps, calories, sleep, respiration, body composition, and detailed stress, pulse-ox, and epoch summaries for all-day activities.

Source:  
<https://developer.garmin.com/gc-developer-program/health-api/>

### 3.2 Official Garmin Activity API

The Garmin Activity API is relevant for workout and post-workout recovery modeling. It provides detailed fitness activity data and supports activity files such as FIT, GPX, and TCX.

Relevant activity-derived signals include:

| Activity signal | Use |
|---|---|
| Activity type | Running, cycling, strength, walking, martial arts/manual, etc. |
| Start/end time | Recovery clock starts here |
| Duration | Load feature |
| Distance / pace / speed | Endurance load |
| Avg HR / max HR | Cardiovascular intensity |
| Time in HR zones | Training load estimation |
| Calories | Energy expenditure proxy |
| Elevation | Additional load context |
| FIT file details | Laps, intervals, cadence, dynamics, etc. |
| Strength training metadata | Sets/reps if available |

Source:  
<https://developer.garmin.com/gc-developer-program/activity-api/>

### 3.3 Python access options

For a personal project, there are three realistic access strategies:

| Option | Pros | Cons |
|---|---|---|
| Official Garmin Health + Activity API | Stable, documented, consent-based, production-like | Requires approval; commercial use may involve licensing |
| `python-garminconnect` | Easy Python experimentation; many Garmin Connect data methods | Unofficial; may break; not production-grade |
| Aggregator or self-hosted wearable API layer | Can normalize multi-device data | Adds another dependency; Garmin access may still be constrained |

For this project, the recommended approach is:

1. Prototype with `python-garminconnect`.
2. Store raw Garmin JSON snapshots.
3. Abstract the ingestion layer.
4. Later replace the data source with the official Garmin API if needed.

`python-garminconnect` source:  
<https://github.com/cyberjunky/python-garminconnect>

---

## 4. Real-time interpretation

This project is better described as **near-real-time** rather than true second-by-second real time.

Garmin data usually becomes available after the device syncs with Garmin Connect. The useful real-time events are:

```text
Device sync event
New sleep summary available
New stress/Body Battery summary available
New activity completed
New heart-rate or epoch summary available
```

The real-time loop is:

```text
Garmin sync
  → ingestion
  → feature update
  → online feature store
  → model prediction
  → dashboard / notification
```

---

## 5. Proposed architecture

```text
                 ┌────────────────────────┐
                 │ Garmin / Garmin Connect│
                 └───────────┬────────────┘
                             │
              Official API or python-garminconnect
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Ingestion service   │
                  │ Python / FastAPI    │
                  └──────────┬──────────┘
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
┌─────────────────────┐             ┌──────────────────────┐
│ Raw data lake       │             │ Hopsworks feature     │
│ JSON/FIT snapshots  │             │ pipelines             │
└─────────────────────┘             └──────────┬───────────┘
                                                │
                                                ▼
                                   ┌────────────────────────┐
                                   │ Hopsworks Feature Store│
                                   │ offline + online       │
                                   └──────────┬─────────────┘
                                              │
             ┌────────────────────────────────┼──────────────────────────────┐
             ▼                                ▼                              ▼
┌──────────────────────┐        ┌──────────────────────┐       ┌──────────────────────┐
│ Readiness model      │        │ Stress anomaly model │       │ Recovery-time model  │
│ classification       │        │ anomaly detection    │       │ regression/survival  │
└──────────┬───────────┘        └──────────┬───────────┘       └──────────┬───────────┘
           │                               │                              │
           └──────────────────────┬────────┴───────────────┬─────────────┘
                                  ▼                        ▼
                       ┌───────────────────┐     ┌─────────────────────┐
                       │ Online API        │     │ Prediction logs     │
                       │ FastAPI/KServe    │     │ feedback loop       │
                       └─────────┬─────────┘     └─────────────────────┘
                                 ▼
                    Dashboard / Telegram / ntfy / mobile UI
```

---

## 6. Hopsworks components

The system should use the following Hopsworks components:

| Component | Purpose |
|---|---|
| Feature Groups | Store raw and derived features |
| Feature Views | Create model-specific datasets and online feature vectors |
| Offline Feature Store | Historical data for training and backtesting |
| Online Feature Store | Latest features for low-latency inference |
| Model Registry | Versioned models and metadata |
| Model Serving | Deploy online prediction services |
| Prediction Logging | Store predictions for monitoring and retraining |

Relevant Hopsworks documentation:

- Main docs: <https://docs.hopsworks.ai/latest/>
- Feature Store API: <https://docs.hopsworks.ai/feature-store-api/3.9/generated/api/feature_store_api/>
- Feature views and queries: <https://docs.hopsworks.ai/latest/user_guides/fs/feature_view/query/>
- Feature vectors: <https://docs.hopsworks.ai/latest/user_guides/fs/feature_view/feature-vectors/>
- Model Registry: <https://docs.hopsworks.ai/latest/user_guides/mlops/registry/>
- Model serving: <https://docs.hopsworks.ai/latest/user_guides/mlops/>

---

## 7. Entity model

Even for a personal project, model it as multi-user from day one.

Core entities:

```text
user_id
date
event_time
activity_id
sync_id
```

For a single-user prototype:

```text
user_id = "javier"
```

---

# 8. Feature store design

## 8.1 Raw feature groups

Raw feature groups should stay close to Garmin data. They preserve fields with minimal transformation.

---

### 8.1.1 `fg_garmin_daily_summary_raw`

**Purpose:** all-day summary metrics.  
**Primary key:** `user_id`, `summary_date`  
**Event time:** `event_time` or `summary_timestamp`  
**Online enabled:** yes  
**Stream enabled:** optional, recommended if updating frequently

| Feature | Type | Notes |
|---|---:|---|
| `user_id` | string | Entity |
| `summary_date` | date | Garmin day |
| `event_time` | timestamp | When summary was generated or received |
| `steps` | int | Daily steps |
| `intensity_minutes` | int | Daily intensity |
| `active_calories` | float | Activity energy |
| `bmr_calories` | float | Basal estimate |
| `resting_hr` | float | Daily resting HR |
| `min_hr` | float | Optional |
| `max_hr` | float | Optional |
| `avg_stress` | float | Daily stress |
| `max_stress` | float | Optional |
| `body_battery_high` | float | Day peak |
| `body_battery_low` | float | Day minimum |
| `body_battery_start` | float | Morning or first value |
| `body_battery_end` | float | Last value |
| `spo2_avg` | float | If available |
| `respiration_avg` | float | If available |
| `source` | string | `official_api` or `garminconnect` |
| `ingested_at` | timestamp | Pipeline timestamp |

---

### 8.1.2 `fg_garmin_sleep_raw`

**Purpose:** sleep and overnight recovery features.  
**Primary key:** `user_id`, `sleep_date`  
**Event time:** `wakeup_time`  
**Online enabled:** yes

| Feature | Type |
|---|---:|
| `user_id` | string |
| `sleep_date` | date |
| `sleep_start_time` | timestamp |
| `wakeup_time` | timestamp |
| `sleep_duration_min` | float |
| `deep_sleep_min` | float |
| `rem_sleep_min` | float |
| `light_sleep_min` | float |
| `awake_min` | float |
| `sleep_score` | float |
| `sleep_efficiency` | float |
| `avg_sleep_hr` | float |
| `avg_sleep_respiration` | float |
| `avg_sleep_spo2` | float |
| `hrv_avg_sleep` | float |
| `hrv_lowest_sleep` | float |
| `hrv_highest_sleep` | float |

HRV availability may vary depending on device and API route. The system should treat HRV as optional and support fallback features.

---

### 8.1.3 `fg_garmin_activity_raw`

**Purpose:** workout and activity-level data.  
**Primary key:** `user_id`, `activity_id`  
**Event time:** `activity_start_time`  
**Online enabled:** yes

| Feature | Type |
|---|---:|
| `user_id` | string |
| `activity_id` | string |
| `activity_start_time` | timestamp |
| `activity_end_time` | timestamp |
| `activity_type` | string |
| `duration_min` | float |
| `moving_duration_min` | float |
| `distance_m` | float |
| `avg_hr` | float |
| `max_hr` | float |
| `calories` | float |
| `avg_speed` | float |
| `max_speed` | float |
| `elevation_gain_m` | float |
| `training_effect_aerobic` | float |
| `training_effect_anaerobic` | float |
| `time_in_zone_1_min` | float |
| `time_in_zone_2_min` | float |
| `time_in_zone_3_min` | float |
| `time_in_zone_4_min` | float |
| `time_in_zone_5_min` | float |
| `fit_file_path` | string |
| `ingested_at` | timestamp |

---

### 8.1.4 `fg_garmin_epoch_raw`

**Purpose:** high-frequency or intra-day summaries.  
**Primary key:** `user_id`, `epoch_start_time`  
**Event time:** `epoch_start_time`  
**Online enabled:** optional  
**Stream enabled:** yes

| Feature | Type |
|---|---:|
| `user_id` | string |
| `epoch_start_time` | timestamp |
| `epoch_end_time` | timestamp |
| `steps` | int |
| `active_calories` | float |
| `avg_hr` | float |
| `stress_level` | float |
| `body_battery` | float |
| `respiration_rate` | float |
| `spo2` | float |

This is the main near-real-time feature group. Full history can stay offline, while recent windows can be kept online.

---

## 8.2 Derived feature groups

Derived feature groups contain model-ready features.

---

### 8.2.1 `fg_recovery_baselines_daily`

**Purpose:** personal recovery baselines.  
**Primary key:** `user_id`, `date`  
**Event time:** `date`  
**Online enabled:** yes

| Feature | Meaning |
|---|---|
| `rhr_7d_mean` | Short-term resting HR baseline |
| `rhr_28d_mean` | Long-term resting HR baseline |
| `rhr_delta_28d` | Today vs long-term baseline |
| `hrv_7d_mean` | Short-term HRV baseline |
| `hrv_28d_mean` | Long-term HRV baseline |
| `hrv_delta_28d_pct` | HRV deviation |
| `sleep_duration_7d_mean` | Sleep trend |
| `sleep_debt_7d_min` | Accumulated sleep deficit |
| `stress_7d_mean` | Stress baseline |
| `body_battery_morning_7d_mean` | Morning energy baseline |
| `body_battery_recharge_7d_mean` | Overnight recharge baseline |
| `respiration_28d_mean` | Baseline respiration |
| `spo2_28d_mean` | Baseline SpO₂ |

Important: these features must be computed with strict point-in-time correctness. Today’s prediction must not use future data.

---

### 8.2.2 `fg_training_load_daily`

**Purpose:** recent training strain and load accumulation.  
**Primary key:** `user_id`, `date`  
**Event time:** `date`  
**Online enabled:** yes

| Feature | Meaning |
|---|---|
| `load_1d` | Yesterday/today load |
| `load_3d_sum` | Very acute load |
| `load_7d_sum` | Acute load |
| `load_28d_sum` | Chronic load |
| `acute_chronic_load_ratio` | 7d / 28d normalized |
| `hard_sessions_7d` | Count of hard workouts |
| `days_since_last_hard_session` | Recovery context |
| `zone4_5_minutes_7d` | High-intensity accumulation |
| `strength_sessions_7d` | Muscular load |
| `leg_dominant_sessions_7d` | Optional manual tag |
| `training_monotony_7d` | Load consistency |
| `training_strain_7d` | Load × monotony |

Initial load formula:

```text
zone_weighted_hr_load =
  1*time_in_zone_1_min
  + 2*time_in_zone_2_min
  + 3*time_in_zone_3_min
  + 4*time_in_zone_4_min
  + 5*time_in_zone_5_min
```

A more customized load formula can later include activity type multipliers:

```text
activity_load =
  zone_weighted_hr_load
  × activity_type_multiplier
  × duration_factor
```

Example multipliers:

| Activity type | Multiplier |
|---|---:|
| Easy walk | 0.5 |
| Zone 2 run/cycle | 1.0 |
| Intervals | 1.4 |
| Strength | 1.2 |
| Karate/kicking session | 1.3 |
| Hard leg-dominant session | 1.5 |

---

### 8.2.3 `fg_realtime_state`

**Purpose:** latest state for online inference.  
**Primary key:** `user_id`  
**Event time:** `event_time`  
**Online enabled:** yes  
**Stream enabled:** yes

| Feature | Meaning |
|---|---|
| `event_time` | Latest state timestamp |
| `minutes_since_wake` | Circadian context |
| `steps_today_so_far` | Day activity |
| `active_minutes_today_so_far` | Activity accumulation |
| `stress_last_30m` | Current stress |
| `stress_last_2h` | Recent stress |
| `stress_zscore_vs_time_of_day` | Anomaly feature |
| `body_battery_current` | Current energy |
| `body_battery_slope_2h` | Depletion/recharge rate |
| `hr_current_or_recent` | Current HR |
| `hr_zscore_vs_time_of_day` | Anomaly feature |
| `last_activity_end_time` | Recent workout context |
| `hours_since_last_activity` | Recovery clock |
| `current_readiness_score` | Optional cached prediction |

---

### 8.2.4 `fg_recovery_episodes`

**Purpose:** post-workout recovery labels and recovery episode features.  
**Primary key:** `user_id`, `activity_id`  
**Event time:** `activity_end_time`  
**Online enabled:** optional; useful for active episodes

| Feature | Meaning |
|---|---|
| `activity_id` | Workout that started the episode |
| `activity_type` | Workout type |
| `activity_end_time` | Recovery clock start |
| `pre_workout_readiness_score` | Baseline before session |
| `activity_load` | Workout load |
| `max_hr_pct` | Intensity |
| `zone4_5_minutes` | High-intensity time |
| `body_battery_drop_during_activity` | Energy depletion |
| `rhr_pre_baseline` | Baseline before workout |
| `hrv_pre_baseline` | Baseline before workout |
| `recovered_at` | First timestamp/date recovered |
| `recovery_hours_label` | Regression label |
| `recovered_within_24h_label` | Binary label |
| `recovered_within_48h_label` | Binary label |
| `censored` | Useful for survival modeling |

Recovery label proposal:

```text
recovered_at = first morning after activity where:
  HRV >= 95% of personal 28d baseline
  AND resting_hr <= baseline + 3 bpm
  AND sleep_score not low
  AND body_battery_morning >= personal baseline - 10
```

This is not a medical definition of recovery. It is a useful proxy for model development.

---

### 8.2.5 `fg_manual_feedback`

**Purpose:** subjective labels and context.  
**Primary key:** `user_id`, `date`  
**Event time:** `submitted_at`  
**Online enabled:** yes

| Feature | Type |
|---|---:|
| `perceived_recovery_1_5` | int |
| `muscle_soreness_1_5` | int |
| `stress_subjective_1_5` | int |
| `motivation_1_5` | int |
| `pain_flag` | bool |
| `illness_flag` | bool |
| `trained_today` | bool |
| `planned_training_type` | string |
| `notes` | string |

Manual feedback is one of the most valuable additions. Garmin features give proxies; manual feedback gives personal ground truth.

---

# 9. Feature views

Feature views should define model-specific joins and avoid training/serving skew.

---

## 9.1 `fv_readiness_daily`

**Purpose:** train and serve the daily readiness classifier.

Joins:

```text
fg_garmin_daily_summary_raw
+ fg_garmin_sleep_raw
+ fg_recovery_baselines_daily
+ fg_training_load_daily
+ fg_realtime_state
+ fg_manual_feedback
```

Entity:

```text
user_id, date
```

Prediction time:

```text
Morning after sleep data is available
```

Targets:

```text
readiness_class ∈ {green, yellow, red}
readiness_score ∈ [0, 100]
hard_training_ok ∈ {0, 1}
```

Feature examples:

```text
sleep_duration_min
sleep_score
deep_sleep_min
rem_sleep_min
resting_hr
rhr_delta_28d
hrv_delta_28d_pct
body_battery_morning
body_battery_recharge
avg_stress_yesterday
stress_7d_mean
load_3d_sum
load_7d_sum
acute_chronic_load_ratio
days_since_last_hard_session
manual_soreness_1_5
manual_perceived_recovery_1_5
```

---

## 9.2 `fv_stress_anomaly_realtime`

**Purpose:** train and serve anomaly detection over current stress and physiological state.

Joins:

```text
fg_realtime_state
+ fg_recovery_baselines_daily
+ fg_training_load_daily
+ fg_garmin_sleep_raw
```

Entity:

```text
user_id, event_time
```

Target:

Usually no supervised target initially. Use anomaly score.

Feature examples:

```text
stress_last_30m
stress_last_2h
stress_zscore_vs_time_of_day
hr_zscore_vs_time_of_day
body_battery_slope_2h
minutes_since_wake
hours_since_last_activity
sleep_score_last_night
rhr_delta_28d
hrv_delta_28d_pct
```

Output:

```text
stress_anomaly_score
stress_anomaly_level ∈ {normal, mild, moderate, high}
```

---

## 9.3 `fv_recovery_time_after_workout`

**Purpose:** train the post-workout recovery-time predictor.

Joins:

```text
fg_garmin_activity_raw
+ fg_training_load_daily
+ fg_recovery_baselines_daily
+ fg_garmin_sleep_raw
+ fg_recovery_episodes
+ fg_manual_feedback
```

Entity:

```text
user_id, activity_id
```

Targets:

```text
recovery_hours_label
recovered_within_24h_label
recovered_within_48h_label
```

Feature examples:

```text
activity_type
duration_min
activity_load
avg_hr
max_hr
zone4_5_minutes
training_effect_aerobic
training_effect_anaerobic
body_battery_drop_during_activity
pre_workout_readiness_score
load_7d_sum_before_activity
acute_chronic_load_ratio_before_activity
sleep_score_previous_night
hrv_delta_before_activity
rhr_delta_before_activity
manual_soreness_before_activity
```

---

# 10. Models

## 10.1 Model A: daily training readiness classifier

### Prediction problem

```text
Input:
  latest recovery, sleep, stress, and training-load features

Output:
  green / yellow / red readiness class
```

### Initial weak-label strategy

Because there may be no labels at the beginning, use weak labels:

```text
red if:
  hrv_delta_28d_pct < -15%
  OR rhr_delta_28d > +7 bpm
  OR sleep_score < 50
  OR body_battery_morning < 35
  OR acute_chronic_load_ratio > 1.5
  OR manual pain/illness flag = true

green if:
  hrv_delta_28d_pct >= -5%
  AND rhr_delta_28d <= +3 bpm
  AND sleep_score >= 75
  AND body_battery_morning >= 65
  AND acute_chronic_load_ratio <= 1.2
  AND no pain/illness flag

else yellow
```

### Model choices

Recommended progression:

```text
Baseline: rule-based score
V1: Logistic Regression or Random Forest
V2: XGBoost or LightGBM
V3: calibrated XGBoost/LightGBM with SHAP explanations
```

### Output schema

```json
{
  "readiness_class": "yellow",
  "readiness_score": 63,
  "hard_training_ok_probability": 0.42,
  "main_factors": [
    "HRV 13% below baseline",
    "Resting HR 6 bpm above baseline",
    "High load in last 48h"
  ]
}
```

---

## 10.2 Model B: stress anomaly detector

### Prediction problem

```text
Input:
  current stress, HR, Body Battery slope, time of day, recent sleep/load

Output:
  anomaly score and explanation
```

### Model choices

Recommended progression:

```text
V0: rolling z-score by time-of-day bucket
V1: Isolation Forest
V2: robust covariance or one-class SVM
V3: temporal autoencoder, only if there is lots of data
```

Recommended MVP:

```text
stress_anomaly_score =
  weighted combination of:
    stress_zscore_vs_time_of_day
    hr_zscore_vs_time_of_day
    body_battery_depletion_zscore
    respiration_zscore
    sleep_recovery_penalty
```

Time-of-day baselines are important. Stress at 10:00 during work may be normal; the same stress at 03:00 during sleep is much more meaningful.

### Output schema

```json
{
  "stress_anomaly_level": "moderate",
  "stress_anomaly_score": 0.71,
  "main_factors": [
    "Stress is unusually high for this time of day",
    "Heart rate is elevated versus your baseline",
    "Body Battery is dropping faster than usual"
  ]
}
```

---

## 10.3 Model C: post-workout recovery time predictor

### Prediction problem

```text
Input:
  workout load + pre-workout readiness + recent load + sleep/recovery history

Output:
  predicted hours until recovered
```

### Model choices

Start with two models:

```text
Model C1: regression
  Target: recovery_hours_label
  Algorithm: LightGBM/XGBoost regressor

Model C2: classification
  Target: recovered_within_24h or recovered_within_48h
  Algorithm: calibrated classifier
```

Later, consider survival modeling:

```text
Cox model
Random survival forest
Gradient-boosted survival models
```

Survival modeling is useful because recovery labels can be censored. For example, you may not know exactly when you recovered if another hard workout happened before recovery was complete.

### Output schema

```json
{
  "predicted_recovery_hours": 31,
  "prob_recovered_24h": 0.38,
  "prob_recovered_48h": 0.81,
  "suggested_next_session": "light technique or mobility",
  "main_factors": [
    "High zone 4/5 time",
    "Low pre-workout HRV",
    "High 7-day accumulated load"
  ]
}
```

---

# 11. Combined decision layer

The three models should not produce conflicting user-facing recommendations. Add a product-level decision layer.

Example:

```text
final_training_status =
  max_risk(
    readiness_model,
    stress_anomaly_model,
    recovery_time_model
  )
```

Decision rules:

```text
green:
  readiness green
  AND stress anomaly normal/mild
  AND predicted recovery completed or < 8h remaining

yellow:
  readiness yellow
  OR stress anomaly moderate
  OR recovery remaining 8–24h

red:
  readiness red
  OR stress anomaly high
  OR recovery remaining > 24h
  OR pain/illness flag
```

Example final output:

```text
Status: Yellow

Recommendation:
Do mobility, easy zone 2, or technical karate practice today.
Avoid max-effort intervals or hard leg-dominant work.

Why:
- Your post-workout recovery model estimates 14h recovery remaining.
- HRV is still below baseline.
- Stress is normal, so this is mostly training-load related.
```

---

# 12. Pipelines

## 12.1 Pipeline 1: Garmin ingestion

Frequency:

```text
Every 15–60 minutes
```

Responsibilities:

```text
Fetch new Garmin data
Normalize JSON/FIT data
Write raw snapshots to object storage
Upsert raw feature groups
```

Implementation options:

```text
Python script
FastAPI worker
Kubernetes CronJob
Airflow DAG
```

Feature groups written:

```text
fg_garmin_daily_summary_raw
fg_garmin_sleep_raw
fg_garmin_activity_raw
fg_garmin_epoch_raw
```

---

## 12.2 Pipeline 2: daily baseline computation

Frequency:

```text
Once after sleep data is available
Optionally again at noon/evening
```

Feature groups written:

```text
fg_recovery_baselines_daily
fg_training_load_daily
```

---

## 12.3 Pipeline 3: real-time state computation

Frequency:

```text
Every sync event or every 15–30 minutes
```

Feature groups written:

```text
fg_realtime_state
```

This should write to online-enabled feature groups.

---

## 12.4 Pipeline 4: label generation

Frequency:

```text
Daily
After enough future data exists
```

Feature groups written:

```text
fg_recovery_episodes
fg_manual_feedback
```

Important: labels should be generated only when the necessary future window has passed. For example, do not assign `recovered_within_48h_label` until 48 hours after the workout.

---

# 13. Training pipelines

## 13.1 Readiness training pipeline

Frequency:

```text
Weekly, or manually during MVP
```

Data source:

```text
fv_readiness_daily
```

Evaluation:

```text
Time-based split
Macro F1
Class-weighted F1
Calibration curve
Confusion matrix
```

Avoid random splits because time-series leakage is easy.

---

## 13.2 Stress anomaly training pipeline

Frequency:

```text
Weekly/monthly baseline refresh
```

Data source:

```text
fv_stress_anomaly_realtime
```

Evaluation:

```text
Manual review of top anomalies
Precision@k
False positives per day
Overlap with poor-sleep / illness / high-stress manual notes
```

This model is harder to evaluate because it is often unsupervised.

---

## 13.3 Recovery time training pipeline

Frequency:

```text
Monthly at first
Weekly once there are many workouts
```

Data source:

```text
fv_recovery_time_after_workout
```

Evaluation:

```text
MAE for recovery hours
Median absolute error
Accuracy for recovered_within_24h
ROC-AUC / PR-AUC for recovered_within_48h
Calibration
```

---

# 14. Model registry

Register three separate models:

```text
readiness_classifier
stress_anomaly_detector
recovery_time_predictor
```

For each model, store:

```text
model artifact
feature view name/version
training dataset version
input schema
output schema
metrics
example input
SHAP/explanation config if used
```

---

# 15. Deployments

## 15.1 Deployment 1: `readiness-serving`

Input:

```json
{
  "user_id": "javier",
  "date": "2026-06-02"
}
```

Behavior:

```text
Retrieve latest feature vector from fv_readiness_daily
Run readiness model
Return readiness class + score + explanation
```

---

## 15.2 Deployment 2: `stress-anomaly-serving`

Input:

```json
{
  "user_id": "javier",
  "event_time": "now"
}
```

Behavior:

```text
Retrieve latest realtime state
Compute anomaly score
Return anomaly level + reason codes
```

---

## 15.3 Deployment 3: `recovery-time-serving`

Input:

```json
{
  "user_id": "javier",
  "activity_id": "latest"
}
```

Behavior:

```text
Retrieve activity and pre-activity features
Predict recovery hours
Return recovery estimate
```

---

## 15.4 Deployment 4: `training-decision-api`

This is the user-facing product API.

Input:

```json
{
  "user_id": "javier"
}
```

Output:

```json
{
  "overall_status": "yellow",
  "readiness": {
    "class": "yellow",
    "score": 63
  },
  "stress": {
    "level": "normal",
    "score": 0.22
  },
  "recovery": {
    "predicted_recovery_hours_remaining": 14
  },
  "recommendation": "Light technical training, mobility, or easy zone 2.",
  "avoid": [
    "Hard intervals",
    "Max-effort lower-body session"
  ],
  "main_reasons": [
    "HRV below baseline",
    "Workout recovery not complete",
    "High load in last 48h"
  ]
}
```

---

# 16. Online vs offline storage

## 16.1 Offline store

Use the offline store for:

```text
Historical Garmin summaries
Training data
Backtesting
Feature recomputation
Model training
Long-term analysis
```

## 16.2 Online store

Use the online store for:

```text
Latest daily recovery state
Latest real-time state
Latest training load features
Latest active recovery episode
Feature vectors for online inference
```

The design goal is to use the same feature definitions for training and serving to reduce training/serving skew.

---

# 17. Data validation

Garmin data can have missing fields depending on device, disabled settings, sync gaps, API limitations, and data-source choice.

Useful validation rules:

```text
resting_hr between 30 and 120
avg_stress between 0 and 100
body_battery_* between 0 and 100
sleep_duration_min between 0 and 900
spo2_avg between 70 and 100
duration_min > 0 for activities
activity_end_time > activity_start_time
event_time not null
user_id not null
```

Hopsworks supports data validation integrations and feature statistics. Relevant docs:

- Data validation: <https://docs.hopsworks.ai/latest/user_guides/fs/feature_group/data_validation/>
- Feature group statistics: <https://docs.hopsworks.ai/latest/concepts/fs/feature_group/fg_statistics/>

---

# 18. Monitoring

## 18.1 Data monitoring

Track:

```text
Missingness by feature
Garmin sync delays
Duplicate activity IDs
Feature freshness
Distribution shift
Validation failures
```

## 18.2 Model monitoring

Track:

```text
Readiness class distribution
Stress anomaly alerts per day
Recovery-time prediction error once labels mature
Manual feedback disagreement
Prediction drift
Feature drift
```

## 18.3 Feedback loop

Every morning, ask:

```text
How recovered do you feel? 1–5
Any soreness? 1–5
Any pain? yes/no
Do you plan to train today?
```

Every evening, ask:

```text
Did you train?
Was the recommendation useful?
Did you feel unusually stressed?
```

---

# 19. MVP roadmap

## MVP 1: historical batch system

Goal: get working training datasets.

Build:

```text
Fetch 90–180 days Garmin data
Create raw daily, sleep, and activity feature groups
Compute baselines and training load
Create readiness weak labels
Train readiness classifier
Create Streamlit dashboard
```

No online serving yet.

---

## MVP 2: online feature store + daily readiness

Goal: morning readiness prediction.

Build:

```text
Online-enabled feature groups
fv_readiness_daily
readiness model deployment
FastAPI endpoint
Daily prediction logging
```

---

## MVP 3: stress anomaly detector

Goal: intra-day anomaly detection.

Build:

```text
fg_realtime_state
time-of-day stress baselines
z-score anomaly detector
notification when anomaly is high
```

---

## MVP 4: post-workout recovery predictor

Goal: after each workout, estimate recovery time.

Build:

```text
fg_recovery_episodes
recovery label generator
recovery_time_predictor
post-workout prediction endpoint
```

---

# 20. Recommended initial tech stack

```text
Python
pandas / polars
garminconnect or official Garmin API client
hopsworks / hsfs / hsml
scikit-learn
xgboost or lightgbm
shap
FastAPI
Streamlit
Docker
cron / Airflow / GitHub Actions / Kubernetes CronJob
ntfy or Telegram for notifications
```

Because the target environment already includes Kubernetes, the eventual deployment could be:

```text
garmin-ingestor as CronJob
feature-pipeline as CronJob
daily-readiness-job as CronJob
realtime-api as Deployment
dashboard as Deployment
```

---

# 21. Key risks and mitigations

| Risk | Mitigation |
|---|---|
| Official Garmin API access may not be granted | Prototype with `python-garminconnect`; keep ingestion abstraction clean |
| Unofficial API wrapper may break | Store raw JSON snapshots; isolate Garmin client code |
| Labels are weak | Add manual feedback |
| Too little data | Start with rules + anomaly detection; train models after 90–180 days |
| Data leakage | Use event time and point-in-time joins |
| Too many false stress alerts | Use personal baselines and time-of-day normalization |
| Medical overclaiming | Frame as fitness/recovery support, not diagnosis |
| Missing HRV/Body Battery fields | Design features as optional and use fallback models |

---

# 22. Final recommended design

Build one unified system with three model services but one user-facing decision layer:

```text
Garmin data
  → Hopsworks raw feature groups
  → baseline/load/realtime derived feature groups
  → three feature views
  → three models
  → combined training decision API
  → dashboard + notification
```

Recommended build order:

1. Daily readiness classifier + dashboard.
2. Stress anomaly detector.
3. Post-workout recovery predictor.
4. Combined recommendation API.
5. Manual feedback and continuous retraining.

This order is practical because readiness can work with weak labels, stress anomaly can work unsupervised, and recovery-time prediction needs the most accumulated workout history.
