# Design Decisions — Garmin Companion ML System

**Status:** Agreed by the Hopsworks expert review panel (2026-06-04)
**Inputs:** the draft proposal (`garmin_hopsworks_recovery_ml_system_draft_report.md`) and a prior
Hopsworks-grounded review. **Panel:** Feature Store architect · MLOps/serving · ML data-correctness &
sports-science skeptic · Tutorial/DX reviewer. Every API claim below was verified against
`hopsworks-api` (`python/hsfs`, `python/hsml`) and the existing tutorials.

This document is the contract the notebooks implement. Each row is **ACCEPT / ACCEPT-WITH-MODS / REJECT**
with the rationale and the concrete implementation choice.

---

## 0. Two cross-cutting decisions

### Scope — *all three models, full E2E, one combined deployment*
The user requires readiness, stress-anomaly, and recovery-time models end-to-end. The DX reviewer's concern
about over-architecture is resolved **not** by dropping models but by **collapsing serving to a single
combined deployment** (see A8). So: three models, three (lightweight) feature views, three registered model
artifacts saved into **one** model directory, served by **one** KServe predictor that runs the §11 decision
layer. No FastAPI tier, no four-deployment fan-out.

### Data — *python-garminconnect is the real path; `DEMO_MODE` keeps notebooks runnable*
The user chose `python-garminconnect` over a synthetic generator. We honor that: `ingestion/garmin_client.py`
wraps `python-garminconnect` behind a swappable interface and is the documented production path. To avoid
hard-failing for a reader with no Garmin account (DX reviewer), a `DEMO_MODE` flag loads compact illustrative
sample data via `ingestion/sample_data.py`. This is **fixture/illustration data, not a physiology simulator**
and not the recommended path — it exists only so cells execute for learning. *Flagged for user veto.*

---

## 1. Feature-store & data-modeling (A-series)

| ID | Decision | Rationale & implementation |
|----|----------|----------------------------|
| **A1** | **ACCEPT-WITH-MODS** | Eliminating train/serve skew is right, but the panel corrected a key nuance: `TransformationStatistics` holds a **single fitted statistic from the training dataset**, so it expresses *training-set-fixed* standardization (e.g. scaling a feature for the model) — it **cannot** express a **trailing 28-day rolling baseline** that must reflect the last 28 days *at serving time*. Therefore: (a) **rolling baselines** (`rhr_28d_mean`, `hrv_28d_mean`, time-of-day means) are computed in `features/baselines.py` and **stored** in `fg_recovery_baselines_daily`; (b) the **delta vs. that stored baseline** (`rhr_delta_28d`, `hrv_delta_28d_pct`) is computed as an **on-demand transformation** (UDF attached to the feature group, recomputable at serving from stored baseline + current value); (c) genuine **model-input scaling** (standardize features for the classifier) is a **model-dependent `@udf` + TransformationStatistics** on the feature view. Per DX reviewer, default to **builtin transformations** (`label_encoder`, `min_max_scaler`) for the common cases and show **one** custom-`@udf` cell flagged "advanced", mirroring `fraud_online_experiments/0_setup.ipynb`. Note in markdown: MDT vs on-demand is decided by **where the UDF is attached** (FV vs FG), not by a flag; `mode=` only selects python/pandas execution. |
| **A2** | **ACCEPT** | Drop `current_readiness_score` as a feature — feeding a prior readiness output into the readiness model is a feedback/target leak. Drop the single-row overwritten `fg_realtime_state`. Intra-day windows (`stress_last_30m`, `body_battery_slope_2h`) are computed from the **request payload** as **on-demand features** via `get_feature_vector(..., request_parameters=...)`. Clarify in markdown: on-demand-from-request-payload vs online-lookup-then-aggregate are different implementations; we use request-payload on-demand. |
| **A3** | **ACCEPT** (DX reviewer's "skip" overruled by full-E2E scope) | Recovery labels are inherently label-driven + censored, so `fs.get_or_create_spine_group(...)` carrying `(user_id, activity_start_time, label, censored)` joined to feature FGs is the correct PIT mechanism. No existing tutorial shows spine groups, so notebook 2 gets a dedicated, heavily-commented markdown explanation introducing the concept gently. |
| **A4** | **ACCEPT-WITH-MODS** | Use feature-view `logging_enabled=True` + `fv.log(...)` **and** `InferenceLogger` — they are **complementary** (InferenceLogger = raw HTTP req/resp to Kafka; `fv.log` = structured feature-level rows to the offline store for monitoring). Correct `fv.log` is **keyword-only**: `fv.log(untransformed_features=[...], transformed_features=[...], predictions=[...], model=<hsml.Model>)` — `model` is an **hsml Model object**, not a string, and not positional. Wire via `fv.init_serving(feature_logger=async_logger)` where `async_logger = fv.create_feature_logger()` (only valid inside a deployment pod). Mirror `fraud_online/2_*` + `3_*` (`pause_logging()`→`materialize_log(wait=True)`→`read_log()`). |
| **A5** | **ACCEPT** | `ttl` / `ttl_enabled` / `online_disk` are real `create_feature_group` params. TTL is measured from **`event_time`**, so the epoch FG must have a proper `event_time`. Use `online_disk=True` + bounded `ttl` for the high-cardinality epoch window. State explicitly: **online TTL-bounded, offline append-only** — they are independent (ties to B5). |
| **A6** | **ACCEPT-WITH-MODS** | §17 rules → GE expectation suites with `validation_ingestion_policy` = `STRICT` (hard physiological bounds, keyed raw FGs) vs `ALWAYS` (soft/observational, ingest-and-flag). Keep it **light** (~3–5 expectations on 1–2 raw FGs), not all six FGs — mirror `fraud_online/1_*` (`ge.from_pandas(df).get_expectation_suite()` + a few `expect_column_values_to_be_between`). |
| **A7** | **ACCEPT** | Native Feature Monitoring confirmed end-to-end: `create_feature_monitoring`/`create_scheduled_statistics` on FG **and** FV, `compare_on(metric, threshold)`, `compare_on_distribution(metric="PSI", ...)` (also KL/JS/Wasserstein/Hellinger/KS), and a **`TRAINING_DATASET` reference window** for serving-time input drift. Use the **simplified** monitoring notebook as the template (`fraud_online/4_*_simplified.ipynb`) with demo-tuned crons + an idempotent cleanup loop. Point one detection window at the FV **logging feature group** for prediction drift. Add a **freshness expectation** (max event_time lag) and align the monitoring schedule to Garmin's bursty/delayed sync cadence to avoid false drift alarms on partially-arrived data. |
| **A8** | **DECIDED → single combined KServe deployment** | One deployment: a `Predict` class loads all three pickles from `MODEL_FILES_PATH`, does **one** `get_feature_vector({"user_id": ...})` round-trip (all three models share the same user context), runs each model, and combines via the §11 decision layer in `predict()`. Reasons: fewer hops / TLS handshakes, single cold-start surface (KServe scale-to-zero), local `try/except` partial-failure handling, and all three scores are needed before any recommendation anyway so independent autoscaling buys nothing. One model artifact dir **can** hold three pickles. Feature-view linkage happens at **`create_model(feature_view=...)`** time (provenance), **not** at `deploy()` (`model.deploy` has **no** `feature_view=` param). Use `MODEL_FILES_PATH` (not deprecated `ARTIFACT_FILES_PATH`). On SaaS, KServe default `num_instances=0` → set `num_instances=1` or document the 30–120 s cold start. `entry` dict keys must match the online FG **primary-key column names** exactly. |

### Feature-store gaps the panel surfaced (must be handled in the notebooks)
- **Join cardinality:** joining a daily FG to an intra-day epoch FG on `user_id` alone fans out (1 day × N epochs). Define the join granularity per FG; the spine/daily side drives the join. *Most likely silent bug.*
- **Prefix collisions:** three models pull overlapping features (`rhr`, `hrv`, `stress`) across FGs → identical names collide in a feature view. Use `Query.join(..., prefix=...)`.
- **PIT needs `event_time` on every participating FG**, not just the spine — a date-only daily FG won't PIT-join cleanly against an `activity_start_time` spine.
- **Online/offline schema divergence for on-demand features:** request-time features (A2) are not stored online; the training data must reconstruct them via the **same** UDF over historical inputs, or the offline TD is missing columns the online vector has. `fv.log` untransformed-capture (A4) is how we validate alignment (links A2↔A4).
- **C correction:** stream FGs do **not** require `event_time` at the API level — it is a strong best-practice for TTL/PIT/time-travel correctness. Reword accordingly.

---

## 2. ML correctness & sports-science (B-series + new traps)

| ID | Decision | Rationale & implementation |
|----|----------|----------------------------|
| **B1** | **ACCEPT-WITH-MODS** | Training a classifier on its own weak-label rule yields a slow lookup table; weak-label F1 is circular and meaningless. **Mods:** the learned model only adds value if it sees **richer features than the rule uses** (intraday HRV, load trends) — state this. Manual feedback (~tens of points) is **too sparse to retrain on**; use it for **evaluation/threshold calibration only**, reported as **Cohen's κ** (rule-vs-feedback agreement), **not** F1. The **rule-based scorer is presented as the product**; any weak-label metric carries a red-box caveat. |
| **B2** | **ACCEPT** | `recovered_at` and pre-workout baselines resolved **as-of `activity_start_time`** via the A3 spine + PIT join (not a hand-rolled groupby). **Freeze** the *label-definition* baseline strictly pre-activity (trailing 28d ending `activity_start_time − 1d`) — a baseline that moves as recovery progresses is both leakage and a degenerate target. **Separate** the *label-definition* baseline from any *feature* baseline. |
| **B3** | **ACCEPT** | Replace ACWR `7d_sum/28d_sum` (nested windows → mathematical coupling, spurious U-shape; Lolli/Impellizzeri) with **EWMA** acute (λ≈0.25) / chronic (λ≈0.07) per Williams 2017, chronic excluding the acute window if a ratio is kept. **Keep raw EWMA acute & chronic as separate features** alongside the ratio (the ratio discards level info). Document: warm-up (~first 28d unstable — don't ratio over warm-up), single-user means it's a **within-person trend**, frame as "less broken," not "validated injury oracle." |
| **B4** | **ACCEPT** | ~120 daily rows, heavily autocorrelated (effective N ≈ 20–40), weak labels → calibrated XGBoost+SHAP is theater. **Rules + robust z-score anomaly detector are the durable product.** ML is framed as "what you graduate to with the **multi-user** entity model." If boosted trees are shown, only on **pooled multi-user data with by-user grouped CV** — never single-user random CV. |
| **B5** | **ACCEPT** | Retain **offline, event-time-keyed history** for the stress-anomaly model — never overwrite the corpus it trains on. Implemented by keying `fg_stress_state` on `user_id` so the *online* store holds the latest state for serving while the *offline* store appends every insert (full history). The online copy is TTL-bounded (A5); the offline store is append-only. |
| **Survival** | **REJECT core / ACCEPT optional appendix** | Censoring is real and must be handled, but Cox/RSF over tens of mostly-censored episodes overfits. **Main path:** binary (`recovered_within_24h/48h`) + regression (`recovery_hours`) with **explicit censored-episode handling** (exclude episodes censored before the horizon from the denominator, or a separate "censored/unknown" class). **Optional appendix:** a descriptive Kaplan–Meier curve only. |
| **Stress eval** | **ACCEPT-WITH-MODS** | Precision@k + FP/day + overlap-with-notes is the right skeleton. **Add:** context-conditional baselines (exclude/condition on active periods — else "anomaly" = "exercised"); **robust MAD** z-scores (mean/SD is masked by the anomalies); **null-model lift** (Fisher/permutation vs base rate of poor-sleep nights); **event-level dedup** (a multi-hour episode is one event, not N samples); a **fixed alert budget** (≤1–2/week) to set k/threshold. |

### New correctness traps the panel added (incorporated into features + markdown)
- **T1 — Body Battery target leakage (CRITICAL):** Body Battery is *itself* a Garmin recovery composite (built from HRV, stress, sleep, activity). Using it as a **feature** to predict readiness is near-circular. **Decision:** exclude Body Battery from readiness **model features** (it may still be shown in the UI/explanation), or explicitly reframe the task as "explain/reproduce Body Battery." Same caution for **T8** Garmin "training load"/EPOC (also a model output) — document whether load is device-derived or computed from duration×intensity; we **compute** load ourselves in `features/training_load.py`.
- **T2 — Timezone/day-boundary:** daily summaries bucket by device-local midnight; travel/DST and midnight-crossing sleep corrupt `summary_date` and trailing windows. Pin a documented local-time convention.
- **T3 — Normalization leakage:** all scaler/z-score stats must be **trailing/as-of**, never full-history. This is exactly why A1 splits rolling-baseline (stored, as-of) from training-set-fixed scaling (MDT).
- **T4 — HRV missingness is MNAR:** missing HRV correlates with poor/short sleep, travel, illness — the days that matter most. **Do not mean-impute**; carry a **missingness indicator** and document MNAR. (Draft already treats HRV as optional — extend with an indicator.)
- **T5 — Manual-feedback selection bias:** feedback is logged non-randomly (extreme days), so agreement metrics are conditional on "days the user chose to rate." Report logging coverage.
- **T6 — Autocorrelation in ALL validation:** mandate **forward-chaining (time-series) CV with a ≥28-day embargo** everywhere — rule threshold tuning, anomaly eval, and any model — not just the XGBoost path.
- **T7 — Recovery-target circularity:** `recovered_at` is defined by HRV/RHR returning to baseline and HRV/RHR are also features → acceptable **only** if features are strictly pre-activity (B2). Flag the overlap.

---

## 3. SDK / naming corrections (C-series) — ACCEPT (one reword)
- `time_travel_format` default = **`"DELTA"`** ✓ · unified **`hopsworks`** package (not standalone `hsfs`/`hsml`) ✓ · **`mr.python.create_model(...)`** + `model.save(dir)` ✓ · `ModelSchema(input_schema=Schema(X), output_schema=Schema(y))` ✓ · `TRAINING_DATASET` reference window ✓ · version-pin doc links ✓.
- **Reword:** stream FGs "require event_time" → "**must set `event_time` for correctness**" (not API-enforced).
- **`model.deploy` has no `feature_view=` param** — linkage is at `create_model(feature_view=...)`. `deploy(name=...)` strips non-alphanumerics from the name.

---

## 4. Resulting build (what the notebooks implement)

```
garmin-companion/
├── README.md · DESIGN_DECISIONS.md · requirements.txt · (draft report kept)
├── ingestion/   garmin_client.py (python-garminconnect, swappable) · normalize.py · sample_data.py (DEMO_MODE)
├── features/    daily_summary · sleep (HRV + missingness indicator, T4) · activity · epoch ·
│                baselines (trailing/as-of, A1/T3) · training_load (EWMA, B3/T8) ·
│                recovery_episodes (PIT, frozen pre-activity baseline, A3/B2/T7) · weak_labels (B1)
├── transformations.py   on-demand delta UDFs (A1b) + optional model-dependent scaler @udf (A1c)
├── 1_garmin_feature_pipeline.ipynb     ingest (DEMO_MODE) · raw+derived FGs · GE suites (A6) · epoch TTL (A5)
├── 2_garmin_training_pipeline.ipynb    3 FVs (prefix joins, logging_enabled A4) · spine group (A3) ·
│                                       weak labels + κ-vs-feedback (B1) · 3 models · forward-chaining CV (T6) ·
│                                       register (one dir, A8) · censored handling (Survival)
├── 3_garmin_inference_pipeline.ipynb   single combined deployment (A8) · §11 decision layer · fv.log/read_log (A4)
├── 4_garmin_feature_monitoring.ipynb   FG/FV/PSI/TRAINING_DATASET + logging-FG drift + freshness (A7)
├── deployments/  predictor.py (combined Predict, A8) · transformer.py (only if used)
├── data/         small illustrative fixtures for DEMO_MODE
└── streamlit_garmin_app.py   optional Status/Recommendation/Why capstone (§1/§15.4)
```

**Reconciled tensions:** full-E2E three models (user) **+** single deployment (MLOps/DX) · spine groups kept
(correctness) but gently introduced (DX) · garminconnect primary (user) **+** DEMO_MODE fallback (DX) ·
advanced `@udf` shown once and flagged, builtins by default (DX/architect).
