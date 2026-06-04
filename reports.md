# Build Report — Garmin Companion ML System

This report documents every bug, incompatibility, and operational difficulty found while
running the four FTI notebooks end-to-end and bringing the system up on the Hopsworks
cluster (project `garmin_companion`, Hopsworks SDK `5.0.0.dev1`, Python 3.13, pandas
2.3.3, scikit-learn 1.8.0). Each entry lists the symptom, the root cause, the fix that
was applied (if any), and whether it is a **code bug**, an **environment** issue, or a
**cluster limitation**.

**Outcome:** notebooks 1–3 run cleanly and the combined model is **deployed and serving
live recommendations**. Notebook 4 was run partially (two monitors created) and then
skipped at the user's request. See [Status by notebook](#status-by-notebook).

---

## How the notebooks were executed

The cluster venv (`/srv/hops/venv`) already has the Hopsworks stack, so the notebooks'
first cell (`!pip install -U 'hopsworks[...]' ...`) was skipped to avoid reinstalling the
SDK. Cells were executed in-process (state persisted across cells, magics/`!` lines
stripped) so failures surface per cell. All fixes below are committed into the notebooks
and supporting modules so a normal Jupyter run reproduces the working system.

---

## Status by notebook

| Notebook | Result | Notes |
|---|---|---|
| 1 · Feature pipeline | ✅ Pass | 8 feature groups created (raw + derived) after 2 fixes |
| 2 · Training pipeline | ✅ Pass | 3 FVs, 3 models, `garmin_combined` v1 registered after 3 fixes |
| 3 · Inference pipeline | ✅ Pass | KServe deployment `garminrecommendation` live; returns a recommendation |
| 4 · Feature monitoring | ⚠️ Partial / skipped | Scheduled-stats + threshold monitors created; PSI/prediction-drift unsupported (see #8/#9) — skipped per user request |

---

## Environment issues

### E1 — No `pip` in the cluster venv; `great_expectations` not preinstalled
- **Severity:** Medium (blocks notebook 1 imports)
- **Symptom:** `great_expectations` import fails; `pip`/`python -m pip` both report
  "No module named pip".
- **Root cause:** `/srv/hops/venv` ships without `pip`; packages are managed with `uv`.
  `requirements.txt` lists `hopsworks[python,great_expectations]` but GE was not present.
- **Fix:** Installed with `uv pip install --python /srv/hops/venv/bin/python
  'great_expectations<0.18'` → GE 0.17.23 (matches the 0.17.x API the notebook uses:
  `ge.core.ExpectationSuite`, `ExpectationConfiguration`).
- **Type:** Environment.

---

## Code bugs (fixed)

### B1 — TTL feature group + on-disk online storage rejected by NDB
- **Severity:** High (blocks notebook 1)
- **Where:** `1_garmin_feature_pipeline.ipynb`, cells creating `fg_garmin_epoch_raw` and
  `fg_stress_state`.
- **Symptom:** `RestAPIError 270067 … Table storage engine 'ndbcluster' does not support
  the create option 'TTL column can't be an on-disk column'`.
- **Root cause:** Both FGs were created with `online_disk=True` **and** `ttl_enabled=True`
  with a TTL keyed on `event_time`. RonDB/NDB does not allow the TTL index column to live
  on disk, so the `CREATE TABLE … TTL=…@event_time … STORAGE DISK` fails.
- **Fix:** Dropped `online_disk=True` on both FGs, keeping the TTL window **in-memory**
  (offline store still keeps full history). The design intent — a bounded online window —
  is preserved.
- **Type:** Code bug (incompatible FG option combination for this engine).

### B2 — pandas 2.x `groupby.apply` regression in baseline construction
- **Severity:** High (blocks notebook 1)
- **Where:** `features/baselines.py`, `build_baselines()`.
- **Symptom:** `ValueError: Length of values (1) does not match length of index (181)`.
- **Root cause:** `df.groupby("user_id", group_keys=False).apply(fn).values` where `fn`
  returns a **Series**. In pandas 2.3, `groupby.apply` of a Series-returning function
  **unstacks** the result into a wide `(1 × N)` frame, so `.values` has length 1 instead
  of the expected per-row Series.
- **Fix:** Rewrote the loop to use `grouped[src_col].transform(...)`, which returns a
  Series index-aligned to the source frame (length 181). Semantics (trailing rolling
  stat, `shift(1)` as-of) are unchanged.
- **Type:** Code bug (pandas-version-sensitive idiom).

### B3 — Label-free feature view `train_test_split` returns a 4-tuple
- **Severity:** Medium (blocks notebook 2 stress model)
- **Where:** `2_garmin_training_pipeline.ipynb`, stress-model cell.
- **Symptom:** `ValueError: too many values to unpack (expected 2)` on
  `Xs, _ = stress_fv.train_test_split(test_size=0.2)`.
- **Root cause:** The notebook assumed a label-free FV returns `(X_train, X_test)`. In SDK
  5.0 it always returns the 4-tuple `(X_train, X_test, y_train, y_test)`; the label halves
  are simply empty for an unsupervised view.
- **Fix:** Unpack `Xs, _, _, _ = stress_fv.train_test_split(...)`.
- **Type:** Code bug (SDK API assumption).

### B4 — Deployment `script_file` given as a local path
- **Severity:** High (blocks notebook 3 deployment)
- **Where:** `3_garmin_inference_pipeline.ipynb`, deploy cell.
- **Symptom:** `RestAPIError 240016 … Predictor script does not exist / Script not found`
  when calling `combined.deploy(script_file="deployments/predictor.py", …)`.
- **Root cause:** The serving backend resolves the predictor from **HopsFS**, not the
  local working directory. A bare relative path is not found server-side.
- **Fix:** Upload the script first via the dataset API and pass the absolute path:
  `dataset_api.upload("deployments/predictor.py", "Resources/deployments/garminrecommendation", overwrite=True)`
  → `script_file="/Projects/garmin_companion/Resources/deployments/garminrecommendation/predictor.py"`.
- **Type:** Code bug (notebook assumed a managed-Jupyter convenience that isn't available
  from a plain client).

---

## Cluster limitations (worked around, not fixable from the client)

### C1 — Prediction-logging feature groups cannot be created (`logging_enabled`)
- **Severity:** High (forces a feature to be disabled)
- **Where:** Any feature view created with `logging_enabled=True`
  (`fv_readiness_daily`, `fv_stress_anomaly_realtime`).
- **Symptom:** `RestAPIError 270001 … Error creating feature group table in the Hive
  Metastore: Add request failed : INSERT INTO 'COLUMNS_V2' …`.
- **Investigation:** Reproduced deterministically even for a **minimal 1-column**
  logging-enabled FV → prediction logging is broken cluster-wide on this Metastore, not a
  schema problem with these specific views. This is a server-side defect that cannot be
  patched from the client (and patching cluster internals is out of scope).
- **Workaround:**
  - Disabled `logging_enabled` on the feature views (notebook 2).
  - Guarded the predictor's `init_serving(feature_logger=…)` in `deployments/predictor.py`
    so a missing logging setup can never block pod startup (the `fv.log` call was already
    wrapped in try/except).
  - Made notebook 3's log-inspection cell degrade gracefully.
- **Impact:** The auxiliary prediction-logging / closed-loop-on-logged-FG path is
  unavailable here. Core serving and the notebook-4 statistics/threshold monitoring are
  unaffected. On a cluster where logging FGs work, re-enabling `logging_enabled` restores
  it with no other changes.
- **Type:** Cluster limitation.

### C2 — Intermittent Metastore race on `COLUMNS_V2` during FV / training-dataset creation
- **Severity:** Medium (flaky, retryable)
- **Symptom:** The same `270001 … INSERT INTO 'COLUMNS_V2'` error sometimes fires when
  creating a **non-logging** feature view (`fv_recovery_time_after_workout`) or other
  metadata tables, then **succeeds on retry**.
- **Root cause:** Appears to be an optimistic-locking / concurrency race in the Hive
  Metastore (DataNucleus) when tables are created in quick succession. Single-table
  operations fail occasionally; multi-table operations (logging FVs, see C1) fail
  consistently.
- **Workaround:** Retry the creation. All three FVs were created successfully this way;
  re-running the notebook is safe because every object uses `get_or_create_*` (idempotent)
  and FG inserts upsert by primary key.
- **Type:** Cluster limitation (transient).

---

## Known issues / observations (not blocking)

### O1 — Stress model returns `null` at serving time on the demo data
- **Severity:** Medium (degraded but non-fatal — recommendation still returns)
- **Symptom:** `deployment.predict(...)` returns
  `stress: {"level": "normal", "score": null}` — the stress branch silently fell back.
- **Root cause:** `fv_stress_anomaly_realtime` reads from `fg_stress_state`, which has a
  **1-day online TTL**. The bundled demo data's `event_time` is `2024-06-30`, i.e. ~2
  years before the current date (2026-06-04), so the online row has **expired** and
  `get_feature_vector({"user_id": "javier"})` returns `None`. The predictor catches the
  resulting error and defaults stress to `("normal", None)`. The readiness model serves
  fine because its feature groups have **no TTL**, so historical online rows persist.
- **Implication:** This is an artifact of replaying ~2-year-old fixture data against a
  *realtime* TTL'd feature group — not a code defect. With live Garmin data (recent
  `event_time`) or a longer/disabled TTL it serves normally. If a demo of the stress path
  is desired, either (a) shift the sample data's dates to "now", or (b) raise/disable the
  `fg_stress_state` online TTL.
- **Type:** Demo-data vs realtime-TTL mismatch.

### O2 — Feature monitoring cannot target on-demand transform outputs
- **Severity:** Low (notebook 4)
- **Symptom:** `ValueError: Invalid feature name … must be one of [...]` for
  `feature_name="d_resting_hr"`.
- **Root cause:** Feature monitoring validates against **stored** features. The readiness
  deltas (`rhr_delta_28d`, `hrv_delta_28d_pct`) are computed by request-time
  transformation functions and are not monitorable columns; also the name `d_resting_hr`
  never existed (the UDF output is `rhr_delta_28d`).
- **Fix applied:** Point the monitor at the stored source feature `resting_hr` (same drift
  signal, monitorable).
- **Type:** Code bug + API constraint.

### O3 — PSI / distribution drift comparison not available in this SDK
- **Severity:** Low (notebook 4 — skipped per user)
- **Symptom:** `AttributeError: 'FeatureMonitoringConfig' object has no attribute
  'compare_on_distribution'`.
- **Root cause:** SDK 5.0 exposes a single `compare_on(metric, threshold, strict,
  relative)` for descriptive-statistic differences (mean, etc.) against a reference
  window / value / training dataset. There is no PSI/Wasserstein distribution-drift
  comparator in this version.
- **Status:** Notebook 4 skipped at the user's request before this was finalized. To keep
  the "input drift vs training dataset" intent on this SDK, replace
  `compare_on_distribution(metric="PSI", …)` with `compare_on(metric="mean", …)` against
  the training-dataset reference window.
- **Type:** Cluster/SDK limitation.

### O4 — Readiness holdout accuracy varies between runs
- **Severity:** Informational
- **Observation:** Reported readiness "accuracy vs weak labels" varied (0.919 → 0.838)
  across runs because `train_test_split` is not seeded. This metric is explicitly flagged
  as circular in the notebook (the model re-learns the rule); the honest metric is Cohen's
  κ vs manual feedback (stable at ~0.568). No action needed.

---

## Summary of files changed

| File | Change |
|---|---|
| `1_garmin_feature_pipeline.ipynb` | Removed `online_disk=True` from epoch & stress-state FGs (B1) |
| `features/baselines.py` | `groupby.apply(...).values` → `groupby[col].transform(...)` (B2) |
| `2_garmin_training_pipeline.ipynb` | Disabled `logging_enabled` on FVs (C1); 4-tuple unpack for stress split (B3) |
| `3_garmin_inference_pipeline.ipynb` | Upload predictor to HopsFS + absolute path (B4); guarded log-inspection (C1) |
| `deployments/predictor.py` | Guarded `init_serving(feature_logger=…)` against missing logging (C1) |
| `4_garmin_feature_monitoring.ipynb` | PSI monitor → stored `resting_hr` (O2); PSI/`compare_on_distribution` unresolved (O3) |

All changes are minimal and preserve the original design intent; each is annotated inline
with the reason.
