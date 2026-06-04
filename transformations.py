"""Hopsworks transformation functions for the Garmin companion system.

The distinction below is the whole point of DESIGN_DECISIONS A1. **Where you attach a
UDF decides its type — there is no flag** (``mode=`` only picks python/pandas execution):

* **Baseline-delta / z-score transforms** are attached to the *feature view* via
  ``transformation_functions=[...]`` (so they are model-dependent and run identically
  at training and serving). They are pure **feature-arithmetic** over two *stored*
  columns — e.g. ``resting_hr`` minus the stored rolling baseline ``rhr_28d_mean`` —
  so they need **no** ``TransformationStatistics`` and sidestep the "a trailing 28-day
  baseline cannot be a single fitted statistic" trap. The rolling baseline lives as a
  stored feature (:mod:`features.baselines`); the delta is just current − baseline.

* **``standard_scaler``** is the contrasting case: model-input scaling whose mean/stddev
  *are* a single statistic fitted on the training dataset, so it uses
  ``TransformationStatistics`` and is skew-safe precisely because it is reused verbatim.

* **Genuinely request-time-only** features (e.g. ``stress_last_2h`` from a payload of
  recent epochs) are passed via ``request_parameters`` at serving — see
  :mod:`deployments.transformer`.

The delta formulas mirror :func:`features.weak_labels.compute_readiness_deltas` exactly
so offline weak labels and online serving never diverge.
"""

from __future__ import annotations

import pandas as pd

from hopsworks import udf
from hopsworks.hsfs.transformation_statistics import TransformationStatistics

# ---------------------------------------------------------------------------------
# Feature-arithmetic transforms (attach to the FEATURE VIEW; no statistics needed)
# ---------------------------------------------------------------------------------


@udf(return_type=float, mode="pandas")
def rhr_delta_28d(resting_hr: pd.Series, rhr_28d_mean: pd.Series) -> pd.Series:
    """Today's resting HR minus the trailing 28-day baseline (bpm)."""
    return resting_hr - rhr_28d_mean


@udf(return_type=float, mode="pandas")
def hrv_delta_28d_pct(hrv_avg_sleep: pd.Series, hrv_28d_mean: pd.Series) -> pd.Series:
    """HRV deviation from the trailing 28-day baseline (percent)."""
    safe = hrv_28d_mean.replace(0, pd.NA)
    return (hrv_avg_sleep - hrv_28d_mean) / safe * 100


@udf(return_type=float, mode="pandas")
def stress_zscore_vs_time_of_day(
    stress_level: pd.Series, stress_median: pd.Series, stress_mad: pd.Series
) -> pd.Series:
    """Robust (median/MAD) stress z-score for the epoch's hour-of-day bucket."""
    return (stress_level - stress_median) / stress_mad.replace(0, pd.NA)


@udf(return_type=float, mode="pandas")
def hr_zscore_vs_time_of_day(avg_hr: pd.Series, hr_median: pd.Series, hr_mad: pd.Series) -> pd.Series:
    """Robust (median/MAD) heart-rate z-score for the epoch's hour-of-day bucket."""
    return (avg_hr - hr_median) / hr_mad.replace(0, pd.NA)


# ---------------------------------------------------------------------------------
# Model-dependent transformation (attach to the FEATURE VIEW)
# ---------------------------------------------------------------------------------

_scaler_stats = TransformationStatistics("feature")


@udf(return_type=float, mode="pandas")
def standard_scaler(feature: pd.Series, statistics=_scaler_stats) -> pd.Series:
    """Standardize a model-input feature using TRAINING-SET statistics.

    Skew-safe as a model-dependent transform because the mean/stddev are fitted once
    on the training dataset and reused verbatim at serving — unlike a rolling baseline.
    """
    mean = statistics.feature.mean
    stddev = statistics.feature.stddev
    if stddev is None or stddev == 0:
        return feature.astype(float) - (mean or 0.0)
    return (feature.astype(float) - mean) / stddev
