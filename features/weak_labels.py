"""Readiness weak labels + manual feedback — draft §8.2.5 / §10.1.

Panel guidance (B1) baked in:

* The weak-label **rule is the product** (V0). A classifier trained on these labels
  using the *same* inputs the rule uses just re-learns the rule — its weak-label F1 is
  circular and meaningless. The learned model is only worth training when it sees
  **richer features** than the rule and is evaluated on real manual feedback.
* Manual feedback is too sparse to retrain on; use it for **evaluation / threshold
  calibration**, reported as **Cohen's κ** (rule-vs-feedback agreement), not F1.
* Manual feedback is logged non-randomly (T5) — agreement is conditional on the days
  the user chose to rate.

The delta features below are computed with the *same formula* as the on-demand
transformations in :mod:`transformations`, so offline weak labels and online serving
agree.
"""

from __future__ import annotations

import random

import pandas as pd


def compute_readiness_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline deltas offline (mirrors the on-demand serving transforms).

    Expects raw values (``resting_hr``, ``hrv_avg_sleep``) joined to as-of baselines
    (``rhr_28d_mean``, ``hrv_28d_mean``).
    """
    out = df.copy()
    out["rhr_delta_28d"] = out["resting_hr"] - out["rhr_28d_mean"]
    out["hrv_delta_28d_pct"] = (
        (out["hrv_avg_sleep"] - out["hrv_28d_mean"]) / out["hrv_28d_mean"].replace(0, pd.NA) * 100
    )
    return out


def readiness_weak_label(row: pd.Series) -> str:
    """Rule-based readiness class (draft §10.1). Returns 'green' | 'yellow' | 'red'.

    NaN-tolerant: a missing signal simply doesn't trigger its clause.
    """
    def lt(x, t):
        return x is not None and pd.notna(x) and x < t

    def gt(x, t):
        return x is not None and pd.notna(x) and x > t

    def ge(x, t):
        return x is not None and pd.notna(x) and x >= t

    def le(x, t):
        return x is not None and pd.notna(x) and x <= t

    pain = bool(row.get("pain_flag", False)) or bool(row.get("illness_flag", False))

    red = (
        lt(row.get("hrv_delta_28d_pct"), -15)
        or gt(row.get("rhr_delta_28d"), 7)
        or lt(row.get("sleep_score"), 50)
        or lt(row.get("body_battery_morning"), 35)
        or gt(row.get("ewma_acwr"), 1.5)
        or pain
    )
    if red:
        return "red"

    # HRV is non-blocking for green when the device records none (this account has no
    # HRV). Missing HRV simply doesn't count against readiness — mirroring how the red
    # rule above already ignores absent signals. The remaining gates still apply.
    hrv_val = row.get("hrv_delta_28d_pct")
    hrv_ok = (hrv_val is None) or pd.isna(hrv_val) or ge(hrv_val, -5)

    green = (
        hrv_ok
        and le(row.get("rhr_delta_28d"), 3)
        and ge(row.get("sleep_score"), 75)
        and ge(row.get("body_battery_morning"), 65)
        and le(row.get("ewma_acwr"), 1.2)
        and not pain
    )
    return "green" if green else "yellow"


def add_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = compute_readiness_deltas(df)
    out["readiness_class"] = out.apply(readiness_weak_label, axis=1)
    score_map = {"green": 85, "yellow": 60, "red": 30}
    out["readiness_score"] = out["readiness_class"].map(score_map)
    out["hard_training_ok"] = (out["readiness_class"] == "green").astype(int)
    return out


# --- Manual feedback (demo) -------------------------------------------------------

MANUAL_FEEDBACK_COLUMNS = [
    "user_id", "date", "event_time",
    "perceived_recovery_1_5", "muscle_soreness_1_5", "stress_subjective_1_5",
    "motivation_1_5", "pain_flag", "illness_flag", "trained_today", "planned_training_type",
]


def generate_demo_manual_feedback(daily_with_labels: pd.DataFrame, seed: int = 7, rate: float = 0.4) -> pd.DataFrame:
    """Sparse, selection-biased subjective feedback for DEMO_MODE evaluation.

    Feedback is logged on ~``rate`` of days, biased toward notably good/bad days (T5).
    perceived_recovery correlates (noisily) with the rule, so κ is non-trivial but < 1.
    """
    rng = random.Random(seed)
    rows = []
    for _, r in daily_with_labels.iterrows():
        cls = r.get("readiness_class")
        extreme = cls in ("green", "red")
        if rng.random() > (rate + (0.3 if extreme else 0.0)):
            continue  # not logged this day (selection bias)
        base = {"green": 4, "yellow": 3, "red": 2}.get(cls, 3)
        perceived = max(1, min(5, base + rng.choice([-1, 0, 0, 1])))
        rows.append(
            {
                "user_id": r["user_id"],
                "date": r["date"],
                "event_time": pd.to_datetime(r["date"]) + pd.Timedelta(hours=7),
                "perceived_recovery_1_5": perceived,
                "muscle_soreness_1_5": max(1, min(5, 6 - perceived + rng.choice([-1, 0, 1]))),
                "stress_subjective_1_5": rng.randint(1, 5),
                "motivation_1_5": perceived,
                "pain_flag": bool(cls == "red" and rng.random() < 0.2),
                "illness_flag": bool(rng.random() < 0.03),
                "trained_today": bool(rng.random() < 0.5),
                "planned_training_type": rng.choice(["rest", "easy", "intervals", "strength"]),
            }
        )
    return pd.DataFrame(rows)


def perceived_to_class(perceived_1_5: int) -> str:
    """Map a 1–5 perceived-recovery rating to the readiness class space for κ."""
    if perceived_1_5 <= 2:
        return "red"
    if perceived_1_5 == 3:
        return "yellow"
    return "green"


def cohens_kappa(y1: list[str], y2: list[str], labels=("red", "yellow", "green")) -> float:
    """Cohen's κ between two categorical label lists (no sklearn dependency)."""
    n = len(y1)
    if n == 0:
        return float("nan")
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        if a in idx and b in idx:
            conf[idx[a]][idx[b]] += 1
    total = sum(sum(r) for r in conf)
    if total == 0:
        return float("nan")
    po = sum(conf[i][i] for i in range(k)) / total
    row = [sum(conf[i]) / total for i in range(k)]
    col = [sum(conf[i][j] for i in range(k)) / total for j in range(k)]
    pe = sum(row[i] * col[i] for i in range(k))
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")
