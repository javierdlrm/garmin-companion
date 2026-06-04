"""Training load & strain — ``fg_training_load_daily`` (draft §8.2.2).

Per the panel (B3), the classic acute:chronic workload ratio ``load_7d_sum /
load_28d_sum`` suffers mathematical coupling (the 7d window is nested in the 28d
window) and a spurious U-shape (Lolli/Impellizzeri). We use the **EWMA** acute/chronic
formulation (Williams et al. 2017) as the primary signal and keep the raw EWMA acute
and chronic loads as separate features (the ratio alone discards level information).
We still expose the rolling-sum ACWR for didactic comparison, clearly labelled.

All daily features are ``shift(1)`` (as-of yesterday) so the *morning* readiness
prediction never uses a workout that hasn't happened yet. The recovery model instead
reads load as-of ``activity_start_time`` via the spine join, not from this FG.

Note the warm-up caveat: the first ~28 days have an unstable chronic estimate; the
``acwr_*`` ratios are left NaN until ``min_periods`` is met.
"""

from __future__ import annotations

import pandas as pd

# EWMA spans (days). span = 2/alpha - 1; span 7 -> alpha .25, span 28 -> alpha .069.
ACUTE_SPAN = 7
CHRONIC_SPAN = 28
HARD_SESSION_ZONE45_MIN = 15.0  # a day with >=15 min in zone 4/5 counts as "hard"


def build_training_load(daily_load_full: pd.DataFrame) -> pd.DataFrame:
    """Build the daily training-load feature group.

    Args:
        daily_load_full: one row per (user_id, date) over the *full* calendar (rest
            days present with ``load=0``), with columns ``load``,
            ``zone4_5_minutes``, ``n_sessions``, ``n_strength``.
    """
    df = daily_load_full.sort_values(["user_id", "date"]).copy()
    g = df.groupby("user_id", group_keys=False)

    df["is_hard"] = (df["zone4_5_minutes"] >= HARD_SESSION_ZONE45_MIN).astype(int)

    def per_user(grp: pd.DataFrame) -> pd.DataFrame:
        load = grp["load"]
        out = pd.DataFrame(index=grp.index)
        out["load_1d"] = load.shift(1)
        out["load_3d_sum"] = load.rolling(3, min_periods=1).sum().shift(1)
        out["load_7d_sum"] = load.rolling(7, min_periods=1).sum().shift(1)
        out["load_28d_sum"] = load.rolling(28, min_periods=1).sum().shift(1)

        ewma_acute = load.ewm(span=ACUTE_SPAN, min_periods=ACUTE_SPAN).mean()
        ewma_chronic = load.ewm(span=CHRONIC_SPAN, min_periods=CHRONIC_SPAN).mean()
        out["ewma_acute_load"] = ewma_acute.shift(1)
        out["ewma_chronic_load"] = ewma_chronic.shift(1)
        out["ewma_acwr"] = (ewma_acute / ewma_chronic.replace(0, pd.NA)).shift(1)
        # Didactic only — the coupled rolling-sum ratio (see module docstring).
        out["acwr_rollingsum_naive"] = (
            out["load_7d_sum"] / (out["load_28d_sum"] / 4).replace(0, pd.NA)
        )

        out["hard_sessions_7d"] = grp["is_hard"].rolling(7, min_periods=1).sum().shift(1)
        out["zone4_5_minutes_7d"] = grp["zone4_5_minutes"].rolling(7, min_periods=1).sum().shift(1)
        out["strength_sessions_7d"] = grp["n_strength"].rolling(7, min_periods=1).sum().shift(1)

        # Monotony = mean/std of daily load over 7d; strain = 7d load * monotony.
        roll7 = load.rolling(7, min_periods=3)
        monotony = (roll7.mean() / roll7.std().replace(0, pd.NA)).shift(1)
        out["training_monotony_7d"] = monotony
        out["training_strain_7d"] = out["load_7d_sum"] * monotony

        # Days since last hard session.
        hard_idx = grp["is_hard"].to_numpy()
        days_since = []
        counter = None
        for h in hard_idx:
            days_since.append(counter if counter is not None else None)
            counter = 0 if h == 1 else (counter + 1 if counter is not None else None)
        out["days_since_last_hard_session"] = pd.Series(days_since, index=grp.index)
        return out

    feats = g.apply(per_user)
    result = pd.concat([df[["user_id", "date"]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    result["date"] = pd.to_datetime(result["date"]).dt.date
    result["event_time"] = pd.to_datetime(result["date"]) + pd.Timedelta(hours=8)
    return result
