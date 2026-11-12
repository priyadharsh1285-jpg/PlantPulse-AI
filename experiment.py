"""Module 4 experiment: does plant appearance add information about coming rain?

Compares two classifiers with time-ordered cross-validation:
  sensors_only        local sensors + weather-service pressure trend + time of day
  sensors_plus_plant  the same, plus plant vision features
The question is whether the second beats the first. Time of day is in BOTH so the plant
cannot "win" just by encoding the daily cycle. Treat the result as exploratory.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from .anomaly import add_time_features

BASE = ["temp_c", "humidity_pct", "lux", "wx_pressure", "pressure_trend_3h", "hour_sin", "hour_cos"]
PLANT = BASE + ["aspect", "centroid_y", "green_ratio", "canopy_area"]


def build_dataset(obs, wx, horizon_h=3, now=None, rain_mm=0.1):
    """Label = measurable rain in the `horizon_h` hours AFTER each observation (past data only)."""
    now = now or pd.Timestamp.now()
    w = wx.sort_values("time").reset_index(drop=True).copy()
    w["wx_pressure"] = w["surface_pressure"]
    w["pressure_trend_3h"] = w["surface_pressure"] - w["surface_pressure"].shift(3)
    fut = w["precipitation"][::-1].rolling(horizon_h, min_periods=horizon_h).sum()[::-1].shift(-1)
    w["label"] = (fut > rain_mm).astype(float).where(fut.notna())
    w = w[w["time"] <= now - pd.Timedelta(hours=horizon_h)]
    df = obs.assign(time=obs["ts"].dt.floor("h")).merge(
        w[["time", "wx_pressure", "pressure_trend_3h", "label"]], on="time", how="inner"
    )
    return df.dropna(subset=["label"])


def evaluate(df, n_splits=5, min_rows=150, min_per_class=20):
    df = add_time_features(df.sort_values("ts")).dropna(subset=PLANT).reset_index(drop=True)
    y = df["label"].astype(int) if len(df) else pd.Series(dtype=int)
    pos, n = int(y.sum()), len(df)
    if n < min_rows or min(pos, n - pos) < min_per_class:
        return {
            "status": "insufficient",
            "n": n, "rain_cases": pos,
            "message": f"Need at least {min_rows} labelled observations with {min_per_class}+ rain and "
                       f"{min_per_class}+ no-rain cases (have {n}, {pos} rain). Keep logging through a rainy spell.",
        }
    results = {}
    for name, cols in {"sensors_only": BASE, "sensors_plus_plant": PLANT}.items():
        aucs = []
        for tr, te in TimeSeriesSplit(n_splits=n_splits).split(df):
            if y.iloc[tr].nunique() < 2 or y.iloc[te].nunique() < 2:
                continue
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced", random_state=0)
            clf.fit(df.iloc[tr][cols], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[te], clf.predict_proba(df.iloc[te][cols])[:, 1]))
        results[name] = {"auc_mean": float(np.mean(aucs)) if aucs else None, "folds": len(aucs)}
    return {"status": "ok", "n": n, "rain_cases": pos, "results": results,
            "note": "AUC 0.5 = no skill. A real gain needs plant > sensors across folds, not by a small margin."}
