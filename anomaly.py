"""Module 3 - Behaviour baseline, anomaly detection and rule-based sensor flags.

The baseline is learned from the first `baseline_days` of observations of ONE plant.
Isolation Forest scores how unusual a new observation is, given the time of day.
Output is "unusual - please inspect", never a disease diagnosis.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

VISION_FEATURES = ["green_ratio", "yellow_ratio", "canopy_area", "aspect", "centroid_y", "scene_brightness"]
MODEL_FEATURES = VISION_FEATURES + ["hour_sin", "hour_cos"]
LABELS = {
    "green_ratio": "green share", "yellow_ratio": "yellow share", "canopy_area": "canopy size",
    "aspect": "canopy height/width", "centroid_y": "leaf-mass position (droop)",
    "scene_brightness": "scene brightness",
}


def add_time_features(df):
    df = df.copy()
    h = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    return df


class PlantBaseline:
    def __init__(self, contamination=0.03, min_samples=60, random_state=0):
        self.contamination = contamination
        self.min_samples = min_samples
        self.random_state = random_state
        self.ready = False
        self.n_train = 0
        self.threshold = np.nan

    def fit(self, df):
        train = add_time_features(df.dropna(subset=VISION_FEATURES))
        self.n_train = len(train)
        if self.n_train < self.min_samples:
            self.ready = False
            return False
        X = self.scaler_fit(train[MODEL_FEATURES].to_numpy(float))
        self.model = IsolationForest(n_estimators=300, random_state=self.random_state).fit(X)
        train_scores = -self.model.score_samples(X)
        self.threshold = float(np.quantile(train_scores, 1 - self.contamination))
        self._fit_stats(train)
        self.ready = True
        return True

    def scaler_fit(self, X):
        self.scaler = StandardScaler().fit(X)
        return self.scaler.transform(X)

    def _fit_stats(self, train):
        def stats(d):
            med = d[VISION_FEATURES].median()
            mad = (d[VISION_FEATURES] - med).abs().median() * 1.4826
            return med, np.maximum(mad, np.maximum(med.abs() * 0.02, 1e-3))

        self.global_stats = stats(train)
        self.bucket_stats = {}
        buckets = train["ts"].dt.hour // 3
        for b in buckets.unique():
            sub = train[buckets == b]
            if len(sub) >= 8:
                self.bucket_stats[int(b)] = stats(sub)

    def explain(self, row):
        med, scale = self.bucket_stats.get(int(row["ts"].hour // 3), self.global_stats)
        z = (row[VISION_FEATURES].astype(float) - med) / scale
        top = z[z.abs() > 3].sort_values(key=lambda s: -s.abs()).head(2)
        if top.empty:
            return "overall pattern unusual"
        return "; ".join(f"{LABELS[k]} {'higher' if v > 0 else 'lower'} than usual (z={v:+.1f})" for k, v in top.items())

    def score(self, df):
        """Adds anomaly_score (higher = more unusual), vision_flag and drivers."""
        out = df.copy().reset_index(drop=True)
        out["anomaly_score"] = np.nan
        out["vision_flag"] = False
        out["drivers"] = ""
        if not self.ready or out.empty:
            return out
        feat = add_time_features(out)
        valid = feat[VISION_FEATURES].notna().all(axis=1)
        if not valid.any():
            return out
        X = self.scaler.transform(feat.loc[valid, MODEL_FEATURES].to_numpy(float))
        s = -self.model.score_samples(X)
        out.loc[valid, "anomaly_score"] = s
        out.loc[valid, "vision_flag"] = s > self.threshold
        out["vision_flag"] = out["vision_flag"].astype(bool)
        for i in out.index[out["vision_flag"]]:
            out.at[i, "drivers"] = self.explain(out.loc[i])
        return out


# ---------- sensor rules ----------

def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def sensor_flags(rec, cfg):
    flags = []
    soil, temp, hum, lux = (_num(rec.get(k)) for k in ("soil_pct", "temp_c", "humidity_pct", "lux"))
    if soil is not None and soil < cfg["soil_low_pct"]:
        flags.append("Low soil moisture")
    if temp is not None and temp > cfg["temp_high_c"]:
        flags.append("High temperature")
    if hum is not None and hum < cfg["humidity_low_pct"]:
        flags.append("Low humidity")
    ts = rec.get("ts")
    if lux is not None and ts is not None and 9 <= pd.Timestamp(ts).hour <= 16 and lux < cfg["lux_low"]:
        flags.append("Low light")
    return flags


def add_sensor_flags(df, cfg):
    out = df.copy()
    flags = [sensor_flags(r, cfg) for _, r in out.iterrows()]
    out["sensor_flags"] = ["; ".join(f) for f in flags]
    out["sensor_flagged"] = [bool(f) for f in flags]
    return out


def debounced_alert(flag_series, n=2):
    """True only if the last n observations were all flagged (cuts one-off false alarms)."""
    s = pd.Series(flag_series).astype(bool)
    return len(s) >= n and bool(s.iloc[-n:].all())


def recommendations(vision_alert, drivers, sflags):
    tips = []
    if "Low soil moisture" in sflags:
        tips.append("Soil is dry: check the pot and water if the top 2-3 cm of soil feels dry.")
    if "High temperature" in sflags:
        tips.append("It is hot: keep the plant away from hot glass and strong afternoon sun, and allow airflow.")
    if "Low humidity" in sflags:
        tips.append("Air is dry: avoid heaters/AC vents near the plant; check soil moisture more often.")
    if "Low light" in sflags:
        tips.append("Light is low for daytime: move the plant closer to a bright window (Tulsi likes several hours of sun).")
    if vision_alert:
        tips.append(
            "The plant looks different from its usual pattern. Inspect leaves, soil and the pot position. "
            "This is a prompt to check, not a diagnosis: drooping or yellowing can come from water stress, "
            "heat, light changes or a camera/lighting change."
        )
        if "yellow" in (drivers or ""):
            tips.append("More yellow than usual: check watering (too much or too little), old leaves and nutrients.")
        if not sflags:
            tips.append("No sensor threshold is breached, so also look at things the sensors cannot see "
                        "(pests, leaf spots, crowded roots) and confirm the camera view or lighting did not change.")
    return tips


def recovery_report(scored, t0, window_hours=24):
    """Did the plant return to its usual pattern after a care action at time t0?"""
    t0 = pd.Timestamp(t0)
    w = pd.Timedelta(hours=window_hours)
    pre = scored[(scored["ts"] <= t0) & (scored["ts"] > t0 - w)]
    post = scored[(scored["ts"] > t0) & (scored["ts"] <= t0 + w)]
    pre, post = pre[pre["anomaly_score"].notna()], post[post["anomaly_score"].notna()]
    if len(post) < 2:
        return {"status": "Waiting for more observations after this action", "n_post": len(post)}
    pre_rate = float(pre["vision_flag"].mean()) if len(pre) else float("nan")
    post_rate = float(post["vision_flag"].mean())
    latest_ok = not post["vision_flag"].iloc[-2:].any()
    if len(pre) and pre_rate > 0:
        status = "Returned to usual pattern" if latest_ok else "Not yet back to usual pattern"
    else:
        status = "No unusual pattern before this action (nothing to recover from)"
    return {"status": status, "pre_flag_rate": pre_rate, "post_flag_rate": post_rate, "n_post": len(post)}
