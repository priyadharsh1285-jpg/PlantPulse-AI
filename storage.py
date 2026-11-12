"""SQLite history of observations and care actions."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd

from . import config

OBS_COLS = [
    "ts", "image_path",
    "green_ratio", "yellow_ratio", "canopy_area", "aspect", "centroid_y", "scene_brightness", "motion",
    "soil_raw", "soil_pct", "temp_c", "humidity_pct", "lux", "pressure_hpa", "source",
]
_TEXT = {"ts", "image_path", "source"}


@contextmanager
def _conn():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init():
    cols = ", ".join(f"{c} {'TEXT' if c in _TEXT else 'REAL'}" for c in OBS_COLS)
    with _conn() as con:
        con.execute(f"CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
        con.execute("CREATE TABLE IF NOT EXISTS interventions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, note TEXT)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(ts)")


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.isoformat(sep=" ", timespec="seconds")
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (int, np.integer)):
        return float(v)
    return v


def add_observation(rec: dict) -> None:
    rec = dict(rec)
    rec.setdefault("ts", datetime.now())
    values = [_clean(rec.get(c)) for c in OBS_COLS]
    with _conn() as con:
        con.execute(
            f"INSERT INTO observations ({', '.join(OBS_COLS)}) VALUES ({', '.join('?' * len(OBS_COLS))})", values
        )


def load_observations(since=None) -> pd.DataFrame:
    init()
    with _conn() as con:
        df = pd.read_sql_query("SELECT * FROM observations ORDER BY ts", con)
    if df.empty:
        return pd.DataFrame(columns=["id"] + OBS_COLS).astype({"ts": "datetime64[ns]"})
    df["ts"] = pd.to_datetime(df["ts"])
    if since is not None:
        df = df[df["ts"] >= pd.Timestamp(since)]
    return df.reset_index(drop=True)


def add_intervention(note: str, ts=None) -> None:
    init()
    with _conn() as con:
        con.execute("INSERT INTO interventions (ts, note) VALUES (?, ?)", (_clean(ts or datetime.now()), note))


def load_interventions() -> pd.DataFrame:
    init()
    with _conn() as con:
        df = pd.read_sql_query("SELECT * FROM interventions ORDER BY ts", con)
    if df.empty:
        return pd.DataFrame({"id": [], "ts": pd.to_datetime([]), "note": []})
    df["ts"] = pd.to_datetime(df["ts"])
    return df
