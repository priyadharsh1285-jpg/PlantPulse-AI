"""Paths and user-editable settings (persisted to data/config.json)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PLANTPULSE_DATA", ROOT / "data"))
IMAGE_DIR = DATA_DIR / "images"
CONFIG_PATH = DATA_DIR / "config.json"
DB_PATH = DATA_DIR / "plantpulse.db"

DEFAULTS = {
    # Location for Open-Meteo. CHANGE THIS to your city (default: New Delhi).
    "latitude": 28.6139,
    "longitude": 77.2090,
    "camera_index": 0,
    # Region of the frame to analyse, as fractions: x0, y0, x1, y1.
    # Crop out the pot, soil and background so only leaves are analysed.
    "roi": [0.0, 0.0, 1.0, 0.7],
    # Soil probe calibration (raw ESP32 ADC, 0-4095). MUST be calibrated:
    # dry = probe in air / completely dry soil, wet = probe in a glass of water.
    "soil_dry_raw": 3200,
    "soil_wet_raw": 1400,
    # Sensor alert thresholds. Calibrate for your plant, pot and sensors.
    "soil_low_pct": 30,
    "temp_high_c": 35,
    "humidity_low_pct": 35,
    "lux_low": 1000,  # only checked between 09:00 and 16:00
    # Behaviour baseline
    "baseline_days": 4,
    "min_baseline_obs": 60,
    "contamination": 0.03,
    "alert_consecutive": 2,  # consecutive unusual observations before alerting
}


def load_config() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except json.JSONDecodeError:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
