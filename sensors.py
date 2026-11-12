"""Module 2 - Environmental Intelligence: ESP32 serial intake and a demo simulator.

The ESP32 prints one JSON object per line, e.g.
{"soil_raw":2100,"temp_c":28.4,"humidity_pct":55.1,"lux":1230.5,"pressure_hpa":1008.2}
(missing sensors are sent as null).
"""
import json
import time
from datetime import datetime

import numpy as np


class SensorError(RuntimeError):
    pass


def soil_pct(raw, dry, wet):
    """Convert raw ADC reading to 0-100 % using the two calibration points."""
    if raw is None or dry == wet:
        return None
    return float(np.clip((dry - raw) / (dry - wet) * 100.0, 0, 100))


def list_ports():
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [p.device for p in list_ports.comports()]


def parse_line(line, cfg):
    try:
        d = json.loads(line.strip())
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(d, dict) or "soil_raw" not in d:
        return None
    rec = {k: d.get(k) for k in ("soil_raw", "temp_c", "humidity_pct", "lux", "pressure_hpa")}
    rec["soil_pct"] = soil_pct(rec["soil_raw"], cfg["soil_dry_raw"], cfg["soil_wet_raw"])
    rec["source"] = "esp32"
    return rec


def read_serial(port=None, baud=115200, timeout=8.0, cfg=None):
    """Return the next valid reading from the ESP32. Auto-picks the first port if none is given."""
    try:
        import serial
    except ImportError as e:
        raise SensorError("pyserial is not installed. Run: pip install pyserial") from e
    if port is None:
        ports = list_ports()
        if not ports:
            raise SensorError("No serial ports found. Is the ESP32 plugged in (and its USB driver installed)?")
        port = ports[0]
    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            ser.reset_input_buffer()
            end = time.time() + timeout
            while time.time() < end:
                rec = parse_line(ser.readline().decode("utf-8", "ignore"), cfg)
                if rec:
                    return rec
    except serial.SerialException as e:
        raise SensorError(f"Serial error on {port}: {e}") from e
    raise SensorError(f"No valid data from {port} within {timeout:.0f}s. Check baud rate (115200) and wiring.")


def simulate(ts=None, rng=None, cfg=None):
    """Plausible indoor readings with a daily cycle, for demos and tests only."""
    ts = ts or datetime.now()
    rng = rng or np.random.default_rng()
    h = ts.hour + ts.minute / 60.0
    day = np.sin((h - 9) / 24 * 2 * np.pi)  # peaks ~15:00
    sun = max(0.0, np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0
    pct = float(np.clip(48 - 1.5 * day + rng.normal(0, 1.5), 0, 100))
    dry = cfg["soil_dry_raw"] if cfg else 3200
    wet = cfg["soil_wet_raw"] if cfg else 1400
    return {
        "soil_raw": float(dry - pct / 100 * (dry - wet)),
        "soil_pct": pct,
        "temp_c": float(27 + 6 * day + rng.normal(0, 0.4)),
        "humidity_pct": float(np.clip(60 - 15 * day + rng.normal(0, 1.5), 5, 100)),
        "lux": float(max(0, 5500 * sun + rng.normal(0, 80))),
        "pressure_hpa": float(1008 + rng.normal(0, 0.6)),
        "source": "simulated",
    }
