"""Module 4 (experimental) - external weather context from Open-Meteo.

Open-Meteo is a forecast/analysis service, not a plant sensor. Cyclone, flood and
heavy-rain WARNINGS must come from the India Meteorological Department (IMD).
"""
import pandas as pd
import requests

API = "https://api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,relative_humidity_2m,surface_pressure,precipitation_probability,precipitation,cloud_cover"

IMD_LINKS = {
    "IMD home (official warnings)": "https://mausam.imd.gov.in/",
    "IMD cyclone information": "https://rsmcnewdelhi.imd.gov.in/",
}


class WeatherError(RuntimeError):
    pass


def fetch_weather(lat, lon, past_days=2, forecast_days=3, timeout=15):
    """Hourly DataFrame with a naive local-time `time` column (timezone=auto)."""
    params = {
        "latitude": lat, "longitude": lon, "hourly": HOURLY,
        "past_days": past_days, "forecast_days": forecast_days, "timezone": "auto",
    }
    try:
        r = requests.get(API, params=params, timeout=timeout)
        r.raise_for_status()
        hourly = r.json()["hourly"]
    except (requests.RequestException, KeyError, ValueError) as e:
        raise WeatherError(f"Could not fetch weather: {e}") from e
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def rain_summary(df, now=None, hours=24):
    now = now or pd.Timestamp.now()
    nxt = df[(df["time"] > now) & (df["time"] <= now + pd.Timedelta(hours=hours))]
    if nxt.empty:
        return {"max_probability": None, "total_mm": None}
    return {
        "max_probability": float(nxt["precipitation_probability"].max()),
        "total_mm": float(nxt["precipitation"].sum()),
    }
