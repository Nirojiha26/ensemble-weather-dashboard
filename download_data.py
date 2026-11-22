"""
Step 1: Download daily weather data for 6 Sri Lankan districts (2015-2025)
Source: Open-Meteo Historical Weather API (ERA5 reanalysis), free, no API key.
Output: sri_lanka_weather.csv

Robust version: retries on timeouts, downloads in small chunks,
and caches each district in raw_data/ so re-running skips finished ones.
"""
import os
import time
import requests
import pandas as pd

DISTRICTS = {
    "Colombo":      (6.9271, 79.8612),
    "Kandy":        (7.2906, 80.6337),
    "Galle":        (6.0535, 80.2210),
    "Jaffna":       (9.6615, 80.0255),
    "Anuradhapura": (8.3114, 80.4037),
    "Nuwara Eliya": (6.9497, 80.7891),
}

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

CHUNKS = [
    ("2015-01-01", "2017-12-31"),
    ("2018-01-01", "2020-12-31"),
    ("2021-01-01", "2023-12-31"),
    ("2024-01-01", "2025-12-31"),
]

URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE_DIR = "raw_data"


def fetch_chunk(lat, lon, start, end, retries=8):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Colombo",
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(URL, params=params, timeout=(30, 120))
            if r.status_code == 200:
                return pd.DataFrame(r.json()["daily"])
            print(f"    HTTP {r.status_code}, retry {attempt}/{retries}...")
        except requests.exceptions.RequestException as e:
            print(f"    {type(e).__name__}, retry {attempt}/{retries}...")
        time.sleep(min(10 * attempt, 60))
    raise RuntimeError(f"Could not download {start} to {end}. Check your internet and run again.")


def fetch_district(name, lat, lon):
    cache = os.path.join(CACHE_DIR, f"{name.replace(' ', '_')}.csv")
    if os.path.exists(cache):
        print(f"{name}: already downloaded, skipping")
        return pd.read_csv(cache)

    print(f"Downloading {name}...")
    parts = []
    for start, end in CHUNKS:
        print(f"  {start} to {end}")
        parts.append(fetch_chunk(lat, lon, start, end))
        time.sleep(2)
    df = pd.concat(parts, ignore_index=True).rename(columns={"time": "date"})
    df["district"] = name
    df["latitude"] = lat
    df["longitude"] = lon
    df.to_csv(cache, index=False)
    return df


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    frames = [fetch_district(n, lat, lon) for n, (lat, lon) in DISTRICTS.items()]

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    data.to_csv("sri_lanka_weather.csv", index=False)

    print("\nDone.")
    print("Shape:", data.shape)
    print(data.head())
    print("\nMissing values per column:")
    print(data.isna().sum())
