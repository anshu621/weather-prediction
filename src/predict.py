"""
predict.py
----------
Use your trained models to predict rainfall and temperature
for any station + month + year combination.

HOW TO USE:
  python src/predict.py

Or import and call predict() from your own script.
"""

import joblib
import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR    = "models"
PROCESSED_DIR = "data/processed"


# ─────────────────────────────────────────────────────────
# Helper: build a single-row feature vector
# ─────────────────────────────────────────────────────────
def _make_feature_row(month, year, station_code,
                      prev_val, rolling_3m_val,
                      feature_cols,
                      year_min=2000, year_max=2023):
    """
    Builds one row matching the feature columns used during training.
    """
    row = {
        "MONTH_SIN":  np.sin(2 * np.pi * month / 12),
        "MONTH_COS":  np.cos(2 * np.pi * month / 12),
        "IS_MONSOON": int(month in [6, 7, 8, 9]),
        "IS_WINTER":  int(month in [11, 12, 1, 2]),
        "IS_SUMMER":  int(month in [3, 4, 5]),
        "STATION_CODE": station_code,
        "YEAR_NORM":  (year - year_min) / max(year_max - year_min, 1),
    }

    # Fill lag features with whatever names exist
    for col in feature_cols:
        if "PREV_" in col and col not in row:
            row[col] = prev_val
        if "ROLLING_" in col and col not in row:
            row[col] = rolling_3m_val

    # Return only the columns the model expects, in order
    return pd.DataFrame([[row.get(c, 0) for c in feature_cols]], columns=feature_cols)


# ─────────────────────────────────────────────────────────
# Predict rainfall
# ─────────────────────────────────────────────────────────
def predict_rainfall(month: int, year: int,
                     station: str = "INDORE",
                     prev_rainfall: float = None,
                     rolling_3m: float = None) -> float:
    """
    Predict rainfall (mm) for a given month/year/station.

    Args:
        month:         1–12
        year:          e.g. 2024
        station:       Station name matching your training data
        prev_rainfall: Last month's actual rainfall (optional)
        rolling_3m:    3-month rolling average (optional)

    Returns:
        Predicted rainfall in mm
    """
    model_path    = os.path.join(MODELS_DIR, "rainfall_model.pkl")
    features_path = os.path.join(MODELS_DIR, "rainfall_features.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Rainfall model not found. Run train.py first.")

    model        = joblib.load(model_path)
    feature_cols = joblib.load(features_path)

    # Get station code from training data if available
    station_code = _get_station_code(station)

    # Default lag features: use monsoon average as fallback
    if prev_rainfall is None:
        prev_rainfall = 80 if month in [6, 7, 8, 9] else 10
    if rolling_3m is None:
        rolling_3m = prev_rainfall

    X = _make_feature_row(month, year, station_code,
                          prev_rainfall, rolling_3m, feature_cols)
    pred = model.predict(X)[0]
    return max(0.0, round(pred, 2))   # Rain can't be negative


# ─────────────────────────────────────────────────────────
# Predict temperature
# ─────────────────────────────────────────────────────────
def predict_temperature(month: int, year: int,
                        station: str = "INDORE",
                        prev_temp: float = None,
                        rolling_3m: float = None) -> float:
    """
    Predict temperature (°C) for a given month/year/station.
    """
    model_path    = os.path.join(MODELS_DIR, "temperature_model.pkl")
    features_path = os.path.join(MODELS_DIR, "temperature_features.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Temperature model not found. Run train.py first.")

    model        = joblib.load(model_path)
    feature_cols = joblib.load(features_path)

    station_code = _get_station_code(station)

    if prev_temp is None:
        # Rough seasonal default for central India
        prev_temp = 25 + 12 * np.sin((month - 4) * np.pi / 6)
    if rolling_3m is None:
        rolling_3m = prev_temp

    X = _make_feature_row(month, year, station_code,
                          prev_temp, rolling_3m, feature_cols)
    pred = model.predict(X)[0]
    return round(pred, 2)


# ─────────────────────────────────────────────────────────
# Helper: get station code from saved training data
# ─────────────────────────────────────────────────────────
def _get_station_code(station_name: str) -> int:
    path = os.path.join(PROCESSED_DIR, "rainfall_clean.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "STATION" in df.columns:
            df["STATION_CODE"] = pd.Categorical(df["STATION"]).codes
            match = df[df["STATION"] == station_name.upper()]
            if len(match) > 0:
                return int(match["STATION_CODE"].iloc[0])
    return 0   # Default if station not found


# ─────────────────────────────────────────────────────────
# Predict a full year (all 12 months)
# ─────────────────────────────────────────────────────────
def predict_year(year: int, station: str = "INDORE"):
    """Predict both rainfall and temperature for all 12 months of a year."""
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    results = []
    for m in range(1, 13):
        rain = predict_rainfall(m, year, station)
        temp = predict_temperature(m, year, station)
        results.append({
            "Month":          month_names[m-1],
            "Rainfall (mm)":  rain,
            "Temperature (°C)": temp
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────
# MAIN — Interactive demo
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  WEATHER PREDICTION")
    print("=" * 50)

    # Check models exist
    if not os.path.exists(os.path.join(MODELS_DIR, "rainfall_model.pkl")):
        print("⚠️  Models not found! Running training first...\n")
        from train import train_rainfall, train_temperature
        from data_loader import load_rainfall, load_temperature, generate_sample_data
        os.makedirs("data/raw", exist_ok=True)
        generate_sample_data()
        load_rainfall()
        load_temperature()
        train_rainfall()
        train_temperature()

    # ── Single prediction ──
    station = "INDORE"
    month   = 7      # July
    year    = 2025

    rain = predict_rainfall(month, year, station)
    temp = predict_temperature(month, year, station)

    month_name = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"][month - 1]

    print(f"\n📍 Station : {station}")
    print(f"📅 Period  : {month_name} {year}")
    print(f"🌧️  Rainfall: {rain} mm")
    print(f"🌡️  Temp    : {temp} °C")

    # ── Full year prediction ──
    print(f"\n📅 Full year prediction for {station} — {year}:\n")
    yearly = predict_year(year, station)
    print(yearly.to_string(index=False))

    out_path = os.path.join("outputs", f"prediction_{station}_{year}.csv")
    yearly.to_csv(out_path, index=False)
    print(f"\n✅ Saved yearly prediction → {out_path}")
