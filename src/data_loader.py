"""
data_loader.py
--------------
This script loads your raw CSV files, cleans them,
and saves the cleaned versions to data/processed/.

HOW TO USE:
  python src/data_loader.py
"""

import pandas as pd
import numpy as np
import os

# ── Paths ────────────────────────────────────────────────
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────
# SECTION 1: Load Rainfall Data
# ─────────────────────────────────────────────────────────
def load_rainfall(filename="imd_rainfall.csv"):
    """
    Loads and cleans the rainfall CSV.
    Expected columns (adjust if yours differ):
      YEAR, JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC
    OR
      YEAR, MONTH, RAINFALL (long format)
    """
    filepath = os.path.join(RAW_DIR, filename)

    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        print("   Please download rainfall data and place it in data/raw/")
        return None

    df = pd.read_csv(filepath)
    print(f"✅ Loaded rainfall data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}\n")

    # Rename columns to uppercase for consistency
    df.columns = [c.strip().upper() for c in df.columns]

    # ── Handle wide format (one column per month) ──
    month_cols = ["JAN","FEB","MAR","APR","MAY","JUN",
                  "JUL","AUG","SEP","OCT","NOV","DEC"]
    wide_cols = [c for c in month_cols if c in df.columns]

    if wide_cols and "YEAR" in df.columns:
        print("   Detected WIDE format → converting to long format...")
        df = df.melt(
            id_vars=[c for c in df.columns if c not in wide_cols],
            value_vars=wide_cols,
            var_name="MONTH_NAME",
            value_name="RAINFALL_MM"
        )
        month_map = {m: i+1 for i, m in enumerate(month_cols)}
        df["MONTH"] = df["MONTH_NAME"].map(month_map)
        df.drop(columns=["MONTH_NAME"], inplace=True)

    # ── Handle long format ──
    elif "RAINFALL" in df.columns or "RAINFALL_MM" in df.columns:
        print("   Detected LONG format ✓")
        if "RAINFALL" in df.columns:
            df.rename(columns={"RAINFALL": "RAINFALL_MM"}, inplace=True)

    # ── Clean up ──
    df["RAINFALL_MM"] = pd.to_numeric(df["RAINFALL_MM"], errors="coerce")
    df.dropna(subset=["RAINFALL_MM"], inplace=True)
    df["RAINFALL_MM"] = df["RAINFALL_MM"].clip(lower=0)   # No negative rain!

    # Encode MONTH as number if it's a string
    if "MONTH" in df.columns:
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")

    out_path = os.path.join(PROCESSED_DIR, "rainfall_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"   Saved to {out_path}\n")
    return df


# ─────────────────────────────────────────────────────────
# SECTION 2: Load Temperature Data
# ─────────────────────────────────────────────────────────
def load_temperature(filename="imd_temperature.csv"):
    """
    Loads and cleans the temperature CSV.
    Expected columns:
      YEAR, MONTH (or month name columns), MAX_TEMP, MIN_TEMP
    OR wide format with month columns.
    """
    filepath = os.path.join(RAW_DIR, filename)

    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        print("   Please download temperature data and place it in data/raw/")
        return None

    df = pd.read_csv(filepath)
    print(f"✅ Loaded temperature data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}\n")

    df.columns = [c.strip().upper() for c in df.columns]

    # Detect & convert wide format
    month_cols = ["JAN","FEB","MAR","APR","MAY","JUN",
                  "JUL","AUG","SEP","OCT","NOV","DEC"]
    wide_cols = [c for c in month_cols if c in df.columns]

    if wide_cols and "YEAR" in df.columns:
        print("   Detected WIDE format → converting to long format...")
        id_cols = [c for c in df.columns if c not in wide_cols]
        df = df.melt(id_vars=id_cols, value_vars=wide_cols,
                     var_name="MONTH_NAME", value_name="TEMPERATURE_C")
        month_map = {m: i+1 for i, m in enumerate(month_cols)}
        df["MONTH"] = df["MONTH_NAME"].map(month_map)
        df.drop(columns=["MONTH_NAME"], inplace=True)

    # Standardise temperature column name
    for possible in ["MAX_TEMP","MIN_TEMP","MEAN_TEMP","TEMPERATURE","TEMP"]:
        if possible in df.columns:
            df.rename(columns={possible: "TEMPERATURE_C"}, inplace=True)
            break

    df["TEMPERATURE_C"] = pd.to_numeric(df.get("TEMPERATURE_C", np.nan), errors="coerce")
    df.dropna(subset=["TEMPERATURE_C"], inplace=True)

    # Sanity check: Indian temps usually between -20°C and 55°C
    df = df[(df["TEMPERATURE_C"] > -20) & (df["TEMPERATURE_C"] < 55)]

    if "MONTH" in df.columns:
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")

    out_path = os.path.join(PROCESSED_DIR, "temperature_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"   Saved to {out_path}\n")
    return df


# ─────────────────────────────────────────────────────────
# SECTION 3: Generate Sample Data (for testing)
# ─────────────────────────────────────────────────────────
def generate_sample_data():
    """
    Creates realistic fake data so you can run the whole pipeline
    immediately, even before downloading real data.
    """
    print("📦 Generating sample data for testing...")
    os.makedirs(RAW_DIR, exist_ok=True)
    np.random.seed(42)

    years = list(range(2000, 2024))
    months = list(range(1, 13))
    stations = ["INDORE", "BHOPAL", "JABALPUR", "GWALIOR"]

    rows = []
    for station in stations:
        for year in years:
            for month in months:
                # Rainfall peaks in monsoon (Jun–Sep)
                monsoon_boost = 80 if month in [6, 7, 8, 9] else 5
                rainfall = max(0, np.random.normal(monsoon_boost, 20))

                # Temperature peaks in May, low in Dec–Jan
                base_temp = 25 + 12 * np.sin((month - 4) * np.pi / 6)
                temperature = base_temp + np.random.normal(0, 2)

                rows.append({
                    "STATION": station,
                    "YEAR": year,
                    "MONTH": month,
                    "RAINFALL_MM": round(rainfall, 1),
                    "TEMPERATURE_C": round(temperature, 1)
                })

    df = pd.DataFrame(rows)

    # Save as separate files
    rain_df = df[["STATION", "YEAR", "MONTH", "RAINFALL_MM"]]
    temp_df = df[["STATION", "YEAR", "MONTH", "TEMPERATURE_C"]]

    rain_df.to_csv(os.path.join(RAW_DIR, "imd_rainfall.csv"), index=False)
    temp_df.to_csv(os.path.join(RAW_DIR, "imd_temperature.csv"), index=False)

    print(f"   ✅ Saved sample data to data/raw/")
    print(f"   Replace these files with real IMD data when ready.\n")
    return df


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  WEATHER DATA LOADER")
    print("=" * 50)

    # If no files exist yet, generate sample data
    rain_path = os.path.join(RAW_DIR, "imd_rainfall.csv")
    temp_path = os.path.join(RAW_DIR, "imd_temperature.csv")

    if not os.path.exists(rain_path) or not os.path.exists(temp_path):
        generate_sample_data()

    rain_df = load_rainfall()
    temp_df = load_temperature()

    if rain_df is not None:
        print(f"Rainfall preview:\n{rain_df.head()}\n")
    if temp_df is not None:
        print(f"Temperature preview:\n{temp_df.head()}\n")

    print("✅ Data loading complete!")
