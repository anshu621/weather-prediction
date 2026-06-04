"""
features.py
-----------
Turns raw cleaned data into ML-ready features.

Features created:
  - MONTH_SIN, MONTH_COS  → captures seasonal cycle
  - PREV_RAINFALL          → last month's rainfall
  - PREV_TEMP              → last month's temperature
  - ROLLING_3M_RAIN        → 3-month rolling average rainfall
  - IS_MONSOON             → 1 if Jun–Sep, else 0
  - YEAR_NORM              → year normalized (for trend detection)
"""

import pandas as pd
import numpy as np
import os

PROCESSED_DIR = "data/processed"


def add_time_features(df):
    """Add cyclical month encoding (better than raw month numbers for ML)."""
    df = df.copy()
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    return df


def add_lag_features(df, target_col, group_col="STATION"):
    """
    Add previous month's value as a feature.
    This helps the model learn from recent history.
    """
    df = df.copy()
    df = df.sort_values([group_col, "YEAR", "MONTH"])

    df[f"PREV_{target_col}"] = df.groupby(group_col)[target_col].shift(1)
    df[f"ROLLING_3M_{target_col}"] = (
        df.groupby(group_col)[target_col]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    return df


def add_season_flag(df):
    """Flag the monsoon months (June to September)."""
    df = df.copy()
    df["IS_MONSOON"] = df["MONTH"].isin([6, 7, 8, 9]).astype(int)
    df["IS_WINTER"] = df["MONTH"].isin([11, 12, 1, 2]).astype(int)
    df["IS_SUMMER"] = df["MONTH"].isin([3, 4, 5]).astype(int)
    return df


def encode_station(df):
    """Convert station names to numbers (Label Encoding)."""
    df = df.copy()
    if "STATION" in df.columns:
        df["STATION_CODE"] = pd.Categorical(df["STATION"]).codes
    return df


def normalize_year(df):
    """Normalize year so the model can detect long-term trends."""
    df = df.copy()
    if "YEAR" in df.columns:
        df["YEAR_NORM"] = (df["YEAR"] - df["YEAR"].min()) / (
            df["YEAR"].max() - df["YEAR"].min() + 1e-9
        )
    return df


def build_rainfall_features(df):
    """Full feature pipeline for rainfall prediction."""
    df = add_time_features(df)
    df = add_lag_features(df, "RAINFALL_MM")
    df = add_season_flag(df)
    df = encode_station(df)
    df = normalize_year(df)
    df.dropna(inplace=True)

    feature_cols = [
        "MONTH_SIN", "MONTH_COS",
        "PREV_RAINFALL_MM", "ROLLING_3M_RAINFALL_MM",
        "IS_MONSOON", "IS_WINTER", "IS_SUMMER",
        "STATION_CODE", "YEAR_NORM"
    ]
    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols


def build_temperature_features(df):
    """Full feature pipeline for temperature prediction."""
    df = add_time_features(df)
    df = add_lag_features(df, "TEMPERATURE_C")
    df = add_season_flag(df)
    df = encode_station(df)
    df = normalize_year(df)
    df.dropna(inplace=True)

    feature_cols = [
        "MONTH_SIN", "MONTH_COS",
        "PREV_TEMPERATURE_C", "ROLLING_3M_TEMPERATURE_C",
        "IS_MONSOON", "IS_WINTER", "IS_SUMMER",
        "STATION_CODE", "YEAR_NORM"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols


if __name__ == "__main__":
    print("Testing feature engineering...")

    # Load cleaned data
    rain_path = os.path.join(PROCESSED_DIR, "rainfall_clean.csv")
    temp_path = os.path.join(PROCESSED_DIR, "temperature_clean.csv")

    if os.path.exists(rain_path):
        rain_df = pd.read_csv(rain_path)
        rain_df, rain_feats = build_rainfall_features(rain_df)
        print(f"✅ Rainfall features: {rain_feats}")
        print(f"   Shape after features: {rain_df.shape}")

    if os.path.exists(temp_path):
        temp_df = pd.read_csv(temp_path)
        temp_df, temp_feats = build_temperature_features(temp_df)
        print(f"✅ Temperature features: {temp_feats}")
        print(f"   Shape after features: {temp_df.shape}")
