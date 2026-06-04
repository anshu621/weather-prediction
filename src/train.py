"""
train.py
--------
Trains ML models for rainfall and temperature prediction.
Saves trained models and evaluation charts.

HOW TO USE:
  python src/train.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # So it works without a display
import seaborn as sns
import os, sys, joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Add parent folder to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_rainfall, load_temperature, generate_sample_data
from src.features import build_rainfall_features, build_temperature_features

PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
OUTPUTS_DIR   = "outputs"

for d in [PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Optional: filter to a single station ──────────────────
# Set to None to use all stations
STATION_FILTER = None    # e.g. "INDORE"


# ─────────────────────────────────────────────────────────
# Utility: Evaluate & print model scores
# ─────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:30s}  R²={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    return {"model": name, "R2": round(r2, 4),
            "RMSE": round(rmse, 4), "MAE": round(mae, 4)}


# ─────────────────────────────────────────────────────────
# Utility: Plot Actual vs Predicted
# ─────────────────────────────────────────────────────────
def plot_predictions(y_test, y_pred, title, unit, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Scatter plot
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.4, s=20, color="#2196F3")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel(f"Actual ({unit})")
    ax.set_ylabel(f"Predicted ({unit})")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual plot
    ax = axes[1]
    residuals = y_test.values - y_pred
    ax.scatter(y_pred, residuals, alpha=0.4, s=20, color="#FF5722")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel(f"Predicted ({unit})")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Plot")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   Chart saved → {out_path}")


# ─────────────────────────────────────────────────────────
# Utility: Feature importance chart
# ─────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols, title, filename):
    if not hasattr(model, "feature_importances_"):
        return
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="barh", color="#4CAF50", ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   Chart saved → {out}")


# ─────────────────────────────────────────────────────────
# TRAIN RAINFALL MODEL
# ─────────────────────────────────────────────────────────
def train_rainfall():
    print("\n" + "="*50)
    print("  TRAINING: Rainfall Prediction")
    print("="*50)

    # Load data
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "rainfall_clean.csv"))
    if STATION_FILTER and "STATION" in df.columns:
        df = df[df["STATION"] == STATION_FILTER]
        print(f"  Filtered to station: {STATION_FILTER} ({len(df)} rows)")

    df, feature_cols = build_rainfall_features(df)
    print(f"  Features used: {feature_cols}")
    print(f"  Total samples: {len(df)}")

    X = df[feature_cols]
    y = df["RAINFALL_MM"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

    scores = []

    # Model 1: Linear Regression (baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    scores.append(evaluate("Linear Regression", y_test, lr.predict(X_test)))

    # Model 2: Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    scores.append(evaluate("Ridge Regression", y_test, ridge.predict(X_test)))

    # Model 3: Random Forest (usually best)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    scores.append(evaluate("Random Forest", y_test, rf.predict(X_test)))

    # Model 4: Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    scores.append(evaluate("Gradient Boosting", y_test, gb.predict(X_test)))

    # Pick best model by R²
    best_score = max(scores, key=lambda x: x["R2"])
    print(f"\n  🏆 Best model: {best_score['model']} (R²={best_score['R2']})")

    # Save best model
    model_map = {
        "Linear Regression":  lr,
        "Ridge Regression":   ridge,
        "Random Forest":      rf,
        "Gradient Boosting":  gb,
    }
    best_model = model_map[best_score["model"]]
    joblib.dump(best_model, os.path.join(MODELS_DIR, "rainfall_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "rainfall_features.pkl"))
    print(f"  ✅ Model saved → models/rainfall_model.pkl")

    # Plots
    y_pred = best_model.predict(X_test)
    plot_predictions(y_test, y_pred,
                     f"Rainfall Prediction ({best_score['model']})",
                     "mm", "rainfall_prediction.png")
    plot_feature_importance(best_model, feature_cols,
                            "Rainfall — Feature Importance",
                            "rainfall_feature_importance.png")

    return pd.DataFrame(scores)


# ─────────────────────────────────────────────────────────
# TRAIN TEMPERATURE MODEL
# ─────────────────────────────────────────────────────────
def train_temperature():
    print("\n" + "="*50)
    print("  TRAINING: Temperature Prediction")
    print("="*50)

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "temperature_clean.csv"))
    if STATION_FILTER and "STATION" in df.columns:
        df = df[df["STATION"] == STATION_FILTER]
        print(f"  Filtered to station: {STATION_FILTER} ({len(df)} rows)")

    df, feature_cols = build_temperature_features(df)
    print(f"  Features used: {feature_cols}")
    print(f"  Total samples: {len(df)}")

    X = df[feature_cols]
    y = df["TEMPERATURE_C"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

    scores = []

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    scores.append(evaluate("Linear Regression", y_test, lr.predict(X_test)))

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    scores.append(evaluate("Ridge Regression", y_test, ridge.predict(X_test)))

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    scores.append(evaluate("Random Forest", y_test, rf.predict(X_test)))

    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    scores.append(evaluate("Gradient Boosting", y_test, gb.predict(X_test)))

    best_score = max(scores, key=lambda x: x["R2"])
    print(f"\n  🏆 Best model: {best_score['model']} (R²={best_score['R2']})")

    model_map = {
        "Linear Regression":  lr,
        "Ridge Regression":   ridge,
        "Random Forest":      rf,
        "Gradient Boosting":  gb,
    }
    best_model = model_map[best_score["model"]]
    joblib.dump(best_model, os.path.join(MODELS_DIR, "temperature_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "temperature_features.pkl"))
    print(f"  ✅ Model saved → models/temperature_model.pkl")

    y_pred = best_model.predict(X_test)
    plot_predictions(y_test, y_pred,
                     f"Temperature Prediction ({best_score['model']})",
                     "°C", "temperature_prediction.png")
    plot_feature_importance(best_model, feature_cols,
                            "Temperature — Feature Importance",
                            "temperature_feature_importance.png")

    return pd.DataFrame(scores)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run data loader if processed files don't exist yet
    rain_ready = os.path.exists(os.path.join(PROCESSED_DIR, "rainfall_clean.csv"))
    temp_ready = os.path.exists(os.path.join(PROCESSED_DIR, "temperature_clean.csv"))

    if not rain_ready or not temp_ready:
        print("Processed data not found. Running data loader first...\n")
        os.makedirs("data/raw", exist_ok=True)
        generate_sample_data()
        load_rainfall()
        load_temperature()

    rain_scores  = train_rainfall()
    temp_scores  = train_temperature()

    # Save combined scores
    all_scores = pd.concat([
        rain_scores.assign(task="Rainfall"),
        temp_scores.assign(task="Temperature")
    ])
    score_path = os.path.join(OUTPUTS_DIR, "model_scores.csv")
    all_scores.to_csv(score_path, index=False)
    print(f"\n📊 All scores saved → {score_path}")
    print("\n✅ Training complete!")
