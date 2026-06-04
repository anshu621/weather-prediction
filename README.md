# 🌦️ India Weather Prediction Project

A beginner-friendly Python ML project to predict **Rainfall** and **Temperature** using IMD (Indian Meteorological Department) data.

---

## 📁 Project Structure

```
weather_prediction/
├── data/
│   ├── raw/           ← Put your downloaded CSV files here
│   └── processed/     ← Cleaned data (auto-generated)
├── notebooks/
│   └── exploration.ipynb   ← Step-by-step Jupyter notebook
├── src/
│   ├── data_loader.py      ← Load & clean data
│   ├── features.py         ← Feature engineering
│   ├── train.py            ← Train ML models
│   └── predict.py          ← Make predictions
├── models/                 ← Saved trained models
├── outputs/                ← Charts, result CSVs
├── tests/                  ← Test scripts
├── requirements.txt
└── README.md
```

---

## 🌐 Data Sources

1. **IMD Extremes (Temperature & Rainfall by Station)**
   - URL: https://dsp.imdpune.gov.in/home_extremes.php
   - What to download: Select a station (e.g., INDORE [42754]) → Choose parameter → Download CSV

2. **Rainfall India (data.gov.in)**
   - URL: https://www.data.gov.in/catalog/rainfall-india
   - What to download: District-wise/subdivision-wise monthly rainfall CSV

---

## 🚀 Getting Started

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download Data
- Go to the IMD portal above
- Download CSVs and place them in `data/raw/`
- Example filenames:
  - `data/raw/imd_rainfall.csv`
  - `data/raw/imd_temperature.csv`

### Step 3 — Run the Pipeline
```bash
# Clean and prepare data
python src/data_loader.py

# Train models
python src/train.py

# Predict
python src/predict.py
```

### Step 4 — Explore in Jupyter
```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 🤖 Models Used

| Task | Model | Why |
|---|---|---|
| Rainfall Prediction | Random Forest Regressor | Handles non-linear patterns well |
| Temperature Prediction | Linear Regression + Random Forest | Simple baseline + better accuracy |

---

## 📊 Output

- `outputs/rainfall_prediction.png` — Actual vs Predicted rainfall chart
- `outputs/temperature_prediction.png` — Actual vs Predicted temperature chart
- `outputs/model_scores.csv` — R² and RMSE scores

---

## 💡 Tips for Beginners

- Start with the **notebook** — it explains every step
- If your CSV columns look different, edit `src/data_loader.py` to match
- The `STATION` variable in `src/train.py` lets you filter by city
