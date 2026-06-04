"""
app.py
------
Flask backend API for the Weather Prediction Web App.

HOW TO RUN:
  py -3.11 -m pip install flask flask-cors
  py -3.11 app.py

Then open your browser at: http://localhost:5000
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_rainfall, predict_temperature, predict_year
from src.data_loader import generate_sample_data, load_rainfall, load_temperature

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

STATIONS = ["INDORE", "BHOPAL", "JABALPUR", "GWALIOR"]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Serve frontend ────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


# ── API: Get all stations ─────────────────────────────────
@app.route("/api/stations")
def get_stations():
    return jsonify({"stations": STATIONS})


# ── API: Predict full year for a station ─────────────────
@app.route("/api/predict")
def predict():
    station = request.args.get("station", "INDORE").upper()
    year    = int(request.args.get("year", 2025))

    if station not in STATIONS:
        return jsonify({"error": f"Station '{station}' not found. Choose from {STATIONS}"}), 400
    if not (2020 <= year <= 2035):
        return jsonify({"error": "Year must be between 2020 and 2035"}), 400

    try:
        results = []
        for m in range(1, 13):
            rain = predict_rainfall(m, year, station)
            temp = predict_temperature(m, year, station)
            results.append({
                "month":       MONTH_NAMES[m - 1],
                "month_num":   m,
                "rainfall_mm": rain,
                "temperature": temp,
                "season":      _get_season(m)
            })

        # Summary stats
        temps  = [r["temperature"]  for r in results]
        rains  = [r["rainfall_mm"]  for r in results]
        summary = {
            "max_temp":      max(temps),
            "min_temp":      min(temps),
            "hottest_month": MONTH_NAMES[temps.index(max(temps))],
            "coldest_month": MONTH_NAMES[temps.index(min(temps))],
            "total_rainfall":round(sum(rains), 1),
            "peak_rain_month": MONTH_NAMES[rains.index(max(rains))],
        }

        return jsonify({
            "station":  station,
            "year":     year,
            "monthly":  results,
            "summary":  summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Compare all stations for a month ────────────────
@app.route("/api/compare")
def compare():
    month = int(request.args.get("month", 7))
    year  = int(request.args.get("year", 2025))

    data = []
    for station in STATIONS:
        rain = predict_rainfall(month, year, station)
        temp = predict_temperature(month, year, station)
        data.append({
            "station":     station,
            "rainfall_mm": rain,
            "temperature": temp
        })

    return jsonify({
        "month": MONTH_NAMES[month - 1],
        "year":  year,
        "data":  data
    })


def _get_season(month):
    if month in [12, 1, 2]:  return "Winter"
    if month in [3, 4, 5]:   return "Summer"
    if month in [6, 7, 8, 9]: return "Monsoon"
    return "Autumn"


if __name__ == "__main__":
    if not os.path.exists("models/rainfall_model.pkl"):
        print("Models not found — training now...")
        os.makedirs("data/raw", exist_ok=True)
        generate_sample_data()
        load_rainfall()
        load_temperature()
        from src.train import train_rainfall, train_temperature
        train_rainfall()
        train_temperature()

    print("\n🌦️  Weather Prediction Server starting...")
    print("📌  Open your browser at: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)