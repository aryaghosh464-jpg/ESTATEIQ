# -*- coding: utf-8 -*-
"""
ESTATEIQ — Flask Web Application
==================================
Routes:
  GET  /                      Home dashboard
  GET  /predict               Predictor form
  POST /predict               AJAX prediction (JSON)
  GET  /analytics             Analytics & EDA page
  GET  /api/locations         JSON list of known locations
  GET  /api/stats             JSON dataset summary
  GET  /api/chart/price-dist  JSON price histogram data
  GET  /api/chart/location-prices  JSON top-20 location prices
  GET  /api/chart/price-area  JSON scatter sample
  GET  /plots/<filename>      Serve EDA PNG plots

Prediction logic is imported directly from predict.py (no duplication).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# Import prediction helpers from predict.py (single source of truth)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent))

from predict import (
    build_feature_row, predict_price,
    LOCATION_ENCODING, GLOBAL_MEAN_PRICE,
    FEATURE_COLS, AMENITY_COLS,
    TIER_HIGH_CUT, TIER_LOW_CUT,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(_BASE_DIR / "templates"),
    static_folder=str(_BASE_DIR / "static"),
)
app.secret_key = "estateiq-dev-2024"

# Pre-load dataset stats once at startup
_DATA_PATH   = _BASE_DIR / "data/processed/Kolkata_features.csv"
_PLOTS_DIR   = _BASE_DIR / "data/plots"
_MODELS_DIR  = _BASE_DIR / "models"

try:
    _df = pd.read_csv(_DATA_PATH)
    _loc_enc_df = pd.read_csv(_MODELS_DIR / "location_target_encoding.csv")
    _KNOWN_LOCATIONS = sorted(_loc_enc_df["Location"].tolist())
    _GLOBAL_MEAN_PRICE = float(_loc_enc_df["MeanPrice"].mean())
    _TOTAL_PROPS = len(_df)
    _AVG_PRICE   = float(_df["Price"].mean())
    _MIN_PRICE   = float(_df["Price"].min())
    _MAX_PRICE   = float(_df["Price"].max())
    _NUM_LOCS    = int(_df["Location"].nunique())
    print(f"[ESTATEIQ] Dataset loaded: {_TOTAL_PROPS} properties, {_NUM_LOCS} locations")
except Exception as e:
    print(f"[WARN] Could not load dataset: {e}")
    _df = None
    _KNOWN_LOCATIONS = []
    _GLOBAL_MEAN_PRICE = 6_138_988
    _TOTAL_PROPS = 6275
    _AVG_PRICE   = 6_138_988
    _MIN_PRICE   = 2_000_000
    _MAX_PRICE   = 14_800_000
    _NUM_LOCS    = 308

# Read evaluation report
_eval_report = {}
try:
    report_text = (_MODELS_DIR / "evaluation_report.txt").read_text(encoding="utf-8")
    for line in report_text.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            _eval_report[k.strip()] = v.strip()
except Exception:
    pass

# Top locations for dashboard
_TOP_LOCATIONS = []
try:
    loc_avg = _df.groupby("Location")["Price"].agg(["mean", "count"])
    loc_avg = loc_avg[loc_avg["count"] >= 5].sort_values("mean", ascending=False).head(10)
    _TOP_LOCATIONS = [
        {"name": loc, "avg_price": int(row["mean"]), "count": int(row["count"])}
        for loc, row in loc_avg.iterrows()
    ]
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_tier(price: float) -> dict:
    """Return tier label and CSS colour for a predicted price."""
    if price >= 10_000_000:
        return {"label": "Luxury Estate", "color": "gold", "icon": "sparkles"}
    elif price >= 5_000_000:
        return {"label": "Premium Residence", "color": "champagne", "icon": "building"}
    elif price >= 3_000_000:
        return {"label": "Classic Home", "color": "taupe", "icon": "home"}
    else:
        return {"label": "Essential Property", "color": "stone", "icon": "layers"}


def _fmt_inr(val: float) -> str:
    """Format as ₹X.X L or ₹X.X Cr."""
    if val >= 10_000_000:
        return f"₹{val/10_000_000:.2f} Cr"
    return f"₹{val/100_000:.1f} L"


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    stats = {
        "total_props":  f"{_TOTAL_PROPS:,}",
        "avg_price":    _fmt_inr(_AVG_PRICE),
        "num_locations": f"{_NUM_LOCS}",
        "within_20pct": "25.4%",
        "min_price":    _fmt_inr(_MIN_PRICE),
        "max_price":    _fmt_inr(_MAX_PRICE),
    }
    return render_template("index.html", stats=stats, top_locations=_TOP_LOCATIONS)


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template(
        "predict.html",
        locations=_KNOWN_LOCATIONS,
        amenity_cols=AMENITY_COLS,
        mape=48.74,
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    """AJAX endpoint — returns JSON prediction result."""
    try:
        data = request.get_json(force=True)

        area      = max(1, int(data.get("area", 1000)))
        bedrooms  = max(1, int(data.get("bedrooms", 2)))
        location  = str(data.get("location", "")).strip()
        resale    = int(bool(data.get("resale", False)))
        amenities = {k: int(bool(v)) for k, v in data.get("amenities", {}).items()}

        df_row = build_feature_row(area, bedrooms, location, resale, amenities)
        price  = predict_price(df_row)

        # Clamp to realistic dataset range (avoid extreme extrapolation)
        price = max(_MIN_PRICE * 0.5, min(price, _MAX_PRICE * 1.5))

        # MAPE-based uncertainty range
        mape_frac = 0.4874
        low  = price * (1 - mape_frac)
        high = price * (1 + mape_frac)

        tier = _price_tier(price)

        # Location info
        loc_known = location in LOCATION_ENCODING
        loc_avg   = LOCATION_ENCODING.get(location, _GLOBAL_MEAN_PRICE)

        return jsonify({
            "success":   True,
            "price":     round(price),
            "price_fmt": _fmt_inr(price),
            "price_cr":  round(price / 10_000_000, 2),
            "price_l":   round(price / 100_000, 1),
            "per_sqft":  round(price / area),
            "range_low":  _fmt_inr(max(0, low)),
            "range_high": _fmt_inr(high),
            "tier":       tier,
            "loc_known":  loc_known,
            "loc_avg_fmt": _fmt_inr(loc_avg),
            "mape":       48.74,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/analytics")
def analytics():
    # Collect available plot files
    plots = []
    plot_meta = {
        "01_price_distribution.png":        ("Price Distribution", "Histogram of property prices"),
        "02_price_boxplot.png":             ("Price Boxplot", "Spread and outliers in Price"),
        "04_price_per_sqft_distribution.png": ("Price / sq.ft", "Distribution of price per square foot"),
        "05_price_vs_area.png":             ("Price vs Area", "Scatter: Area vs Price"),
        "07_price_by_bedrooms.png":         ("Price by Bedrooms", "How price varies with bedroom count"),
        "08_top15_locations_avg_price.png": ("Top 15 Locations", "Locations with highest average price"),
        "09_amenity_availability.png":      ("Amenity Availability", "% of properties with each amenity"),
        "12_amenity_score_distribution.png":("Amenity Score", "Distribution of amenity count per property"),
        "14_bedroom_category_vs_price.png": ("Bedroom Category vs Price", "Compact/Mid/Large comparison"),
        "15_location_tier_vs_price.png":    ("Location Tier vs Price", "Low/Mid/High tier price comparison"),
        "17_actual_vs_predicted.png":       ("Actual vs Predicted", "Model fit on test data"),
        "18_model_comparison.png":          ("Model Comparison", "R² of all 4 models"),
        "20_residual_analysis.png":         ("Residual Analysis", "Error distribution and pattern"),
        "22_cv_scores.png":                 ("Cross-Validation R²", "Stability across 5 folds"),
    }
    for fname, (title, desc) in plot_meta.items():
        if (_PLOTS_DIR / fname).exists():
            plots.append({"file": fname, "title": title, "desc": desc})

    model_metrics = {
        "model_name":   "Ridge Regression",
        "features":     42,
        "train_rows":   5020,
        "test_rows":    1255,
        "r2_log":       0.071,
        "cv_r2":        0.085,
        "cv_std":       0.012,
        "mae_inr":      "₹26.6 L",
        "rmse_inr":     "₹36.2 L",
        "mape":         48.74,
        "within_10":    12.4,
        "within_20":    25.4,
        "within_30":    37.1,
        "best_feature": "Location_Encoded",
        "best_corr":    0.288,
    }
    return render_template("analytics.html", plots=plots, metrics=model_metrics)


@app.route("/about")
def about():
    return render_template("about.html")


# ---------------------------------------------------------------------------
# Routes — APIs
# ---------------------------------------------------------------------------

@app.route("/api/locations")
def api_locations():
    return jsonify(_KNOWN_LOCATIONS)


@app.route("/api/stats")
def api_stats():
    return jsonify({
        "total_properties": _TOTAL_PROPS,
        "avg_price":        round(_AVG_PRICE),
        "min_price":        round(_MIN_PRICE),
        "max_price":        round(_MAX_PRICE),
        "num_locations":    _NUM_LOCS,
    })


@app.route("/api/chart/price-dist")
def api_price_dist():
    if _df is None:
        return jsonify({"labels": [], "values": []})
    bins = list(range(2_000_000, 15_500_000, 500_000))
    counts, edges = np.histogram(_df["Price"], bins=bins)
    labels = [f"₹{int(e/100_000)}L" for e in edges[:-1]]
    return jsonify({"labels": labels, "values": counts.tolist()})


@app.route("/api/chart/location-prices")
def api_location_prices():
    if _df is None:
        return jsonify({"labels": [], "values": []})
    loc_med = (
        _df.groupby("Location")["Price"]
        .agg(["median", "count"])
        .query("count >= 10")
        .sort_values("median", ascending=False)
        .head(20)
    )
    return jsonify({
        "labels": loc_med.index.tolist(),
        "values": [round(v / 100_000, 1) for v in loc_med["median"]],
    })


@app.route("/api/chart/price-area")
def api_price_area():
    if _df is None:
        return jsonify({"points": []})
    sample = _df[["Area", "Price"]].dropna().sample(
        min(600, len(_df)), random_state=42
    )
    return jsonify({
        "points": [
            {"x": round(float(r.Area)), "y": round(float(r.Price) / 100_000, 1)}
            for _, r in sample.iterrows()
        ]
    })


@app.route("/api/chart/bedroom-prices")
def api_bedroom_prices():
    if _df is None:
        return jsonify({"labels": [], "values": []})
    bed_med = _df.groupby("No. of Bedrooms")["Price"].median().sort_index()
    bed_med = bed_med[bed_med.index <= 6]
    return jsonify({
        "labels": [f"{int(b)} BHK" for b in bed_med.index],
        "values": [round(v / 100_000, 1) for v in bed_med],
    })


# ---------------------------------------------------------------------------
# Static: serve EDA plots
# ---------------------------------------------------------------------------

@app.route("/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(str(_PLOTS_DIR.resolve()), filename)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("   ESTATEIQ — Starting Flask Development Server")
    print("=" * 50)
    print(f"   Open: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
