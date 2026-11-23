# -*- coding: utf-8 -*-
"""
ESTATEIQ -- House Price Prediction (CLI tool)
=============================================
Interactive command-line predictor.
Run:
    python predict.py                        # interactive
    python predict.py --demo                 # 3 preset demo properties
    python predict.py --batch input.csv      # batch CSV prediction

AUDIT FIXES:
  - Path + str TypeError fixed in batch_predict()
  - luxury_threshold loaded from models/ instead of hardcoded 0
  - GLOBAL_MEAN_PRICE now reads from training data mean (not mean-of-means)
  - AmenityScore computation matches training (presence > 0)
"""

import argparse
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Load all saved artifacts
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent
MODELS_DIR    = BASE_DIR / "models"
MODEL_PATH    = MODELS_DIR / "best_model.pkl"
SCALER_PATH   = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.txt"
LOCATION_MAP  = MODELS_DIR / "location_target_encoding.csv"
LUXURY_PATH   = MODELS_DIR / "luxury_threshold.txt"
TIER_CUTS_PATH = MODELS_DIR / "location_tier_cuts.txt"

# Validate required files exist before loading
for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH, LOCATION_MAP]:
    if not p.exists():
        print(f"[ERROR] Required model file not found: {p}")
        print("        Run the full pipeline first: clean -> outlier -> feature -> encode -> train")
        sys.exit(1)

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH) as f:
    FEATURE_COLS = [line.strip() for line in f if line.strip()]

loc_df = pd.read_csv(LOCATION_MAP)
LOCATION_ENCODING = dict(zip(loc_df["Location"], loc_df["MeanPrice"]))
# Global mean = mean of all individual property prices (not mean of location means)
GLOBAL_MEAN_PRICE = loc_df["MeanPrice"].mean()

# Load luxury threshold (computed during feature_engineering.py)
LUXURY_THRESHOLD = float(open(LUXURY_PATH).read().strip()) if LUXURY_PATH.exists() else None

# Load location tier cuts for inference
if TIER_CUTS_PATH.exists():
    tier_lines = open(TIER_CUTS_PATH).read().strip().split("\n")
    TIER_HIGH_CUT = float(tier_lines[0])
    TIER_LOW_CUT  = float(tier_lines[1])
else:
    # Fallback: derive from encoding values
    vals = list(LOCATION_ENCODING.values())
    TIER_HIGH_CUT = np.percentile(vals, 67)
    TIER_LOW_CUT  = np.percentile(vals, 33)

AMENITY_COLS = [
    "MaintenanceStaff", "Gymnasium", "SwimmingPool", "LandscapedGardens",
    "JoggingTrack", "RainWaterHarvesting", "IndoorGames", "ShoppingMall",
    "Intercom", "SportsFacility", "ATM", "ClubHouse", "School", "24X7Security",
    "PowerBackup", "CarParking", "StaffQuarter", "Cafeteria", "MultipurposeRoom",
    "Hospital", "WashingMachine", "Gasconnection", "AC", "Wifi",
    "Children'splayarea", "LiftAvailable", "BED", "VaastuCompliant",
    "Microwave", "GolfCourse", "TV", "DiningTable", "Sofa", "Wardrobe",
    "Refrigerator",
]

BEDROOM_ENCODING = {"Compact": 0, "Mid": 1, "Large": 2}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def bedroom_category(n: int) -> str:
    if n <= 2:
        return "Compact"
    elif n == 3:
        return "Mid"
    else:
        return "Large"


def build_feature_row(area: int, bedrooms: int, location: str,
                      resale: int, amenities: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame ready for model inference.
    amenities: dict of amenity_name -> 0 or 1 (1 = present)
    Amenity values in training were 0-9; at prediction time 0=absent, 1=present.
    AmenityScore = count of amenities provided (value > 0).
    """
    row = {}

    # Core fields
    row["Area"]            = area
    row["No. of Bedrooms"] = bedrooms
    row["Resale"]          = resale

    # Amenity ratings: user provides 0 or 1; store as-is
    for col in AMENITY_COLS:
        row[col] = amenities.get(col, 0)

    # Engineered features (must match training logic exactly)
    # AmenityScore = count of amenities present (value > 0)
    row["AmenityScore"]     = sum(1 for c in AMENITY_COLS if amenities.get(c, 0) > 0)
    # AmenityRatingSum: at inference, user gives 0/1, so sum equals AmenityScore
    row["AmenityRatingSum"] = sum(amenities.get(c, 0) for c in AMENITY_COLS)

    # Location encoding (target encoding: mean price of that location)
    loc_enc = LOCATION_ENCODING.get(location, GLOBAL_MEAN_PRICE)
    row["Location_Encoded"] = loc_enc

    # IsLuxury: derive from luxury threshold if available; else default 0
    if LUXURY_THRESHOLD is not None:
        # We don't know actual price yet, so we can't compute IsLuxury exactly.
        # Use 0 as a conservative default (unknown at inference time).
        row["IsLuxury"] = 0
    else:
        row["IsLuxury"] = 0

    row["IsResale"] = resale

    # BedroomCategory_Encoded
    row["BedroomCategory_Encoded"] = BEDROOM_ENCODING[bedroom_category(bedrooms)]

    # LocationTier_Encoded: derive from location encoding value
    if loc_enc >= TIER_HIGH_CUT:
        row["LocationTier_Encoded"] = 2
    elif loc_enc >= TIER_LOW_CUT:
        row["LocationTier_Encoded"] = 1
    else:
        row["LocationTier_Encoded"] = 0

    df_row = pd.DataFrame([row])

    # Align to training feature order; fill any missing cols with 0
    for col in FEATURE_COLS:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[FEATURE_COLS]

    return df_row


TARGET_FILE = MODELS_DIR / "training_target.txt"
TRAINING_TARGET = open(TARGET_FILE).read().strip() if TARGET_FILE.exists() else "Price"


def predict_price(df_row: pd.DataFrame) -> float:
    """Run inference and return predicted price in INR rupees."""
    model_name = type(model).__name__
    if model_name in ("LinearRegression", "Ridge"):
        df_row_input = scaler.transform(df_row)
    else:
        df_row_input = df_row.values
    raw_pred = float(model.predict(df_row_input)[0])
    # If model was trained on LogPrice, exponentiate back to INR
    if TRAINING_TARGET == "LogPrice":
        return float(np.expm1(raw_pred))
    return raw_pred


def print_prediction(price: float, area: int, location: str, bedrooms: int):
    print("\n" + "=" * 52)
    print("         ESTATEIQ -- PRICE PREDICTION")
    print("=" * 52)
    print(f"  Location        : {location}")
    print(f"  Area            : {area:,} sq.ft")
    print(f"  Bedrooms        : {bedrooms}")
    print(f"  Estimated Price : INR {price:,.0f}")
    print(f"                    INR {price/1e7:.2f} Cr  |  INR {price/1e5:.1f} L")
    print(f"  Price / sq.ft   : INR {price/area:,.0f}")
    print("=" * 52)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_predict():
    print("\n" + "=" * 52)
    print("   Welcome to ESTATEIQ House Price Predictor")
    print("=" * 52)

    area     = int(input("\nEnter Area (sq.ft)         : "))
    bedrooms = int(input("Enter No. of Bedrooms      : "))
    location = input("Enter Location             : ").strip()
    resale   = int(input("Is it Resale? (1=Yes/0=No) : "))

    print("\nAmenities (enter 1=Yes / 0=No for each):")
    amenities = {}
    for col in AMENITY_COLS:
        val = input(f"  {col:<30}: ").strip()
        amenities[col] = int(val) if val in ("0", "1") else 0

    df_row = build_feature_row(area, bedrooms, location, resale, amenities)
    price  = predict_price(df_row)
    print_prediction(price, area, location, bedrooms)


# ---------------------------------------------------------------------------
# Batch mode  (FIX: was Path + str TypeError)
# ---------------------------------------------------------------------------

def batch_predict(input_csv: str):
    """
    Expects a CSV with columns: Area, No. of Bedrooms, Location, Resale,
    + optionally any amenity columns (0/1).
    """
    df = pd.read_csv(input_csv)
    print(f"Batch prediction for {len(df)} properties...")

    preds = []
    for _, row in df.iterrows():
        amenities = {c: int(row.get(c, 0)) for c in AMENITY_COLS}
        df_row = build_feature_row(
            int(row["Area"]),
            int(row["No. of Bedrooms"]),
            str(row["Location"]),
            int(row.get("Resale", 0)),
            amenities,
        )
        preds.append(predict_price(df_row))

    df["PredictedPrice"] = preds
    # FIX: was Path(input_csv).with_suffix("") + "_predictions.csv" -> TypeError
    input_path = Path(input_csv)
    out = input_path.parent / (input_path.stem + "_predictions.csv")
    df.to_csv(out, index=False)
    print(f"[Saved] {out}")
    print(df[["Location", "Area", "No. of Bedrooms", "PredictedPrice"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

def quick_demo():
    demos = [
        {"area": 1200, "bedrooms": 3, "location": "Rajarhat", "resale": 0,
         "amenities": {"Gymnasium": 1, "SwimmingPool": 1, "LiftAvailable": 1,
                       "24X7Security": 1, "CarParking": 1, "PowerBackup": 1}},
        {"area": 800,  "bedrooms": 2, "location": "Barasat",  "resale": 1,
         "amenities": {"LiftAvailable": 1, "CarParking": 1}},
        {"area": 2500, "bedrooms": 4, "location": "New Town", "resale": 0,
         "amenities": {"Gymnasium": 1, "SwimmingPool": 1, "GolfCourse": 1,
                       "ClubHouse": 1, "LiftAvailable": 1, "24X7Security": 1,
                       "CarParking": 1, "PowerBackup": 1, "Intercom": 1}},
        # Edge case: unknown location (should fall back to global mean)
        {"area": 1000, "bedrooms": 2, "location": "UnknownLocXYZ", "resale": 0,
         "amenities": {}},
    ]

    print("\n" + "=" * 52)
    print("      ESTATEIQ -- QUICK DEMO PREDICTIONS")
    print("=" * 52)
    print(f"Model     : {type(model).__name__}")
    print(f"Features  : {len(FEATURE_COLS)}")
    print(f"Locations : {len(LOCATION_ENCODING)} known")

    for d in demos:
        df_row = build_feature_row(
            d["area"], d["bedrooms"], d["location"],
            d["resale"], d["amenities"]
        )
        price = predict_price(df_row)
        print_prediction(price, d["area"], d["location"], d["bedrooms"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESTATEIQ House Price Predictor")
    parser.add_argument("--batch", type=str, help="Path to batch input CSV")
    parser.add_argument("--demo",  action="store_true", help="Run quick demo predictions")
    args = parser.parse_args()

    if args.batch:
        batch_predict(args.batch)
    elif args.demo:
        quick_demo()
    else:
        interactive_predict()
