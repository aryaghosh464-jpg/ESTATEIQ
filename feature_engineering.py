# -*- coding: utf-8 -*-
"""
ESTATEIQ -- Feature Engineering
================================
Builds meaningful features from the cleaned/outlier-treated dataset.

BUG FIX (audit): Amenity columns are 0-9 rated, NOT binary 0/1.
  0 = amenity absent, 1-9 = present with quality rating.
  AmenityScore  = count of amenities present (rating > 0)
  AmenityRatingSum = sum of all ratings

New features:
  PricePerSqft, AmenityScore, AmenityRatingSum,
  BedroomCategory, IsLuxury, LocationTier,
  LogPrice (if not from outlier_treatment), LogArea, IsResale

Reads  : data/processed/Kolkata_no_outliers.csv
Saves  : data/processed/Kolkata_features.csv
         models/luxury_threshold.txt   (for inference)
         models/location_tier_cuts.txt (for inference)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Load
# ---------------------------------------------------------------------------
df = pd.read_csv("data/processed/Kolkata_no_outliers.csv")
plots_dir = Path("data/plots")
models_dir = Path("models")
plots_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("        ESTATEIQ -- FEATURE ENGINEERING")
print("=" * 55)
print(f"\nInput shape: {df.shape}")

# ---------------------------------------------------------------------------
# FEATURE 1: Price per sq.ft
# ---------------------------------------------------------------------------
df["PricePerSqft"] = df["Price"] / df["Area"]
print("\n[Feature 1] PricePerSqft -- created")

# ---------------------------------------------------------------------------
# FEATURE 2: AmenityScore & AmenityRatingSum
# NOTE: Amenity columns are on a 0-9 RATING scale, not binary 0/1.
#       0 = absent, 1-9 = present with quality rating.
# ---------------------------------------------------------------------------
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
amenity_cols = [c for c in AMENITY_COLS if c in df.columns]
df["AmenityScore"]     = (df[amenity_cols] > 0).sum(axis=1)   # count present
df["AmenityRatingSum"] = df[amenity_cols].sum(axis=1)          # total quality
print(f"[Feature 2] AmenityScore range      : {df['AmenityScore'].min()} - {df['AmenityScore'].max()}")
print(f"            AmenityRatingSum range  : {df['AmenityRatingSum'].min()} - {df['AmenityRatingSum'].max()}")

# ---------------------------------------------------------------------------
# FEATURE 3: Bedroom Category
# ---------------------------------------------------------------------------
def bedroom_category(n):
    if n <= 2:
        return "Compact"
    elif n == 3:
        return "Mid"
    else:
        return "Large"

df["BedroomCategory"] = df["No. of Bedrooms"].apply(bedroom_category)
print("[Feature 3] BedroomCategory distribution:")
print(df["BedroomCategory"].value_counts().to_string())

# ---------------------------------------------------------------------------
# FEATURE 4: Luxury Flag  (top 30% by price, threshold saved for inference)
# ---------------------------------------------------------------------------
luxury_threshold = df["Price"].quantile(0.70)
df["IsLuxury"] = (df["Price"] >= luxury_threshold).astype(int)

# Save threshold so predict.py can use it consistently
with open(models_dir / "luxury_threshold.txt", "w") as fh:
    fh.write(str(luxury_threshold))

print(f"\n[Feature 4] IsLuxury threshold : {luxury_threshold:,.0f}")
print(f"            Luxury properties : {df['IsLuxury'].sum()} ({df['IsLuxury'].mean()*100:.1f}%)")

# ---------------------------------------------------------------------------
# FEATURE 5: Location Tier  (High / Mid / Low, cuts saved for inference)
# ---------------------------------------------------------------------------
loc_median = df.groupby("Location")["Price"].median()
high_cut = loc_median.quantile(0.67)
low_cut  = loc_median.quantile(0.33)

def location_tier(loc):
    med = loc_median.get(loc, loc_median.median())
    if med >= high_cut:
        return "High"
    elif med >= low_cut:
        return "Mid"
    else:
        return "Low"

df["LocationTier"] = df["Location"].apply(location_tier)

# Save tier cuts for inference
with open(models_dir / "location_tier_cuts.txt", "w") as fh:
    fh.write(f"{high_cut}\n{low_cut}\n")

print(f"\n[Feature 5] LocationTier (high_cut={high_cut:,.0f}, low_cut={low_cut:,.0f})")
print(df["LocationTier"].value_counts().to_string())

# ---------------------------------------------------------------------------
# FEATURE 6: Log Price and Log Area (carry over from outlier_treatment)
# ---------------------------------------------------------------------------
if "LogPrice" not in df.columns:
    df["LogPrice"] = np.log1p(df["Price"])
if "LogArea" not in df.columns:
    df["LogArea"] = np.log1p(df["Area"])
print("\n[Feature 6] LogPrice & LogArea -- confirmed")

# ---------------------------------------------------------------------------
# FEATURE 7: IsResale alias
# ---------------------------------------------------------------------------
df["IsResale"] = df["Resale"]
print("[Feature 7] IsResale -- aliased from Resale")

# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
# AmenityScore distribution
plt.figure(figsize=(10, 5))
df["AmenityScore"].value_counts().sort_index().plot(
    kind="bar", color="mediumpurple", edgecolor="black"
)
plt.title("Amenity Score Distribution")
plt.xlabel("Number of Amenities Present")
plt.ylabel("Property Count")
plt.tight_layout()
plt.savefig(plots_dir / "12_amenity_score_distribution.png", dpi=120)
plt.close()
print("\n[Saved] 12_amenity_score_distribution.png")

# AmenityScore vs Price
plt.figure(figsize=(12, 6))
sns.boxplot(x="AmenityScore", y="Price", data=df,
            hue="AmenityScore", palette="coolwarm", legend=False)
plt.title("Price by Amenity Score (count of amenities present)")
plt.xlabel("Amenity Score")
plt.ylabel("Price (INR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(plots_dir / "13_amenity_score_vs_price.png", dpi=120)
plt.close()
print("[Saved] 13_amenity_score_vs_price.png")

# BedroomCategory vs Price
plt.figure(figsize=(8, 5))
sns.boxplot(x="BedroomCategory", y="Price", data=df,
            order=["Compact", "Mid", "Large"],
            hue="BedroomCategory", palette="Set2", legend=False)
plt.title("Price by Bedroom Category")
plt.xlabel("Bedroom Category")
plt.ylabel("Price (INR)")
plt.tight_layout()
plt.savefig(plots_dir / "14_bedroom_category_vs_price.png", dpi=120)
plt.close()
print("[Saved] 14_bedroom_category_vs_price.png")

# LocationTier vs Price
plt.figure(figsize=(8, 5))
sns.boxplot(x="LocationTier", y="Price", data=df,
            order=["Low", "Mid", "High"],
            hue="LocationTier", palette="Blues", legend=False)
plt.title("Price by Location Tier")
plt.xlabel("Location Tier")
plt.ylabel("Price (INR)")
plt.tight_layout()
plt.savefig(plots_dir / "15_location_tier_vs_price.png", dpi=120)
plt.close()
print("[Saved] 15_location_tier_vs_price.png")

# ---------------------------------------------------------------------------
# Save engineered dataset
# ---------------------------------------------------------------------------
out_path = Path("data/processed/Kolkata_features.csv")
df.to_csv(out_path, index=False)
print(f"\n[Saved] {out_path}  shape={df.shape}")

print("\nNew features added:")
for f in ["PricePerSqft", "AmenityScore", "AmenityRatingSum",
          "BedroomCategory", "IsLuxury", "LocationTier",
          "LogPrice", "LogArea", "IsResale"]:
    print(f"  + {f}")

print("\n===== FEATURE ENGINEERING COMPLETE =====")
