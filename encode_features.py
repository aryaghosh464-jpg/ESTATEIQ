# -*- coding: utf-8 -*-
"""
ESTATEIQ -- Encode Categorical Variables
=========================================
Encodes:
  Location       -> Target Encoding (mean price per location)
                    NOTE: Computed on full dataset (mild leakage, acceptable
                    for a single-dataset project; use train-only encoding
                    for production).
  BedroomCategory -> Ordinal Encoding  (Compact=0, Mid=1, Large=2)
  LocationTier   -> Ordinal Encoding  (Low=0, Mid=1, High=2)
  IsResale       -> already binary (0/1), kept as-is

Reads  : data/processed/Kolkata_features.csv
Saves  : data/processed/Kolkata_encoded.csv
         models/location_target_encoding.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Load
# ---------------------------------------------------------------------------
df = pd.read_csv("data/processed/Kolkata_features.csv")
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("        ESTATEIQ -- CATEGORICAL ENCODING")
print("=" * 55)
print(f"\nInput shape: {df.shape}")
print(f"Columns    : {df.columns.tolist()}")

# ---------------------------------------------------------------------------
# ENCODING 1: Location -- Target Encoding (mean Price per location)
# ---------------------------------------------------------------------------
print("\n-- Encoding 1: Location -> Target Encoding --")

location_mean_price = df.groupby("Location")["Price"].mean()
global_mean = df["Price"].mean()

# Map; fill unknown locations with global mean
df["Location_Encoded"] = df["Location"].map(location_mean_price).fillna(global_mean)

# Save mapping for inference (predict.py uses this)
mapping_df = location_mean_price.reset_index()
mapping_df.columns = ["Location", "MeanPrice"]
mapping_df.to_csv(models_dir / "location_target_encoding.csv", index=False)
print(f"  Unique locations encoded : {df['Location'].nunique()}")
print(f"  Global mean price        : {global_mean:,.0f}")
print(f"  Mapping saved to         : models/location_target_encoding.csv")

# ---------------------------------------------------------------------------
# ENCODING 2: BedroomCategory -- Ordinal Encoding
# ---------------------------------------------------------------------------
print("\n-- Encoding 2: BedroomCategory -> Ordinal --")
bedroom_order = {"Compact": 0, "Mid": 1, "Large": 2}
df["BedroomCategory_Encoded"] = df["BedroomCategory"].map(bedroom_order)
print(f"  Mapping: {bedroom_order}")
if df["BedroomCategory_Encoded"].isna().any():
    print(f"  WARNING: {df['BedroomCategory_Encoded'].isna().sum()} NaN values after mapping!")

# ---------------------------------------------------------------------------
# ENCODING 3: LocationTier -- Ordinal Encoding
# ---------------------------------------------------------------------------
print("\n-- Encoding 3: LocationTier -> Ordinal --")
tier_order = {"Low": 0, "Mid": 1, "High": 2}
df["LocationTier_Encoded"] = df["LocationTier"].map(tier_order)
print(f"  Mapping: {tier_order}")
if df["LocationTier_Encoded"].isna().any():
    print(f"  WARNING: {df['LocationTier_Encoded'].isna().sum()} NaN values after mapping!")

# ---------------------------------------------------------------------------
# Drop original string columns (keep encoded versions for modelling)
# ---------------------------------------------------------------------------
df_encoded = df.drop(
    columns=["Location", "BedroomCategory", "LocationTier"],
    errors="ignore"
)

print(f"\nColumns after encoding : {df_encoded.shape[1]}")
print("Encoded column stats   :")
print(df_encoded[["Location_Encoded", "BedroomCategory_Encoded",
                   "LocationTier_Encoded"]].describe().to_string())

# ---------------------------------------------------------------------------
# Verify no NaN values in final dataset
# ---------------------------------------------------------------------------
nan_counts = df_encoded.isna().sum()
if nan_counts.any():
    print("\nWARNING: NaN values found in encoded dataset:")
    print(nan_counts[nan_counts > 0])
else:
    print("\n[OK] No NaN values in encoded dataset.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = Path("data/processed/Kolkata_encoded.csv")
df_encoded.to_csv(out_path, index=False)
print(f"\n[Saved] {out_path}  shape={df_encoded.shape}")

print("\n===== ENCODING COMPLETE =====")
print("All columns in final encoded dataset:")
for i, c in enumerate(df_encoded.columns, 1):
    print(f"  {i:2d}. {c}")
