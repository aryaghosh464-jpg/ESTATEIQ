"""
ESTATEIQ — Exploratory Data Analysis
=====================================
Existing work preserved + extended with:
  • Price-per-sqft analysis
  • Location-wise price stats
  • Bedroom count distribution
  • Amenity penetration rates
  • Scatter / box plots
  • Correlation heatmap
Saves all figures to data/plots/
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on any machine)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── 0. Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/Kolkata_cleaned.csv")

# ── 1. Original inspection (preserved) ───────────────────────────────────────
print("DATASET SHAPE:")
print(df.shape)
print("\nCOLUMN NAMES:")
print(df.columns.tolist())
print("\nDATA TYPES:")
print(df.dtypes)
print("\nMISSING VALUES:")
print(df.isnull().sum())
print("\nFIRST 5 ROWS:")
print(df.head())
print("\nSTATISTICAL SUMMARY:")
print(df.describe())
print("\nALL COLUMN NAMES:")
for column in df.columns:
    print(column)

with open("column_names.txt", "w") as file:
    for number, column in enumerate(df.columns, start=1):
        file.write(f"{number}. {column}\n")
print("\nAll column names saved to column_names.txt")

print("\nDATASET SHAPE:")
print(df.shape)
print("\nLOCATION COUNT:")
print(df["Location"].nunique())
print("\nTOP 30 LOCATIONS:")
print(df["Location"].value_counts().head(30))
print("\nDATA TYPES:")
print(df.dtypes)
print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\n========== ESTATEIQ DATA ANALYSIS ==========")
print("\n1. DATASET SIZE:")
print(df.shape)
print("\n2. KOLKATA LOCATION CHECK:")
print(df[df["Location"].str.contains("Kolkata", case=False, na=False)]["Location"].value_counts())
print("\n3. TOP LOCATIONS:")
print(df["Location"].value_counts().head(20))
print("\n4. MISSING VALUES:")
print(df.isnull().sum())
print("\n5. DATA TYPES:")
print(df.dtypes)
print("\n========== ANALYSIS COMPLETE ==========")
print("MISSING VALUES:\n", df.isnull().sum())
print("\nDUPLICATE ROWS:", df.duplicated().sum())
print("\nDATASET SHAPE:", df.shape)
print("MISSING VALUES BEFORE CLEANING:")
print(df.isnull().sum().sum())
df = df.dropna()
print("\nMISSING VALUES AFTER CLEANING:")
print(df.isnull().sum().sum())
print("\nDATASET SHAPE AFTER CLEANING:")
print(df.shape)
print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())
print("\n===== UNIQUE VALUES =====")
print(df.nunique())

# ── 2. Output directory ───────────────────────────────────────────────────────
plots_dir = Path("data/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# ── 3. Original price distribution (preserved) ───────────────────────────────
plt.figure(figsize=(10, 5))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("House Price")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.savefig(plots_dir / "01_price_distribution.png", dpi=120)
plt.close()
print("\n[Saved] 01_price_distribution.png")

# ── 4. Original outlier box-plot (preserved) ─────────────────────────────────
print("\n===== OUTLIER ANALYSIS =====")
Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
outliers = df[(df["Price"] < lower_limit) | (df["Price"] > upper_limit)]
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)
print("Number of Outliers:", len(outliers))

plt.figure(figsize=(10, 5))
plt.boxplot(df["Price"])
plt.title("House Price Outlier Detection")
plt.ylabel("House Price")
plt.tight_layout()
plt.savefig(plots_dir / "02_price_boxplot.png", dpi=120)
plt.close()
print("[Saved] 02_price_boxplot.png")

# ── 5. Original correlation heatmap (preserved) ───────────────────────────────
print("\n===== CORRELATION WITH HOUSE PRICE =====")
correlation = df.corr(numeric_only=True)
print(correlation["Price"].sort_values(ascending=False))

plt.figure(figsize=(14, 10))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(plots_dir / "03_correlation_heatmap.png", dpi=120)
plt.close()
print("[Saved] 03_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# EXTENDED EDA (new sections)
# ══════════════════════════════════════════════════════════════════════════════

# ── 6. Price per square foot ──────────────────────────────────────────────────
print("\n===== PRICE PER SQUARE FOOT =====")
df["PricePerSqft"] = df["Price"] / df["Area"]
print(df["PricePerSqft"].describe())

plt.figure(figsize=(10, 5))
sns.histplot(df["PricePerSqft"], bins=40, kde=True, color="steelblue")
plt.title("Distribution of Price per Sq.ft")
plt.xlabel("Price per Sq.ft (₹)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(plots_dir / "04_price_per_sqft_distribution.png", dpi=120)
plt.close()
print("[Saved] 04_price_per_sqft_distribution.png")

# ── 7. Price vs Area scatter ──────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.scatter(df["Area"], df["Price"], alpha=0.3, s=15, color="darkorange")
plt.title("Price vs Area")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price (₹)")
plt.tight_layout()
plt.savefig(plots_dir / "05_price_vs_area.png", dpi=120)
plt.close()
print("[Saved] 05_price_vs_area.png")

# ── 8. Bedroom count distribution ────────────────────────────────────────────
print("\n===== BEDROOM DISTRIBUTION =====")
print(df["No. of Bedrooms"].value_counts().sort_index())

plt.figure(figsize=(8, 5))
df["No. of Bedrooms"].value_counts().sort_index().plot(kind="bar", color="mediumseagreen", edgecolor="black")
plt.title("Number of Properties by Bedroom Count")
plt.xlabel("Bedrooms")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(plots_dir / "06_bedroom_distribution.png", dpi=120)
plt.close()
print("[Saved] 06_bedroom_distribution.png")

# ── 9. Price by bedroom count ─────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
sns.boxplot(x="No. of Bedrooms", y="Price", data=df,
            hue="No. of Bedrooms", palette="Set2", legend=False)
plt.title("House Price by Number of Bedrooms")
plt.xlabel("Bedrooms")
plt.ylabel("Price (INR)")
plt.tight_layout()
plt.savefig(plots_dir / "07_price_by_bedrooms.png", dpi=120)
plt.close()
print("[Saved] 07_price_by_bedrooms.png")

# ── 10. Top 15 locations by average price ────────────────────────────────────
print("\n===== TOP LOCATIONS BY AVERAGE PRICE =====")
location_price = (
    df.groupby("Location")["Price"]
    .agg(["mean", "median", "count"])
    .rename(columns={"mean": "AvgPrice", "median": "MedianPrice", "count": "Count"})
    .query("Count >= 10")
    .sort_values("AvgPrice", ascending=False)
)
print(location_price.head(15).to_string())

top15 = location_price.head(15)
plt.figure(figsize=(12, 6))
plt.barh(top15.index[::-1], top15["AvgPrice"][::-1] / 1e6, color="coral", edgecolor="black")
plt.title("Top 15 Locations by Average House Price")
plt.xlabel("Average Price (₹ Millions)")
plt.tight_layout()
plt.savefig(plots_dir / "08_top15_locations_avg_price.png", dpi=120)
plt.close()
print("[Saved] 08_top15_locations_avg_price.png")

# ── 11. Amenity penetration ───────────────────────────────────────────────────
amenity_cols = [
    "MaintenanceStaff", "Gymnasium", "SwimmingPool", "LandscapedGardens",
    "JoggingTrack", "RainWaterHarvesting", "IndoorGames", "ShoppingMall",
    "Intercom", "SportsFacility", "ATM", "ClubHouse", "School", "24X7Security",
    "PowerBackup", "CarParking", "StaffQuarter", "Cafeteria", "MultipurposeRoom",
    "Hospital", "WashingMachine", "Gasconnection", "AC", "Wifi",
    "Children'splayarea", "LiftAvailable", "BED", "VaastuCompliant",
    "Microwave", "GolfCourse", "TV", "DiningTable", "Sofa", "Wardrobe",
    "Refrigerator",
]
# NOTE: Amenity columns are 0-9 rated (not binary 0/1).
# Treat rating > 0 as "amenity present"
amenity_presence = (df[amenity_cols] > 0).mean().sort_values(ascending=False) * 100
print("\n===== AMENITY AVAILABILITY (%) =====")
print(amenity_presence.to_string())

plt.figure(figsize=(12, 8))
amenity_presence.plot(kind="barh", color="mediumpurple", edgecolor="black")
plt.title("Amenity Availability Across Properties (%)")
plt.xlabel("% Properties with Amenity (rating > 0)")
plt.tight_layout()
plt.savefig(plots_dir / "09_amenity_availability.png", dpi=120)
plt.close()
print("[Saved] 09_amenity_availability.png")

print("\n===== EDA COMPLETE — all plots saved to data/plots/ =====")