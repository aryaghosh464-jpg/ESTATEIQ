# -*- coding: utf-8 -*-
"""
ESTATEIQ -- Outlier Detection & Treatment
==========================================
Detects and treats outliers in Price and Area using:
  • IQR method  (original work preserved and shown)
  • Z-score method
  • Log transformation (for skewed distribution)
Saves a cleaned dataset: data/processed/Kolkata_no_outliers.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# ── 0. Setup ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/Kolkata_cleaned.csv").dropna()
plots_dir = Path("data/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("        ESTATEIQ — OUTLIER DETECTION & TREATMENT")
print("=" * 55)
print(f"\nOriginal dataset shape: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: IQR Method (preserved from explore_data.py + enhanced)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 1: IQR Method ──")

for col in ["Price", "Area"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"\n{col}:")
    print(f"  Q1={Q1:,.0f}  Q3={Q3:,.0f}  IQR={IQR:,.0f}")
    print(f"  Lower limit : {lower:,.0f}")
    print(f"  Upper limit : {upper:,.0f}")
    print(f"  Outliers    : {n_outliers} ({n_outliers/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Z-Score Method
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 2: Z-Score Method (threshold = 3) ──")

for col in ["Price", "Area"]:
    z_scores = np.abs(stats.zscore(df[col]))
    n_outliers = (z_scores > 3).sum()
    print(f"{col}: {n_outliers} outliers detected")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Visualise before treatment
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("ESTATEIQ — Before Outlier Treatment", fontsize=14, fontweight="bold")

axes[0, 0].boxplot(df["Price"])
axes[0, 0].set_title("Price — Box Plot")
axes[0, 0].set_ylabel("Price (₹)")

axes[0, 1].boxplot(df["Area"])
axes[0, 1].set_title("Area — Box Plot")
axes[0, 1].set_ylabel("Area (sq.ft)")

axes[1, 0].hist(df["Price"], bins=50, color="steelblue", edgecolor="white")
axes[1, 0].set_title("Price — Histogram")
axes[1, 0].set_xlabel("Price (₹)")

axes[1, 1].hist(df["Area"], bins=50, color="darkorange", edgecolor="white")
axes[1, 1].set_title("Area — Histogram")
axes[1, 1].set_xlabel("Area (sq.ft)")

plt.tight_layout()
plt.savefig(plots_dir / "10_before_outlier_treatment.png", dpi=120)
plt.close()
print("\n[Saved] 10_before_outlier_treatment.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Treatment — IQR Capping (Winsorisation)
#   We cap instead of drop to retain data volume.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 4: Treatment — IQR Capping (Winsorisation) ──")

df_treated = df.copy()

for col in ["Price", "Area"]:
    Q1 = df_treated[col].quantile(0.25)
    Q3 = df_treated[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_treated[col] = df_treated[col].clip(lower=lower, upper=upper)
    print(f"{col} capped to [{lower:,.0f}, {upper:,.0f}]")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Log Transformation (for skewed Price)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── STEP 5: Log Transformation ──")
df_treated["LogPrice"] = np.log1p(df_treated["Price"])
df_treated["LogArea"]  = np.log1p(df_treated["Area"])

skew_before = df["Price"].skew()
skew_after  = df_treated["LogPrice"].skew()
print(f"Price skewness — before: {skew_before:.3f}  |  after log: {skew_after:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Visualise after treatment
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("ESTATEIQ — After Outlier Treatment", fontsize=14, fontweight="bold")

axes[0, 0].boxplot(df_treated["Price"])
axes[0, 0].set_title("Price — Box Plot (capped)")
axes[0, 0].set_ylabel("Price (₹)")

axes[0, 1].hist(df_treated["Price"], bins=50, color="mediumseagreen", edgecolor="white")
axes[0, 1].set_title("Price — Histogram (capped)")
axes[0, 1].set_xlabel("Price (₹)")

axes[1, 0].hist(df_treated["LogPrice"], bins=50, color="steelblue", edgecolor="white")
axes[1, 0].set_title("Log(Price) — Histogram")
axes[1, 0].set_xlabel("log(1 + Price)")

axes[1, 1].hist(df_treated["LogArea"], bins=50, color="darkorange", edgecolor="white")
axes[1, 1].set_title("Log(Area) — Histogram")
axes[1, 1].set_xlabel("log(1 + Area)")

plt.tight_layout()
plt.savefig(plots_dir / "11_after_outlier_treatment.png", dpi=120)
plt.close()
print("[Saved] 11_after_outlier_treatment.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Save treated dataset
# ══════════════════════════════════════════════════════════════════════════════
out_path = Path("data/processed/Kolkata_no_outliers.csv")
df_treated.to_csv(out_path, index=False)
print(f"\n[Saved] {out_path}  shape={df_treated.shape}")

print("\n===== OUTLIER TREATMENT COMPLETE =====")
print(f"Original rows : {len(df)}")
print(f"Treated rows  : {len(df_treated)}  (no rows dropped — capping used)")
