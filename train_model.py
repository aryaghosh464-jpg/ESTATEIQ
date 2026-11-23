# -*- coding: utf-8 -*-
"""
ESTATEIQ -- Model Training  (v2 - audit-fixed)
===============================================
Key fixes from audit:
  1. IsLuxury removed (direct data leakage, corr=0.847 with Price)
  2. AmenityRatingSum removed (0.98 corr with AmenityScore - redundant)
  3. Near-constant amenity columns removed (>99.5% rows have value=9)
  4. Training on LogPrice target (skew: 12.9->0.27, better for regression)
  5. Predictions exponentiated back to INR for reporting

Models compared:
  1. Linear Regression  (baseline)
  2. Ridge Regression   (L2 regularisation)
  3. Random Forest      (ensemble)
  4. Gradient Boosting  (ensemble)

Saves:
  models/best_model.pkl
  models/scaler.pkl
  models/feature_names.txt
  models/training_target.txt    ('LogPrice' - read by predict.py)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# 0. Load
# ---------------------------------------------------------------------------
df = pd.read_csv("data/processed/Kolkata_encoded.csv")
plots_dir = Path("data/plots")
models_dir = Path("models")
plots_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("        ESTATEIQ -- MODEL TRAINING (v2, audit-fixed)")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")

# ---------------------------------------------------------------------------
# 1. Identify and drop near-constant amenity columns
#    (columns where >99% of values are the same value)
# ---------------------------------------------------------------------------
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
near_constant = []
for col in amenity_cols:
    if col in df.columns:
        top_freq = df[col].value_counts(normalize=True).iloc[0]
        if top_freq >= 0.995:   # >99.5% same value
            near_constant.append(col)

print(f"\n[Audit] Near-constant amenity columns (>99.5% same value): {len(near_constant)}")
for col in near_constant:
    top_val = df[col].value_counts().idxmax()
    top_pct = df[col].value_counts(normalize=True).iloc[0] * 100
    print(f"  {col:<30}: {top_val} in {top_pct:.1f}% of rows -- DROPPED")

# ---------------------------------------------------------------------------
# 2. Define feature set
# ---------------------------------------------------------------------------
# Hard exclusions (leakage or redundant):
#   Price        - target
#   LogPrice     - monotonic transform of target (leakage)
#   PricePerSqft - Price/Area (leakage)
#   LogArea      - redundant with Area
#   IsResale     - duplicate of Resale
#   IsLuxury     - derived from Price >= 70th pct (corr=0.847, leakage)
#   AmenityRatingSum - 0.98 corr with AmenityScore (redundant)
EXCLUDE = {
    "Price", "LogPrice", "PricePerSqft", "LogArea",
    "IsResale", "IsLuxury", "AmenityRatingSum",
}
EXCLUDE.update(near_constant)   # remove near-constant amenity cols

TARGET_COL = "LogPrice"         # train on log-transformed price

feature_cols = [c for c in df.columns if c not in EXCLUDE and c != TARGET_COL]
print(f"\nFeatures used ({len(feature_cols)}):")
for f in feature_cols:
    print(f"  {f}")

# ---------------------------------------------------------------------------
# 3. Check correlations with LogPrice for transparency
# ---------------------------------------------------------------------------
print("\n[Correlations with LogPrice]")
corr_with_log = df[feature_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
print(corr_with_log.head(10).to_string())

X = df[feature_cols]
y = df[TARGET_COL]

# ---------------------------------------------------------------------------
# 4. Train-test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain : {len(X_train)} rows")
print(f"Test  : {len(X_test)} rows")
print(f"Target: {TARGET_COL}  (predictions will be exponentiated back to INR)")

# ---------------------------------------------------------------------------
# 5. Feature scaling (for linear models)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 6. Define models
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=10.0),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    ),
}

# ---------------------------------------------------------------------------
# 7. Train, evaluate, compare
# ---------------------------------------------------------------------------
results = []
print("\n-- Training models on LogPrice --")

# Reference: naive baseline (predict mean LogPrice for everyone)
baseline_pred = np.full(len(y_test), y_train.mean())
baseline_mae  = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
print(f"\nBaseline (mean predictor): MAE_log={baseline_mae:.4f}  RMSE_log={baseline_rmse:.4f}")

for name, model in models.items():
    is_linear = "Regression" in name
    Xtr = X_train_sc if is_linear else X_train
    Xte = X_test_sc  if is_linear else X_test

    model.fit(Xtr, y_train)
    log_pred = model.predict(Xte)

    # Metrics on log scale
    mae_log  = mean_absolute_error(y_test, log_pred)
    rmse_log = np.sqrt(mean_squared_error(y_test, log_pred))
    r2_log   = r2_score(y_test, log_pred)

    # Back-transform to INR and compute price metrics
    price_pred   = np.expm1(log_pred)
    price_actual = np.expm1(y_test.values)
    mae_inr  = mean_absolute_error(price_actual, price_pred)
    rmse_inr = np.sqrt(mean_squared_error(price_actual, price_pred))
    mape     = np.mean(np.abs((price_actual - price_pred) / price_actual)) * 100

    # Cross-validation R² on log scale
    cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring="r2", n_jobs=-1)

    results.append({
        "Model":      name,
        "R2_log":     r2_log,
        "CV_R2_log":  cv_scores.mean(),
        "MAE_log":    mae_log,
        "RMSE_log":   rmse_log,
        "MAE_INR":    mae_inr,
        "RMSE_INR":   rmse_inr,
        "MAPE":       mape,
        "log_pred":   log_pred,
        "price_pred": price_pred,
    })
    print(f"\n{name}")
    print(f"  R2 (log scale)  : {r2_log:.4f}  |  CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  MAE  (INR)      : {mae_inr:>15,.0f}")
    print(f"  RMSE (INR)      : {rmse_inr:>15,.0f}")
    print(f"  MAPE            : {mape:.2f}%")

# ---------------------------------------------------------------------------
# 8. Select best model by CV R2
# ---------------------------------------------------------------------------
results_df = pd.DataFrame([{k: v for k, v in r.items()
                             if k not in ("log_pred", "price_pred")} for r in results])
print("\n-- Model Comparison --")
print(results_df[["Model", "R2_log", "CV_R2_log", "MAE_INR", "RMSE_INR", "MAPE"]].to_string(index=False))

best_idx  = results_df["CV_R2_log"].idxmax()
best_name = results_df.loc[best_idx, "Model"]
print(f"\nBest model: {best_name}  (CV R2={results_df.loc[best_idx,'CV_R2_log']:.4f})")

best_model      = models[best_name]
best_log_pred   = results[best_idx]["log_pred"]
best_price_pred = results[best_idx]["price_pred"]
price_actual    = np.expm1(y_test.values)

# ---------------------------------------------------------------------------
# 9. Feature importance (tree models)
# ---------------------------------------------------------------------------
if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    print("\nTop 15 Feature Importances:")
    print(importance.head(15).to_string())

    plt.figure(figsize=(10, 7))
    importance.head(15).sort_values().plot(kind="barh", color="steelblue", edgecolor="black")
    plt.title(f"Top Feature Importances -- {best_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(plots_dir / "16_feature_importances.png", dpi=120)
    plt.close()
    print("[Saved] 16_feature_importances.png")

# ---------------------------------------------------------------------------
# 10. Plots: actual vs predicted (INR scale)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.scatter(price_actual, best_price_pred, alpha=0.3, s=15, color="darkorange")
max_val = max(price_actual.max(), best_price_pred.max())
plt.plot([0, max_val], [0, max_val], "r--", linewidth=1.5)
plt.title(f"Actual vs Predicted (INR) -- {best_name}")
plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price (INR)")
plt.tight_layout()
plt.savefig(plots_dir / "17_actual_vs_predicted.png", dpi=120)
plt.close()
print("[Saved] 17_actual_vs_predicted.png")

# Model comparison bar chart
plt.figure(figsize=(10, 5))
results_df.set_index("Model")["CV_R2_log"].sort_values().plot(
    kind="barh", color="mediumseagreen", edgecolor="black"
)
plt.title("Model Comparison -- CV R2 (LogPrice)")
plt.xlabel("CV R2")
plt.tight_layout()
plt.savefig(plots_dir / "18_model_comparison.png", dpi=120)
plt.close()
print("[Saved] 18_model_comparison.png")

# Residual plot (INR scale)
residuals = price_actual - best_price_pred
plt.figure(figsize=(10, 5))
plt.scatter(best_price_pred, residuals, alpha=0.3, s=15, color="steelblue")
plt.axhline(0, color="red", linestyle="--")
plt.title(f"Residual Plot (INR) -- {best_name}")
plt.xlabel("Predicted Price (INR)")
plt.ylabel("Residual (INR)")
plt.tight_layout()
plt.savefig(plots_dir / "19_residual_plot.png", dpi=120)
plt.close()
print("[Saved] 19_residual_plot.png")

# ---------------------------------------------------------------------------
# 11. Save model artifacts
# ---------------------------------------------------------------------------
joblib.dump(best_model, models_dir / "best_model.pkl")
joblib.dump(scaler,     models_dir / "scaler.pkl")

with open(models_dir / "feature_names.txt", "w") as f:
    for feat in feature_cols:
        f.write(feat + "\n")

# Save target column name so predict.py knows to exponentiate
with open(models_dir / "training_target.txt", "w") as f:
    f.write(TARGET_COL + "\n")

print(f"\n[Saved] models/best_model.pkl    -- {best_name}")
print(f"[Saved] models/scaler.pkl")
print(f"[Saved] models/feature_names.txt  -- {len(feature_cols)} features")
print(f"[Saved] models/training_target.txt -- {TARGET_COL}")

print("\n===== MODEL TRAINING COMPLETE =====")
print(f"Best model  : {best_name}")
print(f"CV R2 (log) : {results_df.loc[best_idx,'CV_R2_log']:.4f}")
print(f"MAE (INR)   : {results_df.loc[best_idx,'MAE_INR']:,.0f}")
print(f"MAPE        : {results_df.loc[best_idx,'MAPE']:.2f}%")
