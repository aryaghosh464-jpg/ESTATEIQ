# -*- coding: utf-8 -*-
"""
ESTATEIQ -- Model Evaluation  (v2, audit-fixed)
=================================================
Loads the saved best model and evaluates on the same train/test split.
Handles LogPrice target: metrics reported in both log and INR scale.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# 0. Load artifacts
# ---------------------------------------------------------------------------
df     = pd.read_csv("data/processed/Kolkata_encoded.csv")
model  = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
plots_dir = Path("data/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

with open("models/feature_names.txt") as f:
    feature_cols = [line.strip() for line in f if line.strip()]

TARGET_COL = "LogPrice"
if Path("models/training_target.txt").exists():
    TARGET_COL = open("models/training_target.txt").read().strip()

model_name = type(model).__name__
print("=" * 60)
print("        ESTATEIQ -- MODEL EVALUATION (v2)")
print("=" * 60)
print(f"\nModel       : {model_name}")
print(f"Target      : {TARGET_COL}  (predictions exponentiated to INR)")
print(f"Features    : {len(feature_cols)}")

X = df[feature_cols]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Test samples: {len(X_test)}")

# ---------------------------------------------------------------------------
# 1. Predict
# ---------------------------------------------------------------------------
is_linear = model_name in ("LinearRegression", "Ridge")
X_test_input  = scaler.transform(X_test)  if is_linear else X_test
X_train_input = scaler.transform(X_train) if is_linear else X_train
X_all_input   = scaler.transform(X)       if is_linear else X

log_pred     = model.predict(X_test_input)
price_pred   = np.expm1(log_pred)
price_actual = np.expm1(y_test.values)

# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------
# Log-scale metrics
r2_log   = r2_score(y_test, log_pred)
mae_log  = mean_absolute_error(y_test, log_pred)
rmse_log = np.sqrt(mean_squared_error(y_test, log_pred))

# INR-scale metrics
mae_inr  = mean_absolute_error(price_actual, price_pred)
rmse_inr = np.sqrt(mean_squared_error(price_actual, price_pred))
r2_inr   = r2_score(price_actual, price_pred)
mape     = np.mean(np.abs((price_actual - price_pred) / price_actual)) * 100

print("\n-- Core Metrics (Log Scale) --")
print(f"  R2   : {r2_log:.4f}")
print(f"  MAE  : {mae_log:.4f}")
print(f"  RMSE : {rmse_log:.4f}")

print("\n-- Core Metrics (INR Scale, after exponentiation) --")
print(f"  MAE  : INR {mae_inr:>15,.0f}")
print(f"  RMSE : INR {rmse_inr:>15,.0f}")
print(f"  R2   : {r2_inr:.4f}")
print(f"  MAPE : {mape:.2f}%")

# ---------------------------------------------------------------------------
# 3. Cross-validation
# ---------------------------------------------------------------------------
print("\n-- 5-Fold Cross-Validation R2 (log scale) --")
cv_scores = cross_val_score(
    model, X_all_input, y,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="r2", n_jobs=-1
)
print(f"  Fold scores : {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean R2     : {cv_scores.mean():.4f}")
print(f"  Std  R2     : {cv_scores.std():.4f}")

# ---------------------------------------------------------------------------
# 4. Error percentile analysis (INR)
# ---------------------------------------------------------------------------
errors = np.abs(price_actual - price_pred)
print("\n-- Error Percentile Analysis (INR) --")
for p in [25, 50, 75, 90, 95]:
    print(f"  P{p:2d}: INR {np.percentile(errors, p):>12,.0f}")

within_10 = (errors / price_actual < 0.10).mean() * 100
within_20 = (errors / price_actual < 0.20).mean() * 100
within_30 = (errors / price_actual < 0.30).mean() * 100
print(f"\n  Within 10% of actual : {within_10:.1f}%")
print(f"  Within 20% of actual : {within_20:.1f}%")
print(f"  Within 30% of actual : {within_30:.1f}%")

# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
# Residual distribution
residuals = price_actual - price_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Residual Analysis -- {model_name} (INR)", fontsize=13, fontweight="bold")
axes[0].hist(residuals, bins=50, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="red", linestyle="--")
axes[0].set_title("Residual Distribution (INR)")
axes[0].set_xlabel("Residual (INR)")
axes[0].set_ylabel("Count")
axes[1].scatter(price_pred, residuals, alpha=0.3, s=12, color="darkorange")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Residuals vs Predicted (INR)")
axes[1].set_xlabel("Predicted Price (INR)")
axes[1].set_ylabel("Residual (INR)")
plt.tight_layout()
plt.savefig(plots_dir / "20_residual_analysis.png", dpi=120)
plt.close()
print("\n[Saved] 20_residual_analysis.png")

# Actual vs Predicted
plt.figure(figsize=(8, 8))
plt.scatter(price_actual, price_pred, alpha=0.3, s=15, color="mediumseagreen")
max_val = max(price_actual.max(), price_pred.max())
plt.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect")
plt.title(f"Actual vs Predicted (INR) -- {model_name}")
plt.xlabel("Actual Price (INR)")
plt.ylabel("Predicted Price (INR)")
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / "21_actual_vs_predicted_eval.png", dpi=120)
plt.close()
print("[Saved] 21_actual_vs_predicted_eval.png")

# CV score bar chart
plt.figure(figsize=(8, 4))
plt.bar([f"Fold {i+1}" for i in range(len(cv_scores))],
        cv_scores, color="mediumpurple", edgecolor="black")
plt.axhline(cv_scores.mean(), color="red", linestyle="--",
            label=f"Mean = {cv_scores.mean():.4f}")
plt.title("Cross-Validation R2 per Fold (log scale)")
plt.ylabel("R2")
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / "22_cv_scores.png", dpi=120)
plt.close()
print("[Saved] 22_cv_scores.png")

# ---------------------------------------------------------------------------
# 6. Save evaluation report
# ---------------------------------------------------------------------------
report_path = Path("models/evaluation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("ESTATEIQ -- Model Evaluation Report (v2)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Model       : {model_name}\n")
    f.write(f"Target      : {TARGET_COL}\n")
    f.write(f"Features    : {len(feature_cols)}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")
    f.write("Log-Scale Metrics\n")
    f.write(f"  R2   : {r2_log:.4f}\n")
    f.write(f"  MAE  : {mae_log:.4f}\n")
    f.write(f"  RMSE : {rmse_log:.4f}\n\n")
    f.write("INR-Scale Metrics (after exponentiation)\n")
    f.write(f"  MAE  : INR {mae_inr:,.0f}\n")
    f.write(f"  RMSE : INR {rmse_inr:,.0f}\n")
    f.write(f"  R2   : {r2_inr:.4f}\n")
    f.write(f"  MAPE : {mape:.2f}%\n\n")
    f.write("Cross-Validation (5-fold, log scale)\n")
    f.write(f"  Mean R2 : {cv_scores.mean():.4f}\n")
    f.write(f"  Std  R2 : {cv_scores.std():.4f}\n\n")
    f.write("Accuracy (INR scale)\n")
    f.write(f"  Within 10% : {within_10:.1f}%\n")
    f.write(f"  Within 20% : {within_20:.1f}%\n")
    f.write(f"  Within 30% : {within_30:.1f}%\n")

print(f"[Saved] {report_path}")
print("\n===== MODEL EVALUATION COMPLETE =====")
