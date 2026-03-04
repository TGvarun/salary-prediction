"""
Salary Prediction System - Model Training Pipeline
===================================================
Industry filter : IT
Job title filter: Software, Developer, Engineer, Data, AI, Machine Learning,
                  DevOps, Cloud, Backend, Frontend, Full Stack, Cyber
Algorithm       : Linear Regression + Polynomial (degree=2) comparison
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("=" * 65)
print("  SALARY PREDICTION SYSTEM — TRAINING PIPELINE")
print("=" * 65)

# 1. LOAD
df = pd.read_csv("salaries.csv")
print(f"\n[1] Raw dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# 2. FILTER
df = df[df["industry"] == "IT"].copy()
print(f"\n[2] After industry='IT': {len(df):,} rows")
TECH_KEYWORDS = ["Software","Developer","Engineer","Data","AI","Machine Learning","DevOps","Cloud","Backend","Frontend","Full Stack","Cyber"]
pattern = "|".join(TECH_KEYWORDS)
df = df[df["job_title"].str.contains(pattern, case=False, na=False)].copy()
print(f"    After job_title filter: {len(df):,} rows")
print(f"    Titles: {sorted(df['job_title'].unique().tolist())}")

# 3. DROP USELESS COLS
DROP_COLS = ["Unnamed: 0", "employee_id", "industry"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
df.dropna(inplace=True)
print(f"\n[3] Dropped useless cols. Shape: {df.shape}  | Nulls: {df.isnull().sum().sum()}")

# 4. FEATURES
TARGET = "annual_salary"
CATEGORICAL_COLS = ["gender", "education", "job_title", "company_size", "location", "remote_work"]
NUMERICAL_COLS   = ["age", "experience_years", "skills_score", "certifications", "performance_rating"]
X = df[CATEGORICAL_COLS + NUMERICAL_COLS].copy()
y = df[TARGET].copy()
print(f"\n[4] Num features: {NUMERICAL_COLS}")
print(f"    Cat features: {CATEGORICAL_COLS}")
print(f"    Target range: ₹{y.min():,.0f} – ₹{y.max():,.0f}")

# 5. SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n[5] Train: {len(X_train):,} | Test: {len(X_test):,}")

# 6. VIF
print(f"\n[6] VIF Analysis:")
def calc_vif(df_num):
    arr = df_num.values.astype(float)
    cols = list(df_num.columns)
    vifs = {}
    for i, col in enumerate(cols):
        y_i = arr[:, i]
        X_i = np.column_stack([np.ones(len(arr)), np.delete(arr, i, axis=1)])
        coef, *_ = np.linalg.lstsq(X_i, y_i, rcond=None)
        y_hat = X_i @ coef
        ss_res = np.sum((y_i - y_hat)**2)
        ss_tot = np.sum((y_i - y_i.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        vifs[col] = round(1/(1-r2) if r2 < 1 else float("inf"), 3)
    return vifs

sc_vif = StandardScaler()
X_num_sc = pd.DataFrame(sc_vif.fit_transform(X_train[NUMERICAL_COLS]), columns=NUMERICAL_COLS)
vif_dict = calc_vif(X_num_sc)
print(f"    {'Feature':<25} {'VIF':>8}  Decision")
print(f"    {'-'*50}")
HIGH_VIF = []
for feat, vif in vif_dict.items():
    flag = "DROP (>10)" if vif > 10 else "Keep"
    print(f"    {feat:<25} {vif:>8.3f}  {flag}")
    if vif > 10: HIGH_VIF.append(feat)
if HIGH_VIF:
    NUMERICAL_COLS = [c for c in NUMERICAL_COLS if c not in HIGH_VIF]
    X_train = X_train.drop(columns=HIGH_VIF)
    X_test  = X_test.drop(columns=HIGH_VIF)
    print(f"    Dropped: {HIGH_VIF}")
else:
    print(f"    No high-VIF features detected.")

# 7. PREPROCESSOR FACTORY
def build_preprocessor(poly_degree=None):
    num_pipe = Pipeline([("scaler", StandardScaler()), ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False))]) if poly_degree else Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))])
    return ColumnTransformer([("num", num_pipe, NUMERICAL_COLS), ("cat", cat_pipe, CATEGORICAL_COLS)])

# 8. LINEAR REGRESSION
print(f"\n[7] Training Linear Regression …")
lr_pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", LinearRegression())])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
r2_lr, mae_lr, mse_lr = r2_score(y_test, y_pred_lr), mean_absolute_error(y_test, y_pred_lr), mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
print(f"    R²={r2_lr:.4f}  MAE=₹{mae_lr:,.0f}  RMSE=₹{rmse_lr:,.0f}")

# 9. CROSS VALIDATION
print(f"\n[8] 5-Fold CV (Linear Regression):")
cv_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
for i,s in enumerate(cv_scores,1): print(f"    Fold {i}: {s:.4f}")
print(f"    Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 10. POLYNOMIAL
print(f"\n[9] Training Polynomial Regression (degree=2) …")
poly_pipeline = Pipeline([("preprocessor", build_preprocessor(poly_degree=2)), ("model", LinearRegression())])
poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)
r2_poly, mae_poly, mse_poly = r2_score(y_test, y_pred_poly), mean_absolute_error(y_test, y_pred_poly), mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
print(f"    R²={r2_poly:.4f}  MAE=₹{mae_poly:,.0f}  RMSE=₹{rmse_poly:,.0f}")

# 11. COMPARISON
print(f"\n[10] Comparison:")
print(f"    {'Model':<30} {'R²':>8}  {'MAE':>14}  {'RMSE':>14}")
print(f"    {'-'*70}")
print(f"    {'Linear Regression':<30} {r2_lr:>8.4f}  ₹{mae_lr:>12,.0f}  ₹{rmse_lr:>12,.0f}")
print(f"    {'Polynomial (deg=2)':<30} {r2_poly:>8.4f}  ₹{mae_poly:>12,.0f}  ₹{rmse_poly:>12,.0f}")

if r2_poly > r2_lr:
    best_model, best_name = poly_pipeline, "Polynomial Regression (degree=2)"
    best_r2, best_mae, best_rmse = r2_poly, mae_poly, rmse_poly
else:
    best_model, best_name = lr_pipeline, "Linear Regression"
    best_r2, best_mae, best_rmse = r2_lr, mae_lr, rmse_lr
print(f"\n    Best: {best_name}  (R²={best_r2:.4f})")

# 12. SAVE
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/best_model.pkl")
metadata = {
    "best_model_name": best_name,
    "numerical_cols": NUMERICAL_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "metrics": {
        "linear_regression": {"r2": round(r2_lr,4), "mae": round(float(mae_lr),2), "rmse": round(float(rmse_lr),2)},
        "polynomial_d2":     {"r2": round(r2_poly,4), "mae": round(float(mae_poly),2), "rmse": round(float(rmse_poly),2)},
        "best_model":        {"r2": round(best_r2,4), "mae": round(float(best_mae),2), "rmse": round(float(best_rmse),2)},
    },
    "cross_validation": {"folds":5, "scores":[round(float(s),4) for s in cv_scores], "mean_r2": round(float(cv_scores.mean()),4), "std_r2": round(float(cv_scores.std()),4)},
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "feature_options": {
        "gender":       sorted(df["gender"].unique().tolist()),
        "education":    sorted(df["education"].unique().tolist()),
        "job_title":    sorted(df["job_title"].unique().tolist()),
        "company_size": sorted(df["company_size"].unique().tolist()),
        "location":     sorted(df["location"].unique().tolist()),
        "remote_work":  sorted(df["remote_work"].unique().tolist()),
    },
    "numerical_ranges": {col: {"min": int(df[col].min()), "max": int(df[col].max())} for col in NUMERICAL_COLS},
    "vif_results": vif_dict,
    "high_vif_dropped": HIGH_VIF,
}
with open("model/metadata.json","w") as f: json.dump(metadata, f, indent=2)
print(f"\n[11] Saved: model/best_model.pkl  model/metadata.json")
print(f"\n{'='*65}\n  DONE\n{'='*65}")
