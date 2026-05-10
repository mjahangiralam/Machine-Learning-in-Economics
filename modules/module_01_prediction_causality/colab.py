# -*- coding: utf-8 -*-
"""
EC 410K / EC 610I — Module 1 (Colab-ready)
Prediction vs. causal estimands on simulated data.

Run top-to-bottom in Google Colab. This script uses only numpy/pandas/sklearn/statsmodels.
"""

# --- Optional: Colab install (uncomment if needed) ---
# !pip -q install numpy pandas scikit-learn statsmodels matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

# -----------------------------
# 1) Simulate a labor-market-like dataset
# -----------------------------
# True structural idea (for simulation only):
#   Y = baseline ability W + returns-to-schooling * S + noise
#   Schooling S is correlated with W (confounding) unless we randomize S.
rng = np.random.default_rng(42)
n = 5000

W = rng.normal(0, 1, size=n)  # latent "ability" (often unobserved in real data)
S = 0.35 * W + rng.normal(0, 1, size=n)  # schooling is endogenous (depends on ability)
Y = 0.8 * W + 0.25 * S + rng.normal(0, 0.75, size=n)

df = pd.DataFrame({"Y": Y, "S": S, "W": W})

# In applied work, you often do NOT observe W. Here we show two researcher datasets:
#   - df_obs: only (Y, S)  -> naive regression mixes causal effect with confounding
#   - df_full: (Y, S, W)   -> controlling W can recover ~causal coefficient of S (linear case)

df_obs = df[["Y", "S"]].copy()
df_full = df.copy()

# -----------------------------
# 2) "Prediction" task: predict Y from observed features
# -----------------------------
# Use only what a researcher might observe without W (still predictive):
X1 = df_obs["S"].values.reshape(-1, 1)
y = df_obs["Y"].values

X_train, X_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.25, random_state=0
)

lr_pred = LinearRegression().fit(X_train, y_train)
y_hat = lr_pred.predict(X_test)

print("=== Prediction performance (features: S only) ===")
print(f"Test RMSE: {mean_squared_error(y_test, y_hat, squared=False):.4f}")
print(f"Test R2:   {r2_score(y_test, y_hat):.4f}")

# -----------------------------
# 3) "Causal" comparison: short vs long regression (when W is observed)
# -----------------------------
# Short regression (omit confounder): coef on S is biased for true structural effect of S
X_short = sm.add_constant(df_full["S"])
model_short = sm.OLS(df_full["Y"], X_short).fit()

# Long regression (include confounder): closer to data-generating 0.25 for S (finite-sample noise remains)
X_long = sm.add_constant(df_full[["S", "W"]])
model_long = sm.OLS(df_full["Y"], X_long).fit()

print("\n=== OLS: short regression (omit W) ===")
print(model_short.summary().tables[1])

print("\n=== OLS: long regression (include W) ===")
print(model_long.summary().tables[1])

print("\nTakeaway:")
print("- A model can predict well yet still not identify a causal effect of S on Y.")
print("- Adding confounders changes interpretation; ML flexibility does not replace design.")

# -----------------------------
# 4) Simple figure: predicted vs actual on holdout (prediction angle)
# -----------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_hat, alpha=0.35, s=10)
mx = float(max(y_test.max(), y_hat.max()))
mn = float(min(y_test.min(), y_hat.min()))
plt.plot([mn, mx], [mn, mx], linestyle="--", color="black", linewidth=1)
plt.xlabel("Actual Y (holdout)")
plt.ylabel("Predicted Y (holdout)")
plt.title("Module 1 demo: prediction is not causation")
plt.tight_layout()
plt.show()
