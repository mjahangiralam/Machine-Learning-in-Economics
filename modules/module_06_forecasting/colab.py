# -*- coding: utf-8 -*-
"""
EC 410K / EC 610I — Module 6 (Colab-ready)
Pseudo out-of-sample forecasting with lag features.

We simulate a stationary-ish macro series (AR + noise) and compare:
  - OLS on lag features
  - Histogram-based Gradient Boosting (sklearn)

This avoids GPU/torch dependencies while teaching the workflow used in many ML forecasting papers.
"""

# !pip -q install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(11)

# -----------------------------
# 1) Simulate AR(2)-like series with drift + seasonal-ish component
# -----------------------------
T = 900
y = np.zeros(T)
eps = rng.normal(0, 0.35, size=T)
for t in range(2, T):
    y[t] = 0.15 + 0.55 * y[t - 1] - 0.25 * y[t - 2] + 0.08 * np.sin(2 * np.pi * t / 24) + eps[t]

# -----------------------------
# 2) Build supervised dataset with lags
# -----------------------------
max_lag = 12
rows = []
for t in range(max_lag, T):
    lags = [y[t - k] for k in range(1, max_lag + 1)]
    rows.append([t, y[t], *lags])

cols = ["t", "y"] + [f"L{k}" for k in range(1, max_lag + 1)]
data = pd.DataFrame(rows, columns=cols)

# -----------------------------
# 3) Rolling-origin evaluation (one-step-ahead)
# -----------------------------
start_train = 200
test_start = 650

X_all = data[[f"L{k}" for k in range(1, max_lag + 1)]].values
y_all = data["y"].values

y_hat_ols = np.full(len(y_all), np.nan)
y_hat_hgb = np.full(len(y_all), np.nan)

for s in range(test_start, len(y_all)):
    X_tr = X_all[start_train:s]
    y_tr = y_all[start_train:s]
    X_te = X_all[s : s + 1]

    ols = LinearRegression().fit(X_tr, y_tr)
    y_hat_ols[s] = ols.predict(X_te)[0]

    hgb = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        random_state=0,
    )
    hgb.fit(X_tr, y_tr)
    y_hat_hgb[s] = hgb.predict(X_te)[0]

idx = np.arange(test_start, len(y_all))
rmse_ols = mean_squared_error(y_all[idx], y_hat_ols[idx], squared=False)
rmse_hgb = mean_squared_error(y_all[idx], y_hat_hgb[idx], squared=False)

print("=== One-step-ahead pseudo OOS (rolling origin) ===")
print(f"OLS RMSE: {rmse_ols:.4f}")
print(f"HGB  RMSE: {rmse_hgb:.4f}")

plt.figure(figsize=(9, 4))
plt.plot(data["t"], y_all, label="y (simulated)", linewidth=1)
plt.plot(data["t"], y_hat_ols, label="OLS forecast (rolling)", linewidth=1, alpha=0.85)
plt.plot(data["t"], y_hat_hgb, label="HGB forecast (rolling)", linewidth=1, alpha=0.85)
plt.axvline(test_start, color="black", linestyle="--", linewidth=1, alpha=0.6)
plt.xlabel("time")
plt.ylabel("level")
plt.title("Illustration: rolling one-step-ahead forecasts")
plt.legend(frameon=False, ncol=3)
plt.tight_layout()
plt.show()

print(
    "\nTakeaway: ML forecasting is mostly about careful time-series validation and leakage control."
)
print("Compare to baselines; do not trust in-sample fit.")
