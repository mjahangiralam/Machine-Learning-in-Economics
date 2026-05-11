# -*- coding: utf-8 -*-
"""
EC 410K / EC 610I — Module 4 (Colab-ready)
Double / debiased machine learning (DML) demonstration via cross-fitting.

Partially linear model:
  Y = D * theta + g(X) + u
  D = m(X) + v
with nonlinear g, m so that short regression of Y on D is biased.

We estimate theta by:
  1) cross-fitting nuisance regressions for E[Y|X] and E[D|X]
  2) regress (Y - E[Y|X]) on (D - E[D|X])
"""

# !pip -q install numpy pandas scikit-learn statsmodels matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

rng = np.random.default_rng(2026)

# -----------------------------
# 1) Simulate data
# -----------------------------
n = 4000
p = 12
X = rng.normal(0, 1, size=(n, p))

true_theta = 0.5

def g_of_x(Xmat):
    return (
        0.7 * np.sin(Xmat[:, 0])
        + 0.4 * (Xmat[:, 1] ** 2)
        + 0.3 * Xmat[:, 2] * Xmat[:, 3]
    )


def m_of_x(Xmat):
    return 0.25 * np.tanh(Xmat[:, 0] + Xmat[:, 1]) + 0.15 * Xmat[:, 4]


mX = m_of_x(X)
gX = g_of_x(X)

D = mX + rng.normal(0, 0.75, size=n)
Y = true_theta * D + gX + rng.normal(0, 0.75, size=n)

df = pd.DataFrame(X, columns=[f"x{k}" for k in range(p)])
df["D"] = D
df["Y"] = Y

# -----------------------------
# 2) Naive OLS: Y ~ D (misspecified if confounding through X)
# -----------------------------
ols_naive = sm.OLS(df["Y"], sm.add_constant(df["D"])).fit()
print("=== Naive OLS: Y ~ 1 + D ===")
print(f"coef(D) = {ols_naive.params['D']:.4f} (true theta = {true_theta})")

# OLS with *linear* controls in X (g(X) is nonlinear in the DGP — linear misspecification remains)
X_lin = sm.add_constant(pd.concat([df["D"], df[[f"x{k}" for k in range(p)]]], axis=1))
ols_lin = sm.OLS(df["Y"], X_lin).fit()
coef_d_lin = float(ols_lin.params["D"])
print("\n=== OLS: Y ~ 1 + D + linear X (all covariates enter linearly) ===")
print(f"coef(D) = {coef_d_lin:.4f} (true theta = {true_theta})")
print(
    "(If g(X) is nonlinear, controlling X linearly need not remove confounding bias in D.)"
)

# -----------------------------
# 3) DML with cross-fitting (2-fold for speed/clarity)
# -----------------------------
X_mat = df[[f"x{k}" for k in range(p)]].values
Y_vec = df["Y"].values
D_vec = df["D"].values

kf = KFold(n_splits=2, shuffle=True, random_state=0)

Y_resid = np.zeros(n)
D_resid = np.zeros(n)

for train_idx, test_idx in kf.split(X_mat):
    X_tr, X_te = X_mat[train_idx], X_mat[test_idx]
    Y_tr, Y_te = Y_vec[train_idx], Y_vec[test_idx]
    D_tr, D_te = D_vec[train_idx], D_vec[test_idx]

    mod_y = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=5,
        random_state=0,
        n_jobs=-1,
    )
    mod_d = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=5,
        random_state=0,
        n_jobs=-1,
    )

    mod_y.fit(X_tr, Y_tr)
    mod_d.fit(X_tr, D_tr)

    Y_hat_te = mod_y.predict(X_te)
    D_hat_te = mod_d.predict(X_te)

    Y_resid[test_idx] = Y_te - Y_hat_te
    D_resid[test_idx] = D_te - D_hat_te

X_second = sm.add_constant(pd.Series(D_resid, name="D_resid"), has_constant="add")
ols_dml = sm.OLS(Y_resid, X_second).fit()
print("\n=== DML second stage: regress residualized Y on residualized D ===")
theta_hat = float(ols_dml.params["D_resid"])
print(f"coef(D_resid) = {theta_hat:.4f} (true theta = {true_theta})")
print(ols_dml.summary().tables[1])

# -----------------------------
# 4) Quick plot: residuals scatter
# -----------------------------
plt.figure(figsize=(6, 5))
plt.scatter(D_resid, Y_resid, s=8, alpha=0.25)
plt.xlabel("D - E[D|X] (holdout-fold predictions)")
plt.ylabel("Y - E[Y|X] (holdout-fold predictions)")
plt.title("DML orthogonalized variation (illustration)")
plt.tight_layout()
plt.show()

print(
    "\nTakeaway: flexible models for E[Y|X] and E[D|X] can reduce confounding bias from g(X),m(X)."
)
print("Real projects need careful design, more folds, and appropriate inference (standard errors).")
