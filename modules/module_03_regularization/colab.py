# -*- coding: utf-8 -*-
"""
EC 410K / EC 610I — Module 3 (Colab-ready)
Ridge and Lasso for high-dimensional linear prediction.

We simulate Y = X beta + noise where beta is sparse. Students compare CV-chosen Ridge/Lasso.
"""

# !pip -q install numpy pandas scikit-learn matplotlib

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

rng = np.random.default_rng(123)

n = 800
p = 60
X = rng.normal(0, 1, size=(n, p))

beta = np.zeros(p)
beta[:8] = rng.normal(0, 0.6, size=8)  # sparse truth
y = X @ beta + rng.normal(0, 1.5, size=n)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

ridge_pipe = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        (
            "model",
            Ridge(max_iter=20000, random_state=0),
        ),
    ]
)

lasso_pipe = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        (
            "model",
            Lasso(max_iter=20000, random_state=0),
        ),
    ]
)

ridge_grid = GridSearchCV(
    ridge_pipe,
    param_grid={"model__alpha": np.logspace(-3, 3, 25)},
    scoring="neg_mean_squared_error",
    cv=5,
)

lasso_grid = GridSearchCV(
    lasso_pipe,
    param_grid={"model__alpha": np.logspace(-3, -0.5, 25)},
    scoring="neg_mean_squared_error",
    cv=5,
)

ridge_grid.fit(X_train, y_train)
lasso_grid.fit(X_train, y_train)

ridge_hat = ridge_grid.predict(X_test)
lasso_hat = lasso_grid.predict(X_test)

print("=== Best hyperparameters (CV) ===")
print(f"Ridge alpha: {ridge_grid.best_params_}")
print(f"Lasso alpha: {lasso_grid.best_params_}")

print("\n=== Holdout performance ===")
print(
    f"Ridge RMSE: {mean_squared_error(y_test, ridge_hat, squared=False):.4f} | "
    f"R2: {r2_score(y_test, ridge_hat):.4f}"
)
print(
    f"Lasso RMSE: {mean_squared_error(y_test, lasso_hat, squared=False):.4f} | "
    f"R2: {r2_score(y_test, lasso_hat):.4f}"
)

# Sparsity check: compare estimated coefficients to truth (on training+standardization pipeline)
lasso_fit = lasso_grid.best_estimator_
coef = lasso_fit.named_steps["model"].coef_
print(f"\nLasso nonzeros: {np.sum(coef != 0)} / {p}")
print(f"True nonzeros: {np.sum(beta != 0)} / {p}")

plt.figure(figsize=(7, 4))
plt.plot(beta, label="true beta", linewidth=1)
plt.plot(coef, label="lasso estimate", linewidth=1)
plt.xlabel("j (feature index)")
plt.ylabel("coefficient")
plt.title("Sparse truth vs Lasso estimate (illustration)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

print(
    "\nTakeaway: L1 can select variables; L2 shrinks. Choose lambda with CV; interpret causally only with design."
)
