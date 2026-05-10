# -*- coding: utf-8 -*-
"""
EC 410K / EC 610I — Module 9 (Colab-ready)
Project dry-run: compare a simple baseline linear model to a tree ensemble on the SAME simulated task.

Students should adapt this pattern to their replication setting:
  - define Y, X, and (if relevant) treatment D
  - baseline: interpretable linear model
  - ML: ensemble with cross-validated tuning
  - report OOS performance honestly
"""

# !pip -q install numpy pandas scikit-learn

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

rng = np.random.default_rng(42)

n = 3000
p = 25
X = rng.normal(0, 1, size=(n, p))

beta = np.zeros(p)
beta[:5] = rng.normal(0, 0.8, size=5)
y = np.cos(X[:, 0]) + 0.4 * X[:, 1] * X[:, 2] + X @ beta * 0.05 + rng.normal(0, 1.0, size=n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

ridge = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("m", Ridge(alpha=3.0)),
    ]
)

rf = RandomForestRegressor(
    n_estimators=600,
    min_samples_leaf=2,
    random_state=0,
    n_jobs=-1,
)

rf_grid = GridSearchCV(
    rf,
    param_grid={"min_samples_leaf": [1, 2, 5, 10]},
    scoring="neg_mean_squared_error",
    cv=3,
)

ridge.fit(X_train, y_train)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

for name, model in [("Ridge (linear baseline)", ridge), ("Random forest (flexible)", best_rf)]:
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred, squared=False)
    r2 = r2_score(y_test, pred)
    print(f"{name}: RMSE={rmse:.3f}, R2={r2:.3f}")

print("\nProject writing prompts:")
print("1) What is the economic question for the ML part (prediction vs heterogeneity vs measurement)?")
print("2) What is the right baseline from the original paper?")
print("3) What would falsify your ML claim?")
