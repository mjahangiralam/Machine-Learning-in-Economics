# -*- coding: utf-8 -*-
"""
EC 410K / EC 610I — Module 5 (Colab-ready)
Random forest classification + SHAP (TreeExplainer).

Synthetic credit/default-style data (not real microdata).
"""

# !pip -q install numpy pandas scikit-learn matplotlib shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import shap

rng = np.random.default_rng(99)

# -----------------------------
# 1) Simulate classification data
# -----------------------------
n = 6000
X = rng.normal(0, 1, size=(n, 8))

# latent index for default
z = (
    0.9 * X[:, 0]
    + 0.6 * X[:, 1]
    - 0.4 * (X[:, 2] ** 2)
    + 0.25 * X[:, 3] * X[:, 4]
)

p_default = 1 / (1 + np.exp(-z))
y = (rng.uniform(0, 1, size=n) < p_default).astype(int)

feature_names = [f"f{j}" for j in range(X.shape[1])]
dfX = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    dfX, y, test_size=0.25, stratify=y, random_state=0
)

model = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=5,
    random_state=0,
    n_jobs=-1,
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = model.predict(X_test)

print("=== Holdout performance ===")
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.3f}")

# -----------------------------
# 2) SHAP values (TreeExplainer)
# -----------------------------
# Use a modest background sample for speed
background = shap.sample(X_train, 300, random_state=0)
explainer = shap.TreeExplainer(model, data=background)

# Explain a subset of test points
X_exp = X_test.iloc[:400]
shap_values = explainer.shap_values(X_exp)

# For binary classification, shap returns list [class0, class1] in many versions
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

print("\nSHAP values computed for", X_exp.shape[0], "test observations.")

shap.summary_plot(sv, X_exp, feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()
plt.close("all")

mean_abs = np.mean(np.abs(sv), axis=0)
order = np.argsort(-mean_abs)
print("\nMean |SHAP| (global importance proxy, this sample):")
for j in order.tolist():
    print(f"  {feature_names[int(j)]}: {mean_abs[int(j)]:.4f}")

print(
    "\nTakeaway: SHAP explains model predictions (local attributions), not causal effects."
)
