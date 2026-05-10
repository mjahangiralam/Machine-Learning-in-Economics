# -*- coding: utf-8 -*-
"""
EC 410K / EC 610I — Module 2 (Colab-ready)
Supervised prediction + unsupervised structure (PCA, k-means).

Motivation: many economic problems look like "many predictors" (GKX-style asset pricing intuition)
or "find groups/types" (unsupervised). Data are simulated for reproducibility.
"""

# !pip -q install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(7)

# -----------------------------
# 1) Simulate "many characteristics" predicting returns
# -----------------------------
# True model (unknown to the researcher): r = beta' z + eps, with sparse-ish beta
n = 1200
p = 40
Z = rng.normal(0, 1, size=(n, p))

beta = np.zeros(p)
beta[:6] = rng.normal(0, 0.35, size=6)  # only first 6 features matter
eps = rng.normal(0, 1.0, size=n)
r = Z @ beta + eps

# -----------------------------
# 2) Supervised learning: compare Ridge vs Random Forest with CV
# -----------------------------
ridge = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=2.0, random_state=0)),
    ]
)

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    random_state=0,
    n_jobs=-1,
)

cv = KFold(n_splits=5, shuffle=True, random_state=0)

ridge_neg_mse = cross_val_score(
    ridge, Z, r, cv=cv, scoring="neg_mean_squared_error"
)
rf_neg_mse = cross_val_score(rf, Z, r, cv=cv, scoring="neg_mean_squared_error")

print("=== 5-fold CV average MSE (lower is better) ===")
print(f"Ridge: {(-ridge_neg_mse.mean()):.4f} (+/- {ridge_neg_mse.std():.4f})")
print(f"RF:    {(-rf_neg_mse.mean()):.4f} (+/- {rf_neg_mse.std():.4f})")

# Fit on full sample for a quick in-sample vs sanity check (stress: OOS is what matters)
ridge.fit(Z, r)
rf.fit(Z, r)
r_hat_ridge = ridge.predict(Z)
r_hat_rf = rf.predict(Z)

print("\n=== Full-sample fit (illustrative; not a causal claim) ===")
print(f"Ridge R2 (in-sample): {r2_score(r, r_hat_ridge):.4f}")
print(f"RF R2 (in-sample):    {r2_score(r, r_hat_rf):.4f}")

# -----------------------------
# 3) Unsupervised: PCA + k-means on standardized Z
# -----------------------------
Z_std = StandardScaler().fit_transform(Z)

pca = PCA(n_components=5, random_state=0)
scores = pca.fit_transform(Z_std)

print("\n=== PCA ===")
print("Explained variance ratio (first 5 comps):", np.round(pca.explained_variance_ratio_, 4))

km = KMeans(n_clusters=4, n_init="auto", random_state=0)
clusters = km.fit_predict(Z_std)

df = pd.DataFrame(
    {
        "PC1": scores[:, 0],
        "PC2": scores[:, 1],
        "cluster": clusters,
        "r": r,
    }
)

print("\n=== k-means clusters: mean return by cluster (descriptive only) ===")
print(df.groupby("cluster")["r"].mean().sort_values())

# -----------------------------
# 4) Plot: PCA scores colored by cluster
# -----------------------------
plt.figure(figsize=(7, 5))
for c in sorted(np.unique(clusters)):
    sub = df[df["cluster"] == c]
    plt.scatter(sub["PC1"], sub["PC2"], s=10, alpha=0.55, label=f"cluster {c}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Unsupervised structure in simulated characteristics (illustration)")
plt.legend(markerscale=2, frameon=False, ncol=2)
plt.tight_layout()
plt.show()

print(
    "\nTakeaway: supervised methods optimize prediction loss; unsupervised methods find geometry in X."
)
print("Economic meaning of clusters/PCs must be argued separately from the algorithm output.")
