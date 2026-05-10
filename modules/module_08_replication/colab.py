# -*- coding: utf-8 -*-
"""
EC 410K / EC 610I — Module 8 (Colab-ready)
Toy replication exercise: recover a published-style coefficient from a known DGP.

Students practice:
  - pre-register what they are trying to match (theta = 2.0)
  - estimate with OLS
  - report mismatch when the "published" number is perturbed (illustrating replication drift)
"""

# !pip -q install numpy pandas statsmodels

import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(0)

# "True" causal parameter in simulation
TRUE_THETA = 2.0

# Generate data: Y = theta * D + controls' gamma + noise
n = 5000
X1 = rng.normal(0, 1, size=n)
X2 = rng.normal(0, 1, size=n)
D = 0.35 * X1 + 0.10 * X2 + rng.normal(0, 1, size=n)
Y = TRUE_THETA * D + 0.5 * X1 - 0.25 * X2 + rng.normal(0, 1, size=n)

df = pd.DataFrame({"Y": Y, "D": D, "X1": X1, "X2": X2})

# "Published" (fake) table value with a typo / different sample assumption
PUBLISHED_COEF = 2.08

model = sm.OLS(df["Y"], sm.add_constant(df[["D", "X1", "X2"]])).fit()
est = float(model.params["D"])
se = float(model.bse["D"])

print("=== Toy replication memo ===")
print(f"Target estimand (simulation truth): {TRUE_THETA:.3f}")
print(f"OLS estimate of theta (long regression): {est:.3f} (SE={se:.3f})")
print(f"Fake 'published' coefficient: {PUBLISHED_COEF:.3f}")
print(f"Difference (estimate - published): {est - PUBLISHED_COEF:.3f}")

print("\nDiscussion prompts:")
print("- If your estimate differs from the paper, is it sampling variation, coding, or specification?")
print("- What diagnostics would you run next (sample restrictions, clustering, weights)?")
