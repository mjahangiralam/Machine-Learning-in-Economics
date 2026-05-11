# -*- coding: utf-8 -*-
"""
EC 410K / EC 610I — Module 7 (Colab-ready)
TF-IDF + logistic regression for document classification (simulated central-bank-ish snippets).

No external text downloads required.
"""

# !pip -q install numpy pandas scikit-learn

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

rng = np.random.default_rng(5)

hawkish = [
    "inflation remains elevated and persistent",
    "the committee is prepared to raise rates further if needed",
    "price stability is the foundation for sustained maximum employment",
    "underlying inflation pressures are still too high",
    "we will keep policy restrictive until we are confident inflation is moving down",
]

dovish = [
    "inflation has moderated though risks remain two-sided",
    "the committee will proceed carefully in determining additional firming",
    "financial conditions have tightened which may weigh on activity",
    "we are attentive to the cumulative effects of policy tightening",
    "we need balance between returning inflation to target and avoiding unnecessary harm",
]

templates = [hawkish, dovish]


def make_corpus(n_per_class=900):
    texts = []
    labels = []
    for y, bank in enumerate([0, 1]):
        sents = templates[y]
        for _ in range(n_per_class):
            k = int(rng.integers(2, 4))
            parts = list(rng.choice(sents, size=k, replace=True))
            # add mild noise words
            noise = ["data", "forecast", "outlook", "risks", "labor", "growth"]
            parts += list(rng.choice(noise, size=int(rng.integers(0, 3)), replace=True))
            rng.shuffle(parts)
            texts.append(" ".join(parts))
            labels.append(y)
    return texts, labels


texts, y = make_corpus()
df = pd.DataFrame({"text": texts, "y": y})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["y"], test_size=0.25, stratify=df["y"], random_state=0
)

vec = TfidfVectorizer(
    max_features=4000,
    ngram_range=(1, 2),
    min_df=5,
)

Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)
print(f"TF-IDF matrix shape (train): {Xtr.shape} — fit vectorizer on train only to avoid leakage.")

clf = LogisticRegression(max_iter=2000, C=2.0)
clf.fit(Xtr, y_train)

proba = clf.predict_proba(Xte)[:, 1]
pred = clf.predict(Xte)

print("=== Holdout metrics (label: 1 dovish vs 0 hawkish in this toy generator) ===")
print(classification_report(y_test, pred, digits=3))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
print("\nConfusion matrix [rows=true 0,1; cols=pred 0,1]:")
print(confusion_matrix(y_test, pred))

# Show most hawkish/dovish terms by linear coefficients
vocab = np.array(vec.get_feature_names_out())
coef = clf.coef_.ravel()
top_dov = vocab[np.argsort(-coef)[:12]]
top_hawk = vocab[np.argsort(coef)[:12]]

print("\nTop terms associated with dovish label (higher log-odds):")
print(", ".join(top_dov))
print("\nTop terms associated with hawkish label (lower log-odds):")
print(", ".join(top_hawk))

print(
    "\nTakeaway: this is measurement/classification from text, not identification of monetary shocks."
)
print("Aruoba-Drechsel-style work adds economic structure + identification on top of text processing.")
