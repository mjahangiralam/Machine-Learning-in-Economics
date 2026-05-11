# -*- coding: utf-8 -*-
"""
EC 410K / EC 610I — Module 10 (Colab-ready)
Optional: simple poster layout mockup (matplotlib).

This is NOT a replacement for PowerPoint/Illustrator/Canva, but helps students think in panels.
"""

# !pip -q install matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(11, 8.5))  # landscape-ish
ax.set_xlim(0, 11)
ax.set_ylim(0, 8.5)
ax.axis("off")

# Title banner
ax.add_patch(Rectangle((0.3, 7.4), 10.4, 0.9, linewidth=1, edgecolor="black", facecolor="#EEF4FF"))
ax.text(5.5, 7.85, "Poster Title: Replication + ML Extension (replace me)", ha="center", va="center", fontsize=14, weight="bold")
ax.text(5.5, 7.55, "Authors, Institution, Course, Date", ha="center", va="center", fontsize=10)

# Panels
panels = [
    (0.3, 4.6, 5.0, 2.6, "Motivation & Research Question"),
    (5.6, 4.6, 5.1, 2.6, "Data & Empirical Strategy (Baseline)"),
    (0.3, 1.5, 5.0, 2.9, "Main Replication Result (Figure/Table)"),
    (5.6, 1.5, 5.1, 2.9, "ML Extension: Method + Findings"),
]

for x, y, w, h, title in panels:
    ax.add_patch(Rectangle((x, y), w, h, linewidth=1, edgecolor="#333333", facecolor="white"))
    ax.text(x + 0.15, y + h - 0.25, title, ha="left", va="top", fontsize=11, weight="bold")

ax.text(0.3, 1.05, "Footer: QR code to GitHub/Colab + key takeaway in one sentence", ha="left", va="top", fontsize=9)

plt.tight_layout()
plt.show()

CHECKLIST = """
Poster / talk checklist (fill in for your project):
  [ ] One-sentence takeaway (prediction vs causal scope stated)
  [ ] Data source + sample definition
  [ ] Baseline replication result (what you matched from the paper)
  [ ] ML method name + validation strategy (no leakage)
  [ ] Figure readable at ~1 m distance (font size)
  [ ] QR or link to reproducible code
  [ ] Limitations / external validity (1–2 bullets)
"""
print(CHECKLIST)
print(
    "Tip: keep figure text >= 18–22 pt on the final printed poster; "
    "this matplotlib preview is only structural."
)
