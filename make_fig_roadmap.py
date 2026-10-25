# -*- coding: utf-8 -*-
"""Generate Fig. 1 (HealthFL-Sense technical roadmap) as SVG+PDF+PNG.

All on-figure text is English (Latin font, DejaVu Sans). Run from the
repository root: python make_fig_roadmap.py  (writes into figs_mm/).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_mm")
os.makedirs(BASE, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(13.0, 5.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.text(0.5, 0.96, "Fig. 1  HealthFL-Sense Research Technical Roadmap", ha="center", va="top",
        fontsize=15, fontweight="bold", color="#202124")

stages = [
    ("1. Problem Definition", "#1a73e8", [
        "Student mental-health issues",
        "Mature multimodal sensing",
        "Two constraints: sensitivity + silos",
        "Naive FL: gradient-inversion flaw",
    ]),
    ("2. Model Formulation", "#0b8043", [
        "User-level data model (Sec. 3)",
        "Optimization objective",
        "Threat model: honest-but-curious",
        "Three inversion threats defined",
    ]),
    ("3. Method & Mechanism", "#a14200", [
        "Late fusion on-device (4.1)",
        "SecAgg + Top-k (4.2)",
        "DP optional (client-level)",
        "Runnable before theory proof",
    ]),
    ("4. Analytical Framing", "#c5221f", [
        "Info gain: fusion > uni-modal",
        "Convergence: user-level non-IID",
        "Inversion threat worse in high-dim",
        "Communication size controllable",
    ]),
    ("5. Empirical Validation", "#9334e6", [
        "Main compare -> info gain (RQ1)",
        "User gain + convergence (RQ2/3)",
        "Comm + defense -> threat size (RQ4)",
        "Synthetic benchmark tests predictions",
    ]),
]

n = len(stages)
bw = 0.165
gap = (1.0 - bw * n) / (n + 1)
y0, bh = 0.20, 0.60
xs = []
for i, (title, color, items) in enumerate(stages):
    x = gap * (i + 1) + bw * i
    xs.append(x)
    box = FancyBboxPatch((x, y0), bw, bh,
                         boxstyle="round,pad=0.012,rounding_size=0.02",
                         linewidth=1.6, edgecolor=color, facecolor=color + "14")
    ax.add_patch(box)
    ax.add_patch(FancyBboxPatch((x + 0.004, y0 + bh - 0.10), bw - 0.008, 0.075,
                 boxstyle="round,pad=0.004,rounding_size=0.012",
                 linewidth=0, facecolor=color))
    ax.text(x + bw / 2, y0 + bh - 0.0625, title, ha="center", va="center",
            fontsize=7.6, fontweight="bold", color="white")
    yt = y0 + bh - 0.17
    for it in items:
        ax.text(x + bw / 2, yt, "- " + it, ha="center", va="center",
                fontsize=5.8, color="#202124")
        yt -= 0.092

for i in range(n - 1):
    xa = xs[i] + bw
    xb = xs[i + 1]
    ax.add_patch(FancyArrowPatch((xa + 0.005, y0 + bh / 2), (xb - 0.012, y0 + bh / 2),
                 arrowstyle="-|>", mutation_scale=16, linewidth=2.0,
                 color="#5f6368"))

ax.text(0.5, 0.085,
        "The five stages form a 'problem -> model -> method -> analysis -> validation' closed loop; the runnable method (Sec. 4.1-4.2) precedes its analytical framing (Sec. 4.3)",
        ha="center", va="center", fontsize=8.8, color="#5f6368", style="italic")

out_svg = os.path.join(BASE, "fig_mm_roadmap.svg")
out_pdf = os.path.join(BASE, "fig_mm_roadmap.pdf")
out_png = os.path.join(BASE, "fig_mm_roadmap.png")
fig.savefig(out_svg, format="svg", dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight")
print("saved:", out_svg, out_pdf, out_png)
