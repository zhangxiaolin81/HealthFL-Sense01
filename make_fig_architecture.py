# -*- coding: utf-8 -*-
"""Generate Fig. 2 (HealthFL-Sense system architecture) as SVG+PDF+PNG.

All on-figure text is English (Latin font, DejaVu Sans). Run from the
repository root: python make_fig_architecture.py  (writes into figs_mm/).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_mm")
os.makedirs(BASE, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(9.6, 7.2))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 960); ax.set_ylim(720, 0)  # invert so y grows downward like SVG
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


def box(x, y, w, h, fc, ec, lw=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw))


def t(x, y, s, size=12, color="#202124", bold=False, anchor="center"):
    ax.text(x, y, s, fontsize=size, color=color,
            fontweight=("bold" if bold else "normal"), ha=anchor, va="baseline")


def arrow(x1, y1, x2, y2, color, lw=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, linewidth=lw, color=color))

# ---- Title ----
t(480, 30, "Fig. 2  HealthFL-Sense System Architecture", size=15, bold=True)
t(480, 50, "User-level privacy-preserving federated multimodal mental-health sensing", size=11.5)
t(480, 68, "Raw data never leave the device; only SecAgg-masked, Top-k-sparsified \u0394w are uploaded",
  size=10.5, color="#5f6368")

# ---- Central Coordination Server ----
box(300, 80, 360, 120, "#e8f0fe", "#4285f4")
t(480, 104, "Central Coordination Server", size=14, bold=True, color="#1a73e8")
t(318, 128, "1. User-level FedAvg: weighted avg of \u0394w (by n_u)", size=10.5, anchor="left")
t(318, 148, "2. SecAgg: pairwise masks -> only \u03a3\u0394w_u visible", size=10.5, anchor="left")
t(318, 168, "3. Top-k sparsification (k=0.1): -90% comm.", size=10.5, anchor="left")
t(318, 188, "4. Optional DP (client-level guarantee)", size=10.5, anchor="left")

# ---- Communication arrows ----
arrow(415, 200, 415, 298, "#1a73e8", 2)
t(255, 252, "Broadcast global model w_global", size=11, color="#1a73e8", anchor="left")
arrow(545, 298, 545, 200, "#555555", 2)
t(562, 252, "Return secured-sparse \u0394w (SecAgg channel)", size=11, color="#555555", anchor="left")

# ---- On-device client box ----
box(70, 300, 820, 380, "#e6f4ea", "#34a853")
t(480, 326, "On-device client u (example: user_i, total 24 users)", size=14, bold=True, color="#137333")

# ---- Physiological branch (x=90, w=260, center 220) ----
box(90, 352, 260, 36, "#ffffff", "#34a853", 1.5)
t(220, 375, "Physiological input (HR/HRV/EDA)", size=10)
arrow(220, 388, 220, 406, "#555555", 1.5)
box(90, 406, 260, 36, "#ffffff", "#34a853", 1.5)
t(220, 429, "BiLSTM (bidirectional LSTM)", size=10)
arrow(220, 442, 220, 460, "#555555", 1.5)
box(90, 460, 260, 36, "#ffffff", "#34a853", 1.5)
t(220, 483, "64-dim temporal embedding", size=10)

# ---- Behavioral branch (x=610, w=280, center 750) ----
box(610, 352, 280, 36, "#ffffff", "#34a853", 1.5)
t(750, 375, "Behavior input (sleep/social/screen)", size=10)
arrow(750, 388, 750, 406, "#555555", 1.5)
box(610, 406, 280, 36, "#ffffff", "#34a853", 1.5)
t(750, 429, "MLP (multilayer perceptron)", size=10)
arrow(750, 442, 750, 460, "#555555", 1.5)
box(610, 460, 280, 36, "#ffffff", "#34a853", 1.5)
t(750, 483, "32-dim behavior embedding", size=10)

# ---- Fusion ----
arrow(220, 496, 430, 512, "#555555", 1.5)
arrow(750, 496, 550, 512, "#555555", 1.5)
box(330, 510, 300, 36, "#d9ead3", "#34a853", 1.5)
t(480, 533, "Concatenate (64+32) = 96-dim", size=10.5)
arrow(480, 546, 480, 564, "#555555", 1.5)
box(330, 564, 300, 36, "#d9ead3", "#34a853", 1.5)
t(480, 587, "Classification head -> stress/emotion (3)", size=10.5)

# ---- Local training + upload ----
box(110, 612, 820, 40, "#f1f8e9", "#34a853", 1.5)
t(520, 637, "Local training (E=2, copy w_global) -> compute \u0394w -> Top-k sparsify -> SecAgg mask -> secure upload",
  size=10.5)

# ---- Lock banner ----
t(480, 682, "Raw physiological / behavioral data stay on the device at all times, never leave the domain",
  size=12.5, bold=True, color="#c5221f")

out_svg = os.path.join(BASE, "fig_mm_architecture.svg")
out_pdf = os.path.join(BASE, "fig_mm_architecture.pdf")
out_png = os.path.join(BASE, "fig_mm_architecture.png")
fig.savefig(out_svg, format="svg", dpi=100)
fig.savefig(out_pdf, format="pdf", dpi=100)
fig.savefig(out_png, format="png", dpi=100)
print("saved:", out_svg, out_pdf, out_png)
