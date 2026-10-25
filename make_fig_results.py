# -*- coding: utf-8 -*-
"""make_fig_results.py — re-render the paper-③ result figures (Fig. 3 per-user
gain / Fig. 4 convergence curve) as vector graphics.

Reuses the model and data generation of multimodal_experiments, recomputes the
curves and per-user gains, and outputs PDF + SVG (vector, recommended for
submission) plus a refreshed PNG. Run from the repository root after
multimodal_experiments.py; writes into figs_mm/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

import multimodal_experiments as mm

FIG = mm.FIG
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


def recompute():
    clients = mm.build_multimodal()
    train, test = {}, {}
    for u, (p, b, y) in clients.items():
        n_tr = int(len(y) * 0.8)
        idx = np.random.RandomState(mm.SEED + u).permutation(len(y))
        tr, te = idx[:n_tr], idx[n_tr:]
        train[u] = (p[tr], b[tr], y[tr])
        test[u] = (p[te], b[te], y[te])

    gphys_all = np.concatenate([test[n][0] for n in test], 0)
    gbeh_all = np.concatenate([test[n][1] for n in test], 0)
    gy_all = np.concatenate([test[n][2] for n in test], 0)

    variants = [
        ("Full-Late", dict(fusion="late", mode="full")),
        ("Full-Early", dict(fusion="early", mode="full")),
        ("Phys-only", dict(fusion="late", mode="phys")),
        ("Beh-only", dict(fusion="late", mode="beh")),
    ]
    c_a0, c_f0 = mm.centralized_run(train, test, fusion="late", mode="full")
    g_model, curve0 = mm.fed_run(train, test, fusion="late", mode="full")
    f_a0, f_f0, _, _ = mm.metrics_mm(g_model, mm.make_loader(gphys_all, gbeh_all, gy_all, shuffle=False))
    l_a0, l_f0, per_user_local = mm.local_run(train, test, fusion="late", mode="full")

    fed_acc_curves = {}
    for vname, kw in variants:
        if vname == "Full-Late":
            c_a, c_f, f_a, f_f, l_a, l_f, curve = c_a0, c_f0, f_a0, f_f0, l_a0, l_f0, curve0
        else:
            c_a, c_f = mm.centralized_run(train, test, **kw)
            gm_v, curve = mm.fed_run(train, test, **kw)
            f_a, f_f, _, _ = mm.metrics_mm(gm_v, mm.make_loader(gphys_all, gbeh_all, gy_all, shuffle=False))
            l_a, l_f, _ = mm.local_run(train, test, **kw)
        fed_acc_curves[vname] = curve

    # per-user gain
    fed_per_user = {}
    for u in test:
        a, _, _, _ = mm.metrics_mm(g_model, mm.make_loader(*test[u], shuffle=False))
        fed_per_user[u] = a
    gains = [(u, per_user_local[u], fed_per_user[u],
              (fed_per_user[u] - per_user_local[u]) * 100) for u in test]
    gains_sorted = sorted(gains, key=lambda t: t[3])
    return fed_acc_curves, c_a0, gains_sorted


def save_fig(fig, base_name):
    pdf = os.path.join(FIG, base_name + ".pdf")
    svg = os.path.join(FIG, base_name + ".svg")
    png = os.path.join(FIG, base_name + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", pdf, "|", svg, "|", png)


def main():
    fed_acc_curves, c_a0, gains_sorted = recompute()

    # Fig. 4 convergence curve
    fig3 = plt.figure(figsize=(6, 4))
    for vname, curve in fed_acc_curves.items():
        plt.plot(range(1, mm.ROUNDS + 1), curve, marker="o", label=vname)
    plt.axhline(c_a0, ls="--", color="gray", label="Centralized")
    plt.xlabel("Round")
    plt.ylabel("Global Accuracy")
    plt.title("Federated Convergence (multi-modal)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig3, "fig_mm_convergence")

    # Fig. 3 per-user gain
    us = [f"u{u}" for u, _, _, _ in gains_sorted]
    ga = [g for _, _, _, g in gains_sorted]
    fig2 = plt.figure(figsize=(8, 3.5))
    colors = ["#c0392b" if x < 0 else "#27ae60" for x in ga]
    plt.bar(us, ga, color=colors)
    plt.axhline(0, color="black", lw=0.8)
    plt.xlabel("User")
    plt.ylabel("Fed - Local (pp)")
    plt.title("Per-User Gain (Federated vs Local)")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    save_fig(fig2, "fig_mm_per_user")

    print("done.")


if __name__ == "__main__":
    main()
