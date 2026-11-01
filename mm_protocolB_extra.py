"""
mm_protocolB_extra.py — Supplementary comparisons for Protocol B (matched seeds):
  1) Local isolated-training baseline under the exact same data splits and
     seeds as the protocolB experiment (apples-to-apples federation margin)
  2) Accuracy of dense / per-client / Protocol B at k=0.3 (second point of
     the accuracy-communication trade-off)
Run:  python mm_protocolB_extra.py
Output: results_mm/protocolB_extra.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multimodal_experiments as mm
import mm_protocolB as pb

RES = mm.RES
SEEDS = [42, 123, 2024]


def make_split(seed):
    clients = mm.build_multimodal(seed=seed)
    train, test = {}, {}
    for u, (p, b, y) in clients.items():
        n_tr = int(len(y) * 0.8)
        idx = np.random.RandomState(seed + u).permutation(len(y))
        train[u] = (p[idx[:n_tr]], b[idx[:n_tr]], y[idx[:n_tr]])
        test[u] = (p[idx[n_tr:]], b[idx[n_tr:]], y[idx[n_tr:]])
    return train, test


def main():
    rows = []
    for s in SEEDS:
        mm.SEED = s
        np.random.seed(s)
        torch.manual_seed(s)
        train, test = make_split(s)

        # Local isolated training (matched-seed baseline)
        l_a, l_f, _ = mm.local_run(train, test, fusion="late", mode="full")
        rows.append(["local", s, 0.1, l_a, l_f])
        print(f"  [seed {s}] local        acc={l_a:.4f}  macroF1={l_f:.4f}", flush=True)

        # The three federated mechanisms at k=0.3
        for mode in ["dense", "perclient", "protocolB"]:
            acc, f1, _ = pb.fed_run_support(train, test, mode, k=0.3)
            rows.append([mode, s, 0.3, acc, f1])
            print(f"  [seed {s}] {mode:<10} k=0.3  acc={acc:.4f}  macroF1={f1:.4f}", flush=True)

    df = pd.DataFrame(rows, columns=["mode", "seed", "k", "acc", "macroF1"])
    summary = df.groupby(["mode", "k"])[["acc", "macroF1"]].agg(["mean", "std"])
    print("\n" + "=" * 70)
    print("Supplementary comparison summary")
    print("=" * 70)
    print(summary.round(4).to_string())
    df.to_csv(os.path.join(RES, "protocolB_extra.csv"), index=False)
    print(f"\n[DONE] wrote {os.path.join(RES, 'protocolB_extra.csv')}")


if __name__ == "__main__":
    main()
