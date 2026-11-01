"""
mm_protocolB.py — Accuracy measurement of the support-agreement protocols
(Section 4.2 of paper 3).

Motivation: the main-table accuracy of Section 5 was obtained with per-client
magnitude selection, which is an upper bound and is not deployable under
secure aggregation (support sets would not agree). This script measures the
global accuracy of four support rules on the same synthetic benchmark and
quantifies the gap:

  dense       : dense FedAvg (no sparsification; reference)
  perclient   : per-client magnitude Top-k (Section 5 original setting;
                not deployable)
  protocolA   : shared static support S drawn from a public seed
                (|S| = kP, fixed across rounds)
  protocolB   : lagged global Top-k (S_t = top-k coordinates of |dW_{t-1}|;
                the deployable default of the paper)

seeds = [42, 123, 2024] (same multi-seed protocol as Section 5.1).
Run:  python mm_protocolB.py
Output: results_mm/protocolB.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multimodal_experiments as mm

RES = mm.RES
SEEDS = [42, 123, 2024]
K = 0.1
ROUNDS = mm.ROUNDS


# ---------------------------------------------------------------------------
# state_dict <-> flat vector utilities (support operations act on flat coords)
# ---------------------------------------------------------------------------
_ORDER = None


def _order(model):
    global _ORDER
    if _ORDER is None:
        _ORDER = list(model.state_dict().keys())
    return _ORDER


def sd_to_vec(sd):
    return torch.cat([sd[k].flatten() for k in _ORDER])


def vec_to_sd(vec, ref_sd):
    out, i = {}, 0
    for k in _ORDER:
        n = ref_sd[k].numel()
        out[k] = vec[i:i + n].view_as(ref_sd[k]).clone()
        i += n
    return out


def topk_indices(vec, m):
    return torch.topk(vec.abs(), m).indices


# ---------------------------------------------------------------------------
# Federated training under the four support rules
# ---------------------------------------------------------------------------
def fed_run_support(train, test, mode, k=K, rounds=ROUNDS):
    """mode in {dense, perclient, protocolA, protocolB}. Returns (acc, macroF1, acc_curve)."""
    names = list(train)
    train_loaders = {n: mm.make_loader(*train[n]) for n in names}
    gphys = np.concatenate([test[n][0] for n in names], 0)
    gbeh = np.concatenate([test[n][1] for n in names], 0)
    gy = np.concatenate([test[n][2] for n in names], 0)
    gtest = mm.make_loader(gphys, gbeh, gy, shuffle=False)

    global_model = mm.MultiModalModel().to(mm.DEVICE)
    _order(global_model)                               # initialise flat coord order
    ref_sd = {kk: v.clone() for kk, v in global_model.state_dict().items()}
    P = sum(v.numel() for v in ref_sd.values())
    m = int(round(k * P))

    rng = np.random.default_rng(777)          # public seed (round-1 support of A/B)
    static_support = torch.tensor(rng.choice(P, m, replace=False), dtype=torch.long)

    w_vec = sd_to_vec(ref_sd)
    prev_agg_delta = None
    acc_curve = []
    for r in range(rounds):
        deltas, wts = [], []
        for n in names:
            sd, w = mm.train_client_mm(global_model, train_loaders[n])
            deltas.append(sd_to_vec(sd) - w_vec)
            wts.append(w)
        total = float(sum(wts))

        if mode == "dense":
            dW = sum(d * (w / total) for d, w in zip(deltas, wts))
        elif mode == "perclient":
            masked = []
            for d in deltas:
                dm = torch.zeros_like(d)
                dm[topk_indices(d, m)] = d[topk_indices(d, m)]
                masked.append(dm)
            dW = sum(d * (w / total) for d, w in zip(masked, wts))
        elif mode == "protocolA":
            masked = []
            for d in deltas:
                dm = torch.zeros_like(d)
                dm[static_support] = d[static_support]
                masked.append(dm)
            dW = sum(d * (w / total) for d, w in zip(masked, wts))
        elif mode == "protocolB":
            support = topk_indices(prev_agg_delta, m) if prev_agg_delta is not None \
                else static_support                     # round 1: fall back to public-seed support
            masked = []
            for d in deltas:
                dm = torch.zeros_like(d)
                dm[support] = d[support]
                masked.append(dm)
            dW = sum(d * (w / total) for d, w in zip(masked, wts))
        else:
            raise ValueError(mode)

        w_vec = w_vec + dW
        prev_agg_delta = dW
        global_model.load_state_dict(vec_to_sd(w_vec, ref_sd))
        acc, _, _, _ = mm.metrics_mm(global_model, gtest)
        acc_curve.append(acc)

    acc, f1, _, _ = mm.metrics_mm(global_model, gtest)
    return acc, f1, acc_curve


def main():
    print("=" * 70)
    print("Support-agreement protocol accuracy (k=0.1, 15 rounds)")
    print("=" * 70)
    modes = ["dense", "perclient", "protocolA", "protocolB"]
    rows = []
    for s in SEEDS:
        mm.SEED = s
        np.random.seed(s)
        torch.manual_seed(s)
        clients = mm.build_multimodal(seed=s)
        train, test = {}, {}
        for u, (p, b, y) in clients.items():
            n_tr = int(len(y) * 0.8)
            idx = np.random.RandomState(s + u).permutation(len(y))
            tr, te = idx[:n_tr], idx[n_tr:]
            train[u] = (p[tr], b[tr], y[tr])
            test[u] = (p[te], b[te], y[te])
        for mode in modes:
            acc, f1, curve = fed_run_support(train, test, mode)
            rows.append([mode, s, f"{acc:.4f}", f"{f1:.4f}"])
            print(f"  [seed {s}] {mode:<10} acc={acc:.4f}  macroF1={f1:.4f}  "
                  f"(R1={curve[0]:.3f} -> R{ROUNDS}={curve[-1]:.3f})")

    df = pd.DataFrame(rows, columns=["mode", "seed", "acc", "macroF1"])
    df["acc"] = df["acc"].astype(float)
    df["macroF1"] = df["macroF1"].astype(float)
    summary = df.groupby("mode")[["acc", "macroF1"]].agg(["mean", "std"])
    print("\n" + "=" * 70)
    print("Summary (3 seeds, mean +/- std)")
    print("=" * 70)
    print(summary.round(4).to_string())
    df.to_csv(os.path.join(RES, "protocolB.csv"), index=False)
    print(f"\n[DONE] wrote {os.path.join(RES, 'protocolB.csv')}")


if __name__ == "__main__":
    main()
