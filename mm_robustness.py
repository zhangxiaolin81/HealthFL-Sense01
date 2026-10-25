"""mm_robustness.py — Paper ③ robustness experiments: multi-seed + stronger
baseline (FedProx).

Purpose (in response to reviewer question 5: missing multiple seeds / stronger
baselines):
  (1) Multi-seed: repeat the main comparison over SEED in {42, 123, 2024},
      reporting mean±std;
  (2) Stronger baseline: add FedProx (Li et al. 2020, MLSys) as a non-IID-robust
      aggregation baseline alongside FedAvg, validating the default FedAvg
      choice of HealthFL-Sense;
  (3) Cover both the synthetic benchmark and the public real SWELL-KW
      corroboration.

Reuses MultiModalModel and the evaluation routines of multimodal_experiments;
only the local training loop is re-implemented here (FedProx version) to
inject the proximal term.

Run:
  python mm_robustness.py                            # synthetic benchmark, 3 seeds
  python mm_robustness.py --data swellkw_processed   # SWELL-KW real, 3 seeds
Outputs:
  results_mm/table_mm_robust.csv
  results_mm_real/table_swellkw_robust.csv
"""
import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multimodal_experiments as MM
from multimodal_experiments import (
    MultiModalModel, build_multimodal, fedavg_mm, metrics_mm,
    DEVICE, ROUNDS, CLIENT_EPOCHS, LR, N_CLASSES, BATCH,
)

MU = 0.1  # FedProx proximal-term coefficient (standard default)

VARIANTS = [
    ("Full-Late", dict(fusion="late", mode="full")),
    ("Full-Early", dict(fusion="early", mode="full")),
    ("Phys-only", dict(fusion="late", mode="phys")),
    ("Beh-only", dict(fusion="late", mode="beh")),
]


# ---------------------------------------------------------------------------
# Deterministic loader (seed-controlled)
# ---------------------------------------------------------------------------
def make_loader_seed(phys, beh, y, seed, batch=BATCH, shuffle=True):
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(MM.MMUserDataset(phys, beh, y), batch_size=batch,
                      shuffle=shuffle, generator=gen)


# ---------------------------------------------------------------------------
# FedProx local training (proximal term mu/2 * ||w - w_global||^2)
# ---------------------------------------------------------------------------
def train_client_prox(global_sd, loader, epochs, lr, mu, fusion="late",
                      mode="full", device=DEVICE):
    model = MultiModalModel(fusion=fusion, mode=mode).to(device)
    model.load_state_dict(global_sd)
    w0 = {k: v.clone().to(device) for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for phys, beh, y in loader:
            phys, beh, y = phys.to(device), beh.to(device), y.to(device)
            opt.zero_grad()
            out = model(phys, beh)
            loss = crit(out, y)
            prox = torch.tensor(0.0, device=device)
            for k, v in model.named_parameters():
                prox = prox + (mu / 2.0) * torch.sum((v - w0[k]) ** 2)
            (loss + prox).backward()
            opt.step()
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}, len(loader.dataset)


def fedprox_run(clients_train, clients_test, fusion, mode, mu, seed,
                rounds=ROUNDS, epochs=CLIENT_EPOCHS):
    names = list(clients_train)
    train_loaders = {n: make_loader_seed(*clients_train[n], seed=seed) for n in names}
    gphys = np.concatenate([clients_test[n][0] for n in names], 0)
    gbeh = np.concatenate([clients_test[n][1] for n in names], 0)
    gy = np.concatenate([clients_test[n][2] for n in names], 0)
    gloader = make_loader_seed(gphys, gbeh, gy, seed=seed, shuffle=False)
    torch.manual_seed(seed)
    global_model = MultiModalModel(fusion=fusion, mode=mode).to(DEVICE)
    sd0 = {k: v.clone() for k, v in global_model.state_dict().items()}
    for r in range(rounds):
        sds, wts = [], []
        for n in names:
            sd, w = train_client_prox(sd0, train_loaders[n], epochs, LR, mu,
                                      fusion=fusion, mode=mode)
            sds.append(sd); wts.append(w)
        agg = fedavg_mm(sds, wts)
        global_model.load_state_dict(agg)
        sd0 = {k: v.clone() for k, v in agg.items()}
    a, f, _, _ = metrics_mm(global_model, gloader)
    return a, f


# ---------------------------------------------------------------------------
# Full evaluation of a single variant under a single seed
# ---------------------------------------------------------------------------
def eval_variant(train, test, kw, seed):
    MM.SEED = seed
    torch.manual_seed(seed)
    gphys = np.concatenate([test[n][0] for n in test], 0)
    gbeh = np.concatenate([test[n][1] for n in test], 0)
    gy = np.concatenate([test[n][2] for n in test], 0)
    gloader = make_loader_seed(gphys, gbeh, gy, seed=seed, shuffle=False)

    c_a, c_f = MM.centralized_run(train, test, **kw)
    gm_fed, _ = MM.fed_run(train, test, **kw)
    f_a, f_f, _, _ = metrics_mm(gm_fed, gloader)
    p_a, p_f = fedprox_run(train, test, kw["fusion"], kw["mode"], MU, seed)
    l_a, l_f, _ = MM.local_run(train, test, **kw)
    return dict(cen=(c_a, c_f), fed=(f_a, f_f), fpx=(p_a, p_f), loc=(l_a, l_f))


def build_splits(clients, seed):
    train, test = {}, {}
    for u, (p, b, y) in clients.items():
        n_tr = int(len(y) * 0.8)
        idx = np.random.RandomState(seed + u).permutation(len(y))
        tr, te = idx[:n_tr], idx[n_tr:]
        train[u] = (p[tr], b[tr], y[tr]); test[u] = (p[te], b[te], y[te])
    return train, test


def run_seed_sweep(clients_factory, seeds, tag):
    rows = []
    agg = {v[0]: {m: [] for m in ("cen", "fed", "fpx", "loc")} for v in VARIANTS}
    for seed in seeds:
        train, test = clients_factory(seed)
        for vname, kw in VARIANTS:
            r = eval_variant(train, test, kw, seed)
            for m in ("cen", "fed", "fpx", "loc"):
                agg[vname][m].append(r[m])
            print(f"  seed={seed:>4} {vname:<10} "
                  f"cen={r['cen'][0]:.3f} fed={r['fed'][0]:.3f} "
                  f"fpx={r['fpx'][0]:.3f} loc={r['loc'][0]:.3f}")
    # Aggregate mean±std
    for vname, _ in VARIANTS:
        row = {"Method": vname}
        for m, lab in (("cen", "Cen"), ("fed", "FedAvg"), ("fpx", "FedProx"), ("loc", "Local")):
            vals = np.array(agg[vname][m])           # (n_seeds, 2) acc/f1
            a = vals[:, 0]; f1 = vals[:, 1]
            row[f"{lab}_acc_mean"] = float(a.mean()); row[f"{lab}_acc_std"] = float(a.std())
            row[f"{lab}_f1_mean"] = float(f1.mean()); row[f"{lab}_f1_std"] = float(f1.std())
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="windowed SWELL-KW directory (real run)")
    ap.add_argument("--seeds", default="42,123,2024", help="comma-separated random seeds")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.data:
        import swellkw_pipeline as SW
        print("=" * 70)
        print(f"Robustness: SWELL-KW real data ({args.data}), seeds={seeds}")
        print("=" * 70)

        def factory(seed):
            # train/test split varies with seed; all other preprocessing
            # (incl. local standardization) is fixed
            clients = SW.load_swellkw(args.data)
            train, test = {}, {}
            rng = np.random.RandomState(seed)
            for u, (p, b, y) in clients.items():
                idx = rng.permutation(len(y))
                n_tr = int(len(y) * 0.8)
                tr, te = idx[:n_tr], idx[n_tr:]
                train[u] = (p[tr], b[tr], y[tr]); test[u] = (p[te], b[te], y[te])
            return {"train": train, "test": test, "_wrap": True}

        # The SWELL-KW factory must return (train, test) tuples for eval_variant
        def factory_wrap(seed):
            d = factory(seed)
            return d["train"], d["test"]

        rows = run_seed_sweep(factory_wrap, seeds, "SWELL-KW real")
        out = os.path.join(MM.ROOT, "results_mm_real", "table_swellkw_robust.csv")
    else:
        print("=" * 70)
        print(f"Robustness: synthetic multimodal benchmark, seeds={seeds}")
        print("=" * 70)

        def factory(seed):
            clients = build_multimodal(seed=seed)
            return build_splits(clients, seed)

        rows = run_seed_sweep(factory, seeds, "synthetic")
        out = os.path.join(MM.ROOT, "results_mm", "table_mm_robust.csv")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
