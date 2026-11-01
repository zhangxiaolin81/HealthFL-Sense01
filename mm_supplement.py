"""
mm_supplement.py — Online-supplement table generator for paper 3.

Produces the two supplement files referenced in the manuscript:
  1. results_mm/table_user_gains.csv   — full version of Table 2:
     per-user federated-vs-local gains for all 24 users plus summary
     statistics (mean, std, count of users whose accuracy decreases).
  2. results_mm/table_mm_perclass.csv  — per-class F1 breakdown of Table 1
     (Section 5.1): FedAvg and Local isolated training for the four
     fusion/unimodal variants.

The training setup is identical to multimodal_experiments.py (same seed=42,
same generator, same splits, same hyper-parameters), so the per-class F1
values belong to the same experiment family as the main table.

Run:  python mm_supplement.py
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from multimodal_experiments import (
    build_multimodal, fed_run, metrics_mm, make_loader,
    MultiModalModel, ROUNDS, CLIENT_EPOCHS, LR, DEVICE, SEED, RES,
)

N_CLASSES = 3


def local_run_preds(clients_train, clients_test, fusion="late", mode="full"):
    """Train one model per client, evaluate all on the global test set.
    Returns (mean acc, per-class F1 averaged over client models)."""
    import torch.nn as nn
    names = list(clients_train)
    gphys = np.concatenate([clients_test[n][0] for n in names], 0)
    gbeh = np.concatenate([clients_test[n][1] for n in names], 0)
    gy = np.concatenate([clients_test[n][2] for n in names], 0)
    te_loader = make_loader(gphys, gbeh, gy, shuffle=False)
    accs, all_labels, all_preds = [], [], []
    for n in names:
        m = MultiModalModel(fusion=fusion, mode=mode).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        crit = nn.CrossEntropyLoss()
        m.train()
        for _ in range(ROUNDS * CLIENT_EPOCHS):
            for phys, beh, y in make_loader(*clients_train[n]):
                phys, beh, y = phys.to(DEVICE), beh.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(); loss = crit(m(phys, beh), y); loss.backward(); opt.step()
        a, f, lab, prd = metrics_mm(m, te_loader)
        accs.append(a); all_labels.append(lab); all_preds.append(prd)
    # per-class F1 averaged over the per-client models, matching the
    # mean-accuracy convention of the Local column in Table 1
    pc = np.mean([
        f1_score(l, p, average=None, labels=list(range(N_CLASSES)))
        for l, p in zip(all_labels, all_preds)
    ], axis=0)
    return float(np.mean(accs)), pc


def perclass_of(model, loader):
    _, _, lab, prd = metrics_mm(model, loader)
    return f1_score(lab, prd, average=None, labels=list(range(N_CLASSES)))


def make_user_gains():
    """Derive the full Table 2 supplement from the released per-user table."""
    src = os.path.join(RES, "table_mm_per_user.csv")
    df = pd.read_csv(src)
    df["gain_pp"] = df["gain_pp"].astype(float)
    neg = df[df.gain_pp < 0]
    pos = df[df.gain_pp >= 0]
    summary = pd.DataFrame([
        dict(item="n_users", local_acc=len(df), fed_acc=np.nan, gain_pp=np.nan),
        dict(item="mean_gain_pp", local_acc=np.nan, fed_acc=np.nan,
             gain_pp=round(df.gain_pp.mean(), 2)),
        dict(item="std_gain_pp", local_acc=np.nan, fed_acc=np.nan,
             gain_pp=round(df.gain_pp.std(ddof=1), 2)),
        dict(item="median_gain_pp", local_acc=np.nan, fed_acc=np.nan,
             gain_pp=round(df.gain_pp.median(), 2)),
        dict(item="max_gain_pp", local_acc=np.nan, fed_acc=np.nan,
             gain_pp=round(df.gain_pp.max(), 2)),
        dict(item="min_gain_pp", local_acc=np.nan, fed_acc=np.nan,
             gain_pp=round(df.gain_pp.min(), 2)),
        dict(item="count_negative_gain", local_acc=len(neg), fed_acc=np.nan, gain_pp=np.nan),
        dict(item="count_nonnegative_gain", local_acc=len(pos), fed_acc=np.nan, gain_pp=np.nan),
        dict(item="negative_gain_users",
             local_acc=",".join(f"user_{u}" for u in neg.user), fed_acc=np.nan, gain_pp=np.nan),
    ])
    out = pd.concat([summary, df.rename(columns={"user": "item"})], axis=0,
                    ignore_index=True)
    path = os.path.join(RES, "table_user_gains.csv")
    out.to_csv(path, index=False)
    print(f"[1/2] wrote {path}")
    print(f"      mean={df.gain_pp.mean():+.2f}pp  neg={len(neg)}/24  "
          f"max=+{df.gain_pp.max():.2f}pp")
    return path


def make_perclass():
    """Reproduce the Section 5.1 experiment (seed=42) and save per-class F1."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    clients = build_multimodal()
    train, test = {}, {}
    for u, (p, b, y) in clients.items():
        n_tr = int(len(y) * 0.8)
        idx = np.random.RandomState(SEED + u).permutation(len(y))
        tr, te = idx[:n_tr], idx[n_tr:]
        train[u] = (p[tr], b[tr], y[tr])
        test[u] = (p[te], b[te], y[te])

    gphys = np.concatenate([test[n][0] for n in test], 0)
    gbeh = np.concatenate([test[n][1] for n in test], 0)
    gy = np.concatenate([test[n][2] for n in test], 0)
    gtest = make_loader(gphys, gbeh, gy, shuffle=False)

    variants = [
        ("Full-Late", dict(fusion="late", mode="full")),
        ("Full-Early", dict(fusion="early", mode="full")),
        ("Phys-only", dict(fusion="late", mode="phys")),
        ("Beh-only", dict(fusion="late", mode="beh")),
    ]
    rows = []
    for vname, kw in variants:
        print(f"  running {vname} ...", flush=True)
        gm, _ = fed_run(train, test, **kw)
        fed_acc, fed_f1, _, _ = metrics_mm(gm, gtest)
        fed_pc = perclass_of(gm, gtest)
        loc_acc, loc_pc = local_run_preds(train, test, **kw)
        rows.append(dict(
            Method=vname,
            FedAvg_acc=round(fed_acc, 3),
            FedAvg_macroF1=round(fed_f1, 3),
            **{f"FedAvg_F1_class{i}": round(float(fed_pc[i]), 3) for i in range(N_CLASSES)},
            Local_acc=round(loc_acc, 3),
            **{f"Local_F1_class{i}": round(float(loc_pc[i]), 3) for i in range(N_CLASSES)},
        ))
        print(f"    FedAvg acc={fed_acc:.3f}  per-class={np.round(fed_pc, 3)}", flush=True)
        print(f"    Local  acc={loc_acc:.3f}  per-class={np.round(loc_pc, 3)}", flush=True)
    df = pd.DataFrame(rows)
    path = os.path.join(RES, "table_mm_perclass.csv")
    df.to_csv(path, index=False)
    print(f"[2/2] wrote {path}")
    return path


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    print("=" * 60)
    print("Supplement tables (table_user_gains / table_mm_perclass)")
    print("=" * 60)
    make_user_gains()
    make_perclass()
    print("done.")
