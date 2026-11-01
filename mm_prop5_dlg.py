"""
mm_prop5_dlg.py — Empirical pairing of Proposition 5 (gradient-inversion / DLG).

Design (the experiment pre-registered in Section 6.2 of the paper):
  On a trained Full-Late global model, take the true gradient g of a single
  window (phys 24x3 + beh 4) and let the attacker observe one of three views:
  V1 single  : the plain single-client update g (dense; no defense at all)
  V2 U2-sub  : with U=2, participant i computes s - dw_i = dw_j, which is
               exactly the bare gradient of client j
               (Prop. 5 corollary: SecAgg gives zero protection at U=2)
  V3 U4-mean : the SecAgg mean gbar = (1/4) sum g_u with U=4, observable only
               on the shared support S (|S| = 0.1P, top-k of |gbar|), i.e. the
               setting of Proposition 5
  Attack: DLG gradient matching (Adam, 120 iters, 2 restarts per view/window);
  metrics = input reconstruction MSE (z-score space, averaged over the 76
  input coordinates) + top-1 label recovery (random baseline 0.333).

Prediction: V1 ~ V2 (low MSE, high label recovery); V3 degrades sharply
(MSE towards the data variance, label recovery towards random).
Run:  python mm_prop5_dlg.py
Output: results_mm/prop5_dlg.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multimodal_experiments as mm

RES = mm.RES
SEED = 42
N_PROBE = 12          # number of probe windows (drawn from client 0's training windows)
ITERS = 120
RESTARTS = 2
K = 0.1

torch.set_num_threads(4)   # avoid core contention (LSTM second-order backprop is CPU-heavy)


# ---------------------------------------------------------------------------
# Train the global model (dense FedAvg, seed 42 — identical to Section 5)
# ---------------------------------------------------------------------------
def train_global_model():
    clients = mm.build_multimodal(seed=SEED)
    train, test = {}, {}
    for u, (p, b, y) in clients.items():
        n_tr = int(len(y) * 0.8)
        idx = np.random.RandomState(SEED + u).permutation(len(y))
        train[u] = (p[idx[:n_tr]], b[idx[:n_tr]], y[idx[:n_tr]])
        test[u] = (p[idx[n_tr:]], b[idx[n_tr:]], y[idx[n_tr:]])
    model, curve = mm.fed_run(train, test, fusion="late", mode="full")
    return model, train, test


# ---------------------------------------------------------------------------
# True gradient of a single window
# ---------------------------------------------------------------------------
def window_grad(model, phys, beh, y):
    model.zero_grad()
    p = torch.tensor(phys[None], dtype=torch.float32)
    b = torch.tensor(beh[None], dtype=torch.float32)
    t = torch.tensor([y], dtype=torch.long)
    out = model(p, b)
    nn.CrossEntropyLoss()(out, t).backward()
    return [pp.grad.detach().clone() for pp in model.parameters()]


# ---------------------------------------------------------------------------
# DLG attack (single window; obs_mask: global flat-coordinate bool vector,
# None = all coordinates observable)
# ---------------------------------------------------------------------------
def dlg_attack(model, real_grad, phys_true, beh_true, y_true,
               obs_mask=None, iters=ITERS, restarts=RESTARTS):
    params = [pp for pp in model.parameters()]
    shapes = [pp.shape for pp in params]
    sizes = [pp.numel() for pp in params]

    # build the per-parameter mask once
    if obs_mask is None:
        param_masks = None
    else:
        param_masks, i = [], 0
        for n, sh in zip(sizes, shapes):
            param_masks.append(obs_mask[i:i + n].view(sh))
            i += n

    phys_t = torch.tensor(phys_true[None], dtype=torch.float32)
    beh_t = torch.tensor(beh_true[None], dtype=torch.float32)
    best_mse, best_lab = float("inf"), -1
    for r in range(restarts):
        rng = np.random.default_rng(2026 + r)
        dummy_p = torch.randn(*phys_t.shape).requires_grad_(True)
        dummy_b = torch.randn(*beh_t.shape).requires_grad_(True)
        dl = int(rng.integers(0, 3))
        dummy_oh = nn.functional.one_hot(torch.tensor(dl), 3).float()[None].requires_grad_(True)
        opt = torch.optim.Adam([dummy_p, dummy_b, dummy_oh], lr=0.1)
        for _ in range(iters):
            opt.zero_grad()
            out = model(dummy_p, dummy_b)
            loss = nn.CrossEntropyLoss()(out, dummy_oh)
            g_dummy = torch.autograd.grad(loss, params, create_graph=True)
            if param_masks is None:
                d = sum(((a - b) ** 2).sum() for a, b in zip(g_dummy, real_grad))
            else:
                d = sum((((a - b) * mk) ** 2).sum()
                        for a, b, mk in zip(g_dummy, real_grad, param_masks))
            opt.zero_grad()
            d.backward()
            opt.step()
        with torch.no_grad():
            # mean MSE in the 76-dim z-score space (phys 72 values and beh 4
            # values averaged with equal weight)
            mse_p = ((dummy_p[0] - phys_t[0]) ** 2).mean().item()
            mse_b = ((dummy_b[0] - beh_t[0]) ** 2).mean().item()
            mse = 0.5 * (mse_p + mse_b)
            lab = int(dummy_oh.argmax(1).item())
        if mse < best_mse:
            best_mse, best_lab = mse, lab
    return best_mse, int(best_lab == y_true)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("=" * 70)
    print("Proposition 5 - three-view DLG inversion experiment")
    print("=" * 70)
    model, train, test = train_global_model()
    model.eval()
    P = sum(pp.numel() for pp in model.parameters())
    m = int(round(K * P))
    print(f"model params P={P:,}, support m={m:,} (k={K}), probe windows {N_PROBE}, "
          f"iters={ITERS}, restarts={RESTARTS}", flush=True)

    phys0, beh0, y0 = train[0]
    rng = np.random.default_rng(2026)
    idx = rng.choice(len(y0), N_PROBE, replace=False)
    other_clients = [u for u in train if u != 0]

    rows = []
    for n, i in enumerate(idx):
        g0 = window_grad(model, phys0[i], beh0[i], int(y0[i]))

        # ---- V1: bare single-client gradient (dense) ----
        mse, lab = dlg_attack(model, g0, phys0[i], beh0[i], int(y0[i]))
        rows.append(["V1_single", n, mse, lab, int(y0[i])])
        print(f"  [w{n:02d}] V1 single   : MSE={mse:.4f} label_ok={lab} (true {int(y0[i])})", flush=True)

        # ---- V2: U=2 subtraction case (s - g_i = g_j; reconstructs j's window) ----
        j = other_clients[n % 3]
        physj, behj, yj = train[j]
        jj = int(rng.choice(len(yj)))
        gj = window_grad(model, physj[jj], behj[jj], int(yj[jj]))
        s2 = [a + b for a, b in zip(g0, gj)]
        view2 = [sa - ga for sa, ga in zip(s2, g0)]       # = g_j
        mse, lab = dlg_attack(model, view2, physj[jj], behj[jj], int(yj[jj]))
        rows.append(["V2_U2sub", n, mse, lab, int(yj[jj])])
        print(f"  [w{n:02d}] V2 U2-sub   : MSE={mse:.4f} label_ok={lab} (true {int(yj[jj])})", flush=True)

        # ---- V3: U=4 SecAgg mean restricted to the shared support S ----
        gs = [g0]
        for u in other_clients[:3]:
            iu = int(rng.choice(len(train[u][2])))
            gs.append(window_grad(model, train[u][0][iu], train[u][1][iu], int(train[u][2][iu])))
        gbar = [sum(t) / len(gs) for t in zip(*gs)]
        flat = torch.cat([g.flatten() for g in gbar])
        support = torch.topk(flat.abs(), m).indices
        obs_mask = torch.zeros(P, dtype=torch.bool)
        obs_mask[support] = True
        mse, lab = dlg_attack(model, gbar, phys0[i], beh0[i], int(y0[i]), obs_mask=obs_mask)
        rows.append(["V3_U4mean_S", n, mse, lab, int(y0[i])])
        print(f"  [w{n:02d}] V3 U4-mean+S: MSE={mse:.4f} label_ok={lab} (true {int(y0[i])})", flush=True)

    df = pd.DataFrame(rows, columns=["view", "window", "input_mse", "label_ok", "label_true"])
    summary = df.groupby("view").agg(mse_mean=("input_mse", "mean"),
                                     mse_std=("input_mse", "std"),
                                     label_acc=("label_ok", "mean"))
    print("\n" + "=" * 70)
    print(f"Summary ({N_PROBE} probe windows; MSE in z-score space, data variance ~ 1)")
    print("=" * 70)
    print(summary.round(4).to_string())
    df.to_csv(os.path.join(RES, "prop5_dlg.csv"), index=False)
    print(f"\n[DONE] wrote {os.path.join(RES, 'prop5_dlg.csv')}")


if __name__ == "__main__":
    main()
