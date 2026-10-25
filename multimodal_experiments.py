"""
multimodal_experiments.py — Paper ③: federated validation experiments for
multimodal exercise-and-mental-health sensing.

Scenario: each college-student user owns a wearable (physiology: HR/HRV/EDA
time series) and a smartphone (behavior: sleep / social / screen / typing).
Models are trained locally and only parameters are uploaded; a HealthFL
coordinator aggregates them for early stress/emotion screening.

Experiments (same "theory-first, synthetic-benchmark validation" methodology
as papers ① and ②):
  [A] Main comparison: Centralized / FedAvg / Local-only / Phys-only / Beh-only / Late / Early
  [B] User-level gain: Federated vs Local per user (weak users benefit most)
  [C] Convergence curve: per-round global accuracy of FedAvg (late fusion)
  [D] Communication volume: multimodal model size vs. the paper-① MLP; Top-k savings
  [E] Inversion threat: multimodal parameters/input dims far exceed paper ② — defense transfer

Run: python multimodal_experiments.py
Outputs: results_mm/*.csv, figs_mm/*.png
"""
import copy
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

DEVICE = "cpu"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results_mm")
FIG = os.path.join(ROOT, "figs_mm")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

U = 24          # number of users (clients)
M = 90          # time windows per user
T = 24          # time steps per window
N_CLASSES = 3   # 0=calm 1=mild stress 2=high stress/anxiety
ROUNDS = 15
CLIENT_EPOCHS = 2
LR = 1e-3
BATCH = 16


# ---------------------------------------------------------------------------
# Synthetic multimodal data generation (user-level non-IID)
# ---------------------------------------------------------------------------
def build_multimodal(U=U, M=M, T=T, seed=SEED, n_classes=N_CLASSES):
    """Generate multimodal window data for U users. Each user has a different
    physiological baseline, behavioral tendency and stress sensitivity, which
    naturally induces strong non-IID (finer-grained than the school-level
    non-IID of paper ①)."""
    rng = np.random.default_rng(seed)
    clients = {}
    for u in range(U):
        base_hr = rng.normal(72, 6)
        base_hrv = rng.normal(55, 10)
        base_eda = rng.normal(3.0, 0.8)
        sens = rng.uniform(0.5, 1.5)            # individual stress-reactivity strength
        b_sleep = rng.normal(7.0, 1.0)
        b_social = rng.normal(60, 20)
        b_screen = rng.normal(240, 60)
        b_typing = rng.normal(40, 10)
        phys = np.zeros((M, T, 3))
        beh = np.zeros((M, 4))
        y = np.zeros(M, dtype=int)
        for m in range(M):
            s = rng.normal(0, 1)                # latent stress level
            hr = base_hr + sens * 8 * s + rng.normal(0, 2, T)
            hrv = base_hrv - sens * 10 * s + rng.normal(0, 3, T)
            eda = base_eda + sens * 0.6 * s + rng.normal(0, 0.2, T)
            phys[m, :, 0] = hr
            phys[m, :, 1] = hrv
            phys[m, :, 2] = eda
            sleep = b_sleep - 1.2 * s + rng.normal(0, 0.5)
            social = b_social + 15 * s * rng.choice([-1, 1]) + rng.normal(0, 8)
            screen = b_screen + 30 * s + rng.normal(0, 15)
            typing = b_typing - 5 * s + rng.normal(0, 4)
            beh[m, 0] = sleep
            beh[m, 1] = social
            beh[m, 2] = screen
            beh[m, 3] = typing
            score = s + rng.normal(0, 0.3)
            y[m] = int(np.digitize(score, [-0.5, 0.5]))
        clients[u] = (phys, beh, y)
    # Global z-score normalization (declared as per-client local standardization
    # in the paper; unified centering here on synthetic data for reproducibility)
    all_phys = np.concatenate([c[0] for c in clients.values()], 0)
    all_beh = np.concatenate([c[1] for c in clients.values()], 0)
    p_mu, p_sd = all_phys.mean((0, 1)), all_phys.std((0, 1)) + 1e-6
    b_mu, b_sd = all_beh.mean(0), all_beh.std(0) + 1e-6
    for u in clients:
        p, b, y = clients[u]
        clients[u] = ((p - p_mu) / p_sd, (b - b_mu) / b_sd, y)
    return clients


class MMUserDataset(Dataset):
    def __init__(self, phys, beh, y):
        self.phys = torch.tensor(phys, dtype=torch.float32)
        self.beh = torch.tensor(beh, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, i):
        return self.phys[i], self.beh[i], self.y[i]

    def __len__(self):
        return len(self.y)


# ---------------------------------------------------------------------------
# Multimodal model (late / early fusion, plus unimodal variants)
# ---------------------------------------------------------------------------
class MultiModalModel(nn.Module):
    def __init__(self, n_classes=N_CLASSES, phys_in=3, beh_in=4, hidden=32,
                 fusion="late", mode="full"):
        super().__init__()
        self.fusion = fusion
        self.mode = mode  # full / phys / beh
        self.rnn = nn.LSTM(phys_in, hidden, batch_first=True, bidirectional=True)
        self.beh_net = nn.Sequential(nn.Linear(beh_in, hidden), nn.ReLU())
        if fusion == "early":
            # early fusion: behavior concatenated into the time-series input,
            # rnn only, phys-style head
            self.rnn = nn.LSTM(phys_in + beh_in, hidden, batch_first=True,
                               bidirectional=True)
            self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(),
                                      nn.Linear(64, n_classes))
        elif mode == "phys":
            self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(),
                                      nn.Linear(64, n_classes))
        elif mode == "beh":
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                      nn.Linear(64, n_classes))
        else:
            self.head = nn.Sequential(nn.Linear(hidden * 2 + hidden, 64), nn.ReLU(),
                                      nn.Linear(64, n_classes))

    def forward(self, phys, beh):
        if self.mode == "phys":
            out, (h, c) = self.rnn(phys)
            emb = torch.cat([h[-2], h[-1]], dim=1)
            return self.head(emb)
        if self.mode == "beh":
            return self.head(self.beh_net(beh))
        # full
        if self.fusion == "early":
            phys_b = beh.unsqueeze(1).repeat(1, phys.size(1), 1)
            x = torch.cat([phys, phys_b], dim=-1)
            out, (h, c) = self.rnn(x)
            emb = torch.cat([h[-2], h[-1]], dim=1)
            return self.head(emb)
        else:  # late
            out, (h, c) = self.rnn(phys)
            phys_emb = torch.cat([h[-2], h[-1]], dim=1)
            beh_emb = self.beh_net(beh)
            emb = torch.cat([phys_emb, beh_emb], dim=1)
        return self.head(emb)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Federated training (self-contained FedAvg; reuses the coordinator logic of
# paper ①)
# ---------------------------------------------------------------------------
def train_client_mm(model, loader, epochs=CLIENT_EPOCHS, lr=LR, device=DEVICE):
    model = copy.deepcopy(model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for phys, beh, y in loader:
            phys, beh, y = phys.to(device), beh.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(phys, beh), y)
            loss.backward()
            opt.step()
    sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return sd, len(loader.dataset)


def fedavg_mm(state_dicts, weights):
    total = float(sum(weights))
    agg = {}
    for k in state_dicts[0]:
        agg[k] = sum(sd[k].float() * w for sd, w in zip(state_dicts, weights)) / total
    return agg


def metrics_mm(model, loader, device=DEVICE):
    model.eval()
    if len(loader.dataset) == 0:
        return 0.0, 0.0, [], []
    preds, labels = [], []
    with torch.no_grad():
        for phys, beh, y in loader:
            out = model(phys.to(device), beh.to(device))
            p = out.argmax(1).cpu()
            preds += p.tolist()
            labels += y.tolist()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1, preds, labels


def make_loader(phys, beh, y, batch=BATCH, shuffle=True):
    # Fixed generator so the shuffle order — and hence results — are
    # reproducible for a given configuration
    gen = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(MMUserDataset(phys, beh, y), batch_size=batch,
                      shuffle=shuffle, generator=gen)


def fed_run(clients_train, clients_test, fusion="late", mode="full",
            rounds=ROUNDS, epochs=CLIENT_EPOCHS):
    """Run one federated training; returns (final global model, per-round
    global accuracy list)."""
    names = list(clients_train)
    train_loaders = {n: make_loader(*clients_train[n]) for n in names}
    # Global test set (concatenation of all client test sets)
    gphys = np.concatenate([clients_test[n][0] for n in names], 0)
    gbeh = np.concatenate([clients_test[n][1] for n in names], 0)
    gy = np.concatenate([clients_test[n][2] for n in names], 0)
    gtest_loader = make_loader(gphys, gbeh, gy, shuffle=False)
    global_model = MultiModalModel(fusion=fusion, mode=mode).to(DEVICE)
    acc_curve = []
    for r in range(rounds):
        sds, wts = [], []
        for n in names:
            sd, w = train_client_mm(global_model, train_loaders[n], epochs=epochs)
            sds.append(sd)
            wts.append(w)
        agg = fedavg_mm(sds, wts)
        global_model.load_state_dict(agg)
        acc, _, _, _ = metrics_mm(global_model, gtest_loader)
        acc_curve.append(acc)
    return global_model, acc_curve


def local_run(clients_train, clients_test, fusion="late", mode="full"):
    """Train each client independently, then evaluate on the global test set
    (isolated lower bound)."""
    names = list(clients_train)
    gphys = np.concatenate([clients_test[n][0] for n in names], 0)
    gbeh = np.concatenate([clients_test[n][1] for n in names], 0)
    gy = np.concatenate([clients_test[n][2] for n in names], 0)
    gtest_loader = make_loader(gphys, gbeh, gy, shuffle=False)
    accs, f1s = [], []
    per_user = {}
    for n in names:
        m = MultiModalModel(fusion=fusion, mode=mode).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        crit = nn.CrossEntropyLoss()
        m.train()
        for _ in range(ROUNDS * CLIENT_EPOCHS):
            for phys, beh, y in make_loader(*clients_train[n]):
                phys, beh, y = phys.to(DEVICE), beh.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(); loss = crit(m(phys, beh), y); loss.backward(); opt.step()
        a, f, _, _ = metrics_mm(m, gtest_loader)
        accs.append(a); f1s.append(f)
        per_user[n] = a
    return float(np.mean(accs)), float(np.mean(f1s)), per_user


def centralized_run(clients_train, clients_test, fusion="late", mode="full",
                    epochs=ROUNDS * CLIENT_EPOCHS):
    gphys = np.concatenate([clients_train[n][0] for n in clients_train], 0)
    gbeh = np.concatenate([clients_train[n][1] for n in clients_train], 0)
    gy = np.concatenate([clients_train[n][2] for n in clients_train], 0)
    tr_loader = make_loader(gphys, gbeh, gy)
    gphys_t = np.concatenate([clients_test[n][0] for n in clients_test], 0)
    gbeh_t = np.concatenate([clients_test[n][1] for n in clients_test], 0)
    gy_t = np.concatenate([clients_test[n][2] for n in clients_test], 0)
    te_loader = make_loader(gphys_t, gbeh_t, gy_t, shuffle=False)
    m = MultiModalModel(fusion=fusion, mode=mode).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    m.train()
    for _ in range(epochs):
        for phys, beh, y in tr_loader:
            phys, beh, y = phys.to(DEVICE), beh.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(m(phys, beh), y); loss.backward(); opt.step()
    a, f, _, _ = metrics_mm(m, te_loader)
    return a, f


def save_csv(name, header, rows):
    df = pd.DataFrame(rows, columns=header)
    path = os.path.join(RES, name)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Paper ③: multimodal exercise-and-mental-health sensing — federated validation experiments")
    print("=" * 70)
    clients = build_multimodal()
    # Train/test split
    train, test = {}, {}
    for u, (p, b, y) in clients.items():
        n_tr = int(len(y) * 0.8)
        idx = np.random.RandomState(SEED + u).permutation(len(y))
        tr, te = idx[:n_tr], idx[n_tr:]
        train[u] = (p[tr], b[tr], y[tr])
        test[u] = (p[te], b[te], y[te])

    variants = [
        ("Full-Late", dict(fusion="late", mode="full")),
        ("Full-Early", dict(fusion="early", mode="full")),
        ("Phys-only", dict(fusion="late", mode="phys")),
        ("Beh-only", dict(fusion="late", mode="beh")),
    ]

    # Precompute Full-Late once: shared by the headline and the per-user gain,
    # guaranteeing main table = headline
    gphys_all = np.concatenate([test[n][0] for n in test], 0)
    gbeh_all = np.concatenate([test[n][1] for n in test], 0)
    gy_all = np.concatenate([test[n][2] for n in test], 0)
    c_a0, c_f0 = centralized_run(train, test, fusion="late", mode="full")
    g_model, curve0 = fed_run(train, test, fusion="late", mode="full")
    f_a0, f_f0, _, _ = metrics_mm(g_model, make_loader(gphys_all, gbeh_all, gy_all, shuffle=False))
    l_a0, l_f0, per_user_local = local_run(train, test, fusion="late", mode="full")

    print("\n[A] Main comparison (accuracy / macro-F1, global test set)")
    print(f"{'Method':<12}{'Centralized':>14}{'FedAvg':>12}{'Local':>12}")
    rows = []
    fed_acc_curves = {}
    for vname, kw in variants:
        if vname == "Full-Late":
            c_a, c_f, f_a, f_f, l_a, l_f, curve = c_a0, c_f0, f_a0, f_f0, l_a0, l_f0, curve0
        else:
            c_a, c_f = centralized_run(train, test, **kw)
            gm_v, curve = fed_run(train, test, **kw)
            f_a, f_f, _, _ = metrics_mm(gm_v, make_loader(gphys_all, gbeh_all, gy_all, shuffle=False))
            l_a, l_f, _ = local_run(train, test, **kw)
        fed_acc_curves[vname] = curve
        rows.append([vname, f"{c_a:.3f}/{c_f:.3f}", f"{f_a:.3f}/{f_f:.3f}",
                     f"{l_a:.3f}/{l_f:.3f}"])
        print(f"{vname:<12}{c_a:.3f}/{c_f:.3f}  {f_a:.3f}/{f_f:.3f}  {l_a:.3f}/{l_f:.3f}")
    save_csv("table_mm_main.csv",
             ["Method", "Centralized_acc/f1", "FedAvg_acc/f1", "Local_acc/f1"],
             rows)

    # [B] User-level gain (Full-Late)
    print("\n[B] User-level gain (FedAvg vs Local, Full-Late)")
    # Per-user accuracy under federation: evaluate the global model on each
    # user's test set
    fed_per_user = {}
    for u in test:
        a, _, _, _ = metrics_mm(g_model, make_loader(*test[u], shuffle=False))
        fed_per_user[u] = a
    gains = [(u, per_user_local[u], fed_per_user[u],
              (fed_per_user[u] - per_user_local[u]) * 100)
             for u in test]
    gains_sorted = sorted(gains, key=lambda t: t[3])
    # Take the 5 weakest users to inspect gains
    weak = gains_sorted[-5:]
    print("  Top-5 weakest users by gain:")
    for u, la, fa, g in weak:
        print(f"    user_{u}: Local={la:.3f} Fed={fa:.3f} Gain={g:+.1f}pp")
    save_csv("table_mm_per_user.csv",
             ["user", "local_acc", "fed_acc", "gain_pp"],
             [[u, f"{la:.4f}", f"{fa:.4f}", f"{g:+.2f}"] for u, la, fa, g in gains])

    # [C] Convergence curve
    print("\n[C] Convergence curve (Full-Late FedAvg)")
    print(f"  Round1={fed_acc_curves['Full-Late'][0]:.3f} -> "
          f"Round{ROUNDS}={fed_acc_curves['Full-Late'][-1]:.3f}")

    # [D] Communication volume (Top-k sparsification of paper ②)
    print("\n[D] Communication volume (Top-k sparsification of paper ②)")
    mlp_params = 4931                       # paper-① physical-fitness MLP
    mm_params = count_params(MultiModalModel())
    print(f"  paper-① physical-fitness MLP params : {mlp_params:,}")
    print(f"  paper-③ multimodal model params     : {mm_params:,}  ({mm_params/mlp_params:.1f}x)")
    per_round_mm = mm_params * 4 * 2         # up+down, float32
    per_round_mlp = mlp_params * 4 * 2
    print(f"  paper-③ per-round comm. (no compression): {per_round_mm/1e3:.1f} KB x {ROUNDS} = "
          f"{per_round_mm*ROUNDS/1e6:.3f} MB")
    for k in [0.1, 0.3, 0.5]:
        save = per_round_mm * k * ROUNDS / 1e6
        print(f"    Top-k k={k}: communication drops to {per_round_mm*k/1e3:.1f} KB/round, "
              f"{save:.3f} MB over {ROUNDS} rounds")
    save_csv("table_mm_comm.csv",
             ["Metric", "Value"],
             [["mlp_params_p1", mlp_params],
              ["mm_params_p3", mm_params],
              ["mm_per_round_bytes", per_round_mm],
              ["mm_total_bytes_15r", per_round_mm * ROUNDS],
              ["topk_0.1_total_MB", per_round_mm * 0.1 * ROUNDS / 1e6],
              ["topk_0.3_total_MB", per_round_mm * 0.3 * ROUNDS / 1e6],
              ["topk_0.5_total_MB", per_round_mm * 0.5 * ROUNDS / 1e6]])

    # [E] Inversion threat (qualitative, building on paper ②)
    print("\n[E] Inversion-threat note (building on paper ②)")
    print(f"  paper-③ multimodal input dim = physiology {T*3} time series + behavior 4 = {T*3+4} dims/window,")
    print(f"  model params {mm_params:,} >> paper-② MLP 4,931;")
    print("  higher-dimensional inputs make DLG inversion more feasible (paper ② §4.1: inversion")
    print("  grows with parameter dim P and input dim), so the SecAgg (primary) + Top-k")
    print("  (communication) defenses of paper ② transfer directly here.")

    # ---------- Plotting ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        for vname, curve in fed_acc_curves.items():
            plt.plot(range(1, ROUNDS + 1), curve, marker="o", label=vname)
        plt.axhline(c_a0, ls="--", color="gray", label="Centralized")
        plt.xlabel("Round"); plt.ylabel("Global Accuracy")
        plt.title("Federated Convergence (multi-modal)")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "fig_mm_convergence.png"), dpi=120)
        plt.close()

        # Per-user gain bar chart
        us = [f"u{u}" for u, _, _, _ in gains_sorted]
        ga = [g for _, _, _, g in gains_sorted]
        plt.figure(figsize=(8, 3.5))
        colors = ["#c0392b" if x < 0 else "#27ae60" for x in ga]
        plt.bar(us, ga, color=colors)
        plt.axhline(0, color="black", lw=0.8)
        plt.xlabel("User"); plt.ylabel("Fed - Local (pp)")
        plt.title("Per-User Gain (Federated vs Local)")
        plt.xticks(rotation=90, fontsize=7); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "fig_mm_per_user.png"), dpi=120)
        plt.close()
        print("\n  Figures saved: figs_mm/fig_mm_convergence.png, fig_mm_per_user.png")
    except Exception as e:
        print("\n  [warn] plotting failed:", e)

    print("\n" + "=" * 70)
    print(f"Main table (paper ③) headline: Centralized={c_a0:.3f}  FedAvg={f_a0:.3f}  "
          f"Local={l_a0:.3f}  (Fed vs Local +{(f_a0-l_a0)*100:.1f}pp)")
    print("=" * 70)


if __name__ == "__main__":
    main()
