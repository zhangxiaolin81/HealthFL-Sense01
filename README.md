# HealthFL-Sense: Code & Results

Companion repository for the manuscript:

> **Privacy–Communication Co-Design for Federated Multimodal Learning:
> A Support-Agreement Rule Reconciling Top-k Sparsification with Secure Aggregation**
> (HealthFL research programme, paper ③ — user-level federated multimodal sensing)

This repository contains **all code, result tables, and figures** behind every
number in the paper. The main experiments run on a **synthetic cross-institution
benchmark** (fixed seed `SEED=42`; generation rules derived from the public
college physical-fitness scoring standards — no personal data).

The paper makes three kinds of claims, kept explicitly separate:

- **Measured** (Sections 5.1–5.4): multimodal fusion vs. unimodal/isolated
  baselines; per-user gains; communication accounting; and the accuracy of the
  *deployable* support-agreement protocols (Protocol A: shared static support;
  Protocol B: lagged global Top-k) against the per-client upper bound.
- **Measured, security** (Section 5.5): a three-view DLG gradient-inversion
  experiment that pairs the analytical guarantee of Proposition 5 — over a
  shared support, the SecAgg aggregate leaves each client's update undetermined
  in a `(U−1)m`-dimensional subspace; a participating client defeats SecAgg
  unless `U ≥ 3`.
- **Proved** (Proposition 5, exact per-round) vs. **argued** (Proposition 3,
  exposure rationale): the paper's Table 7 ("What supports each claim") labels
  each claim accordingly.

Section 5.6 additionally reports an **ordering check on the public, archived,
anonymized SWELL-KW wearable dataset**; the raw data are not redistributable
here (CC-BY-NC-SA-4.0), but `swellkw_preprocess.py` + `swellkw_pipeline.py`
reproduce that section end-to-end from the official download.

## Repository structure

```
.
├── multimodal_experiments.py   # Main synthetic experiments (§5.1–5.3; Tables 1–3; Figs. 3–4)
├── mm_robustness.py            # Multi-seed robustness (42/123/2024) + FedProx baseline (Table 1 supplement)
├── mm_protocolB.py             # Support-rule accuracy: dense / per-client / Protocol A / Protocol B, 3 seeds (§5.4; Table 4)
├── mm_protocolB_extra.py       # Matched-seed Local baseline + k=0.3 trade-off point (§5.4; Table 4)
├── mm_prop5_dlg.py             # Three-view DLG attack pairing Proposition 5 (§5.5; Table 5)
├── mm_supplement.py            # Online-supplement tables (table_user_gains.csv, table_mm_perclass.csv)
├── make_fig_roadmap.py         # Fig. 1 technical roadmap (SVG/PDF/PNG)
├── make_fig_architecture.py    # Fig. 2 system architecture (SVG/PDF/PNG)
├── make_fig_results.py         # Figs. 3–4 as vector PDF/SVG (+ refreshed PNG)
├── swellkw_preprocess.py       # Raw SWELL-KW files -> windowed per-subject CSVs (swellkw_raw/ -> swellkw_processed/)
├── swellkw_pipeline.py         # SWELL-KW real-data ordering check (§5.6; Table 6)
├── results_mm/                 # Synthetic-benchmark result tables (CSV)
│   ├── table_mm_main.csv       #   Main comparison (Table 1)
│   ├── table_mm_per_user.csv   #   Per-user federated-vs-local gains (Table 2)
│   ├── table_mm_comm.csv       #   Model size and communication overhead (Table 3)
│   ├── table_mm_robust.csv     #   Multi-seed mean±std, FedAvg/FedProx (Table 1 supplement)
│   ├── table_mm_perclass.csv   #   Per-class F1 breakdown (Table 1 supplement)
│   ├── table_user_gains.csv    #   Full per-user gains + summary stats (Table 2 supplement)
│   ├── protocolB.csv           #   Support-rule accuracy, 4 modes × 3 seeds (§5.4)
│   ├── protocolB_extra.csv     #   Matched-seed Local baseline + k=0.3 sweep (§5.4)
│   └── prop5_dlg.csv           #   DLG per-window results, 3 views × 12 windows (§5.5)
├── results_mm_real/            # SWELL-KW real-data result tables (CSV)
│   ├── table_swellkw_main.csv  #   Representative split (Table 6)
│   ├── table_swellkw_weak.csv  #   Weak-subject gains on SWELL-KW
│   └── table_swellkw_robust.csv#   3 split seeds mean±std (§5.6 supplement)
├── figs_mm/                    # Figures (SVG / PDF / PNG)
├── manuscript/                 # Manuscript source (LaTeX + compiled PDF)
├── requirements.txt
└── LICENSE                     # Code: MIT
```

## Quick start

Environment: Python 3.10+ with CPU-only PyTorch is sufficient
(verified with torch 2.6, CPU-only).

```bash
pip install -r requirements.txt

# 1) Main synthetic experiments (~1 min): regenerates results_mm/table_mm_*.csv
#    and the PNG versions of Figs. 3-4
python multimodal_experiments.py

# 2) Multi-seed robustness (SEED = 42/123/2024, mean±std) + FedProx baseline:
python mm_robustness.py

# 3) Support-agreement protocol accuracy — the deployable-protocol experiment
#    of Section 5.4 (4 support rules x 3 seeds; ~15 min on CPU):
python mm_protocolB.py
python mm_protocolB_extra.py        # matched-seed Local baseline + k=0.3 point

# 4) Three-view DLG gradient-inversion experiment of Section 5.5 (~12 min):
python mm_prop5_dlg.py

# 5) Online-supplement tables (table_user_gains.csv, table_mm_perclass.csv):
python mm_supplement.py

# 6) Figures:
python make_fig_results.py          # Figs. 3-4 (vector)
python make_fig_roadmap.py          # Fig. 1
python make_fig_architecture.py     # Fig. 2

# 7) (Optional) SWELL-KW ordering check — requires downloading the dataset first:
#    Download SWELL-KW from DANS: https://doi.org/10.17026/DANS-X55-69ZP (CC-BY-NC-SA-4.0)
#    and place the raw files under swellkw_raw/ (see the docstring of swellkw_preprocess.py).
python swellkw_preprocess.py                                  # swellkw_raw/ -> swellkw_processed/
python swellkw_pipeline.py --data swellkw_processed           # Table 6 (representative split, seed 2026)
python mm_robustness.py --data swellkw_processed              # 3 split seeds (2026/2027/2028), mean±std
```

All numbers in the paper are reproduced with fixed random seeds
(synthetic: `SEED=42`; robustness: seeds 42/123/2024; SWELL-KW: split seeds
2026/2027/2028).
Runtime on a CPU-only laptop: synthetic main experiments ≈ 1 min; robustness
sweep ≈ 5 min; protocol-accuracy sweep ≈ 15 min; DLG experiment ≈ 12 min;
SWELL-KW pipeline ≈ 2 min per seed.

## Reproducibility

- Under the verified torch version (2.6, CPU-only), the shipped CSVs in
  `results_mm/` and `results_mm_real/` reproduce up to CPU float-reduction
  order; on some torch versions the *Local-only* (isolated training) columns
  may differ in the last decimal (e.g. macro-F1 0.645 vs 0.646), while
  centralized / federated / FedProx values are stable.
- `mm_protocolB.py` and `mm_prop5_dlg.py` were the actual runs behind the
  paper's Table 4 and Table 5; the shipped `protocolB.csv`,
  `protocolB_extra.csv`, and `prop5_dlg.csv` are the exact per-window /
  per-seed outputs of those runs.
- The figures in `figs_mm/` are what the included figure scripts produce;
  regenerated SVG/PDF files are content-identical (matplotlib embeds a
  creation date, so bytes may differ).
- `swellkw_pipeline.py --mock` runs the full federated analysis on
  SWELL-KW-shaped synthetic data without downloading the dataset (self-check).

## Headline results (for orientation)

| Quantity | Value |
|---|---|
| Federated multimodal (3 seeds) | 0.712 ± 0.029 vs 0.808 centralized, 0.663 isolated |
| Weakest-user gain | up to +35.2 pp (max; mean +7.4 pp, 6/24 users negative) |
| Protocol B (deployable, k=0.1) | 0.657 ± 0.013 (−5.1 pp vs dense; +1.3 pp vs matched-seed Local) |
| Protocol B (k=0.3) | 0.669 ± 0.044 (+2.5 pp vs matched-seed Local) |
| Protocol A (static support) | 0.583 ± 0.066 — fails; motivates the lagged-global rule |
| DLG on bare single-client update | median MSE 0.26, label recovery 11/12 |
| DLG on U=2 subtraction | median MSE 0.33, label recovery 12/12 (zero protection, as proved) |
| DLG on U=4 SecAgg mean, shared support | median MSE 1.09 (×4.3), label recovery 9/12 |
| Communication (shared support) | 0.20× dense payload incl. SecAgg masks; 87.5–90% saving |

## Data statements

- **Synthetic benchmark (main experiments).** 24 simulated users × 90 windows
  per user; each window holds a 3-channel physiological time series
  (HR/HRV/EDA, 24 steps) and 4 behavioral features
  (sleep/social/screen/typing), with 3-class stress/emotion labels. Per-user
  physiological baselines, behavioral tendencies and stress sensitivity
  induce the user-level non-IID structure the theory relies on; labels follow
  a documented latent-stress threshold rule. No personal or identifiable data
  are involved.
- **SWELL-KW (external ordering check, §5.6 only).** Public, archived,
  anonymized wearable records (Koldijk et al., ICMI 2014;
  DOI 10.17026/DANS-X55-69ZP, CC-BY-NC-SA-4.0). The original study obtained
  ethics approval; this repository only re-uses the public de-identified data
  and releases the windowing/preprocessing adapters. Raw data are **not**
  redistributed here — download from DANS.

## Citation

If you use this code, please cite the manuscript and the Zenodo DOI of this
repository (10.5281/zenodo.23640739 — a new version will be minted for this
release).

## License

- Code in this repository: [MIT](LICENSE).
- SWELL-KW dataset: CC-BY-NC-SA-4.0 (held by the original authors, not
  redistributed here).
