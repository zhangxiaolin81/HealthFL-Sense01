# HealthFL-Sense: Code & Results

Companion repository for the manuscript:

> **HealthFL-Sense: Privacy-Preserving Federated Multimodal Fusion for College Students' Exercise and Mental-Health Sensing**
> (HealthFL research programme, paper ③ — user-level federated multimodal sensing)

This repository contains **all code, result tables, and figures** behind every
number in the paper. The main experiments run on a **synthetic cross-institution
benchmark** (fixed seed `SEED=42`; generation rules derived from the public
college physical-fitness scoring standards — no personal data).
Section 5.4 of the paper additionally reports an **external corroboration on the
public, archived, anonymized SWELL-KW wearable dataset**; the raw data are not
redistributable here (CC-BY-NC-SA-4.0), but `swellkw_preprocess.py` +
`swellkw_pipeline.py` reproduce that section end-to-end from the official
download.

## Repository structure

```
.
├── multimodal_experiments.py   # Main synthetic experiments (§5.1–5.3; Tables 1–3; Figs. 3–4)
├── mm_robustness.py            # Multi-seed robustness (42/123/2024) + FedProx baseline (Table 1 supplement; §5.4 robustness)
├── make_fig_roadmap.py         # Fig. 1 technical roadmap (SVG/PDF/PNG)
├── make_fig_architecture.py    # Fig. 2 system architecture (SVG/PDF/PNG)
├── make_fig_results.py         # Figs. 3–4 as vector PDF/SVG (+ refreshed PNG)
├── swellkw_preprocess.py       # Raw SWELL-KW files -> windowed per-subject CSVs (swellkw_raw/ -> swellkw_processed/)
├── swellkw_pipeline.py         # SWELL-KW real-data corroboration (§5.4; Table 4)
├── results_mm/                 # Synthetic-benchmark result tables (CSV)
│   ├── table_mm_main.csv       #   Main comparison (Table 1)
│   ├── table_mm_per_user.csv   #   Per-user federated-vs-local gains (Table 2)
│   ├── table_mm_comm.csv       #   Model size and communication overhead (Table 3)
│   └── table_mm_robust.csv     #   Multi-seed mean±std, FedAvg/FedProx (Table 1 supplement)
├── results_mm_real/            # SWELL-KW real-data result tables (CSV)
│   ├── table_swellkw_main.csv  #   Representative split (Table 4)
│   ├── table_swellkw_weak.csv  #   Weak-subject gains on SWELL-KW
│   └── table_swellkw_robust.csv#   3 split seeds mean±std (§5.4 supplement)
├── figs_mm/                    # Figures (SVG / PDF / PNG)
├── manuscript/                 # Manuscript source (Markdown, English) — optional, delete if you prefer
├── requirements.txt
└── LICENSE                     # Code: MIT
```

## Quick start

Environment: Python 3.10+ with CPU-only PyTorch is sufficient
(verified with torch 2.6 and 2.14, CPU-only).

```bash
pip install -r requirements.txt

# 1) Main synthetic experiments (~1 min): regenerates results_mm/*.csv and
#    the PNG versions of Figs. 3-4
python multimodal_experiments.py

# 2) Re-render Figs. 3-4 as vector PDF/SVG (recommended for print/submission):
python make_fig_results.py

# 3) Multi-seed robustness (SEED = 42/123/2024, mean±std) + FedProx baseline:
python mm_robustness.py

# 4) Figs. 1-2 (standalone diagrams):
python make_fig_roadmap.py
python make_fig_architecture.py

# 5) (Optional) SWELL-KW real-data corroboration — requires downloading the dataset first:
#    Download SWELL-KW from DANS: https://doi.org/10.17026/DANS-X55-69ZP (CC-BY-NC-SA-4.0)
#    and place the raw files under swellkw_raw/ (see the docstring of swellkw_preprocess.py).
python swellkw_preprocess.py                                  # swellkw_raw/ -> swellkw_processed/
python swellkw_pipeline.py --data swellkw_processed           # Table 4 (representative split, seed 2026)
python mm_robustness.py --data swellkw_processed              # 3 split seeds (2026/2027/2028), mean±std
```

All numbers in the paper are reproduced with fixed random seeds
(synthetic: `SEED=42`; robustness: seeds 42/123/2024; SWELL-KW: split seeds
2026/2027/2028).
Runtime on a CPU-only laptop: synthetic main experiments ≈ 1 min; robustness
sweep ≈ 5 min; SWELL-KW pipeline ≈ 2 min per seed.

## Reproducibility

Verified end-to-end (every script exits 0) with **torch 2.6 and 2.14
(CPU-only)** and **matplotlib 3.9 / 3.11**:

- Under the verified torch versions, the shipped CSVs in `results_mm/` and
  `results_mm_real/` reproduce **byte-identically** (synthetic main
  experiments, vector figures, both diagram figures, and the multi-seed /
  FedProx robustness sweep).
- On some torch versions the *Local-only* (isolated training) columns of the
  main / per-user / robustness tables may differ in the last decimal
  (e.g. macro-F1 0.645 vs 0.646) because of CPU float reduction order;
  centralized / federated / FedProx values are stable.
- The four figures in `figs_mm/` are exactly what the included figure scripts
  produce under the verified environment. Regenerated SVG/PDF files are
  content-identical; matplotlib embeds a creation date and random internal
  clip-path IDs, so those bytes may differ.
- `swellkw_pipeline.py --mock` runs the full federated analysis on
  SWELL-KW-shaped synthetic data without downloading the dataset (self-check).

## Data statements

- **Synthetic benchmark (main experiments).** 24 simulated users × 90 windows
  per user; each window holds a 3-channel physiological time series
  (HR/HRV/EDA, 24 steps) and 4 behavioral features
  (sleep/social/screen/typing), with 3-class stress/emotion labels. Per-user
  physiological baselines, behavioral tendencies and stress sensitivity
  induce the user-level non-IID structure the theory relies on; labels follow
  a documented latent-stress threshold rule. No personal or identifiable data
  are involved.
- **SWELL-KW (external corroboration, §5.4 only).** Public, archived,
  anonymized wearable records (Koldijk et al., ICMI 2014;
  DOI 10.17026/DANS-X55-69ZP, CC-BY-NC-SA-4.0). The original study obtained
  ethics approval; this repository only re-uses the public de-identified data
  and releases the windowing/preprocessing adapters. Raw data are **not**
  redistributed here — download from DANS.

## Citation

If you use this code, please cite the manuscript and the Zenodo DOI of this
repository (10.5281/zenodo.XXXXXXX, assigned upon archival).

## License

- Code in this repository: [MIT](LICENSE).
- SWELL-KW dataset: CC-BY-NC-SA-4.0 (held by the original authors, not
  redistributed here).
