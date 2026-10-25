"""swellkw_preprocess.py — preprocess raw SWELL-KW feature files into the
windowed contract format consumed by swellkw_pipeline.

Inputs (swellkw_raw/):
  - "D - Physiology features (HR_HRV_SCL - final).csv"  physiology: PP, C,
    Condition, timestamp, HR, RMSSD, SCL
  - "Behavioral-features - per minute.tab"             behavior: PP, Blok,
    Condition, timestamp, ... Sn* computer logs

Outputs (swellkw_processed/): one CSV per subject, columns
  subj, win, hr_0..hr_23, hrv_0..hrv_23, eda_0..eda_23, beh_0..beh_3, label
  where phys=[HR, HRV(RMSSD), EDA(SCL)] and beh=[SnKeyStrokes, SnMouseAct,
  SnAppChange, SnTabfocusChange] averaged within each window;
  label: 0=N (neutral/no stressor), 1=I (email interruption), 2=T (time
  pressure); R (relax) is discarded.

Windowing rule: within each subject, split into contiguous same-Condition
segments (timestamp-continuous and condition-invariant), keep pure-condition
segments of length >= T (24) minutes, and take windows at a half-overlapping
stride of 12 minutes; minutes with missing values (999/NaN) are linearly
interpolated within their segment.
"""
import os
import csv
import glob
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "swellkw_raw")            # official raw downloads
OUT = os.path.join(ROOT, "swellkw_processed")      # windowed contract format
T = 24
STRIDE = 12
NANSET = {"NaN", "999", "", " "}

# behavioral file columns (.tab, 172 columns)
BEH_COLS = {
    "SnKeyStrokes": "beh_0",
    "SnMouseAct": "beh_1",
    "SnAppChange": "beh_2",
    "SnTabfocusChange": "beh_3",
}


def isnan(v):
    return v is None or v.strip() in NANSET


def fnum(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def load_physio():
    """Return {(PP, timestamp): (HR, RMSSD, SCL)}; 999/empty -> nan."""
    path = None
    for cand in glob.glob(os.path.join(RAW, "*Physiology*final*.csv")):
        path = cand
        break
    assert path, "physiology feature CSV not found"
    d = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        h = next(r)
        idx = {n: i for i, n in enumerate(h)}
        for row in r:
            pp = row[idx["PP"]].strip()
            ts = row[idx["timestamp"]].strip()
            hr = fnum(row[idx["HR"]]) if not isnan(row[idx["HR"]]) else np.nan
            rmssd = fnum(row[idx["RMSSD"]]) if not isnan(row[idx["RMSSD"]]) else np.nan
            scl = fnum(row[idx["SCL"]]) if not isnan(row[idx["SCL"]]) else np.nan
            d[(pp, ts)] = (hr, rmssd, scl)
    return d


def load_behavioral(physio):
    """Return ordered per-minute records grouped by subject
    [(ts, hr, hrv, eda, beh4, cond), ...]."""
    path = None
    for cand in glob.glob(os.path.join(RAW, "*Behavioral*per minute*.tab")):
        path = cand
        break
    assert path, "behavioral feature TAB not found"
    by_subj = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        idx = {n: i for i, n in enumerate(h)}
        for row in r:
            pp = row[idx["PP"]].strip()
            ts = row[idx["timestamp"]].strip()
            cond = row[idx["Condition"]].strip()
            # physiology taken from the physiology file first; fall back to the
            # HR/RMSSD/SCL columns of the behavioral file
            hr = hrv = eda = np.nan
            if (pp, ts) in physio:
                hr, hrv, eda = physio[(pp, ts)]
            if np.isnan(hr) and not isnan(row[idx["HR"]]):
                hr = fnum(row[idx["HR"]])
            if np.isnan(hrv) and not isnan(row[idx["RMSSD"]]):
                hrv = fnum(row[idx["RMSSD"]])
            if np.isnan(eda) and not isnan(row[idx["SCL"]]):
                eda = fnum(row[idx["SCL"]])
            beh = []
            for c in BEH_COLS:
                v = fnum(row[idx[c]]) if not isnan(row[idx[c]]) else np.nan
                beh.append(v)  # always keep the 4 dims; missing values stored
                               # as NaN, interpolated by windowize
            # keep all task minutes (do not drop missing); interpolation fills
            # gaps within windowize
            by_subj.setdefault(pp, []).append((ts, hr, hrv, eda, beh, cond))
    return by_subj


def impute(arr):
    """Linearly interpolate NaN within a segment (extrapolate outside the
    endpoints to the nearest valid value); return None if the whole segment
    is NaN."""
    a = np.array(arr, dtype=np.float64)
    valid = ~np.isnan(a)
    if not valid.any():
        return None
    idx = np.arange(len(a))
    a[~valid] = np.interp(idx[~valid], idx[valid], a[valid])
    return a


def windowize(subj_rows):
    """Split the ordered per-minute records of one subject into windows by
    contiguous same-condition segments; linearly interpolate missing channels
    within each segment. Returns a list of windows
    [(hr24, hrv24, eda24, beh4, label), ...]."""
    subj_rows.sort(key=lambda x: x[0])
    label_map = {"N": 0, "I": 1, "T": 2}
    windows = []
    i = 0
    n = len(subj_rows)
    while i < n:
        cond = subj_rows[i][5]
        if cond not in label_map:  # R (relax) discarded
            i += 1
            continue
        # collect a contiguous same-condition segment (timestamp-continuous
        # and condition-invariant)
        j = i
        while j + 1 < n and subj_rows[j + 1][5] == cond:
            j += 1
        seg = subj_rows[i:j + 1]  # contiguous segment (L minutes)
        L = len(seg)
        if L < T:
            i = j + 1
            continue
        # interpolate missing channels within the segment
        hr = impute([x[1] for x in seg])
        hrv = impute([x[2] for x in seg])
        eda = impute([x[3] for x in seg])
        beh = [impute([x[4][b] for x in seg]) for b in range(4)]
        if hr is None or hrv is None or eda is None or any(b is None for b in beh):
            i = j + 1
            continue
        beh = np.array(beh)  # (4, L)
        # half-overlapping stride=12 windows (common practice for
        # physiological-signal ML; enlarges small samples and gives >= 2
        # windows per client)
        k = 0
        while k + T <= L:
            w_hr = hr[k:k + T]
            w_hrv = hrv[k:k + T]
            w_eda = eda[k:k + T]
            w_beh = beh[:, k:k + T].mean(axis=1)
            windows.append((w_hr.tolist(), w_hrv.tolist(), w_eda.tolist(),
                            w_beh.tolist(), label_map[cond]))
            k += STRIDE
        i = j + 1
    return windows


def main():
    os.makedirs(OUT, exist_ok=True)
    physio = load_physio()
    by_subj = load_behavioral(physio)
    print(f"physiology file records: {len(physio)}")
    print(f"subjects: {len(by_subj)}")

    total_windows = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    for pp in sorted(by_subj):
        rows = by_subj[pp]
        wins = windowize(rows)
        if not wins:
            continue
        out_fp = os.path.join(OUT, f"subject_{pp}.csv")
        header = ["subj", "win"] + [f"hr_{t}" for t in range(T)] + \
                 [f"hrv_{t}" for t in range(T)] + [f"eda_{t}" for t in range(T)] + \
                 [f"beh_{b}" for b in range(4)] + ["label"]
        with open(out_fp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for wi, (hr24, hrv24, eda24, beh4, lab) in enumerate(wins):
                w.writerow([pp, wi] + hr24 + hrv24 + eda24 + beh4 + [lab])
        for _, _, _, _, lab in wins:
            class_counts[lab] += 1
        total_windows += len(wins)
        print(f"  {pp}: {len(wins)} windows")
    print(f"\ntotal windows: {total_windows}")
    print(f"class distribution  N(0)={class_counts[0]}  I(1)={class_counts[1]}  T(2)={class_counts[2]}")
    print(f"output directory: {OUT}")


if __name__ == "__main__":
    main()
