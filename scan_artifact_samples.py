"""Build the per-window exclusion mask: every corrupt SAMPLE in every case.

Two rules, OR-ed.  Both are necessary and neither subsumes the other.

  RULE S -- shape (scan_corrupt_cases.py, unchanged)
      T > 10 kN  AND  iso > 10  AND  ( coh > 3  OR  T > 3 x fairlead_max )
    Catches the ONE-SAMPLE spike geometries.  Structurally blind to a
    sustained divergence: once the solve has diverged, neighbouring samples
    are diverged too, so iso ~ 1 and the conjunct can never fire.

  RULE M -- magnitude ceiling
      max_i T[t,i] > ARTIFACT_TENSION_CEILING_N        (default 50 kN)
    Covers exactly that blind spot.  Justification is PROVENANCE, not
    magnitude: measured over the whole cache, 558/558 samples that this rule
    catches and RULE S misses lie in cases whose record elsewhere exceeds
    1 MN.  The ceiling is 1.53x the maximum tension found anywhere in the
    2891 cases RULE S never fires on (32,751 N) and 2.45x the largest
    verified physical snap (20,411 N), so it is set from cases it does not
    filter.  Sensitivity 35/50/75 kN -> 1030/1030/1053 reachable snap events.

  RULE C -- same-instant catenary monotonicity (added 2026-09-23, CLAUDE.md 12.14)
      any station i < fairlead with  T[t,i] > CAT_FLOOR (1 kN)
                                AND  T[t,i] > k x max(T[t,fairlead], 1 N)      k = 1.0
    No station may out-pull the fairlead at the same instant. In this solver the
    line is frictionless on the seabed and axially quasi-static at dt = 0.1 s:
    median (T_f - T_anchor) = 239.3 N/m x depth against w_s = 240.5 N/m over the
    2891 clean simulations. Samples violating it miss the axial momentum balance
    against the recorded motion by 7-25 kN (ordinary samples: 0.1-0.25 kN), and
    snap candidates in clean simulations drop from 65 in [0.9,1.0) to 1 in [1.0,1.1).
    RULE C catches the whole-line one-sample geometry with a tension that FALLS
    towards the fairlead, which is spatially smooth (coh silent) and whose record
    fairlead maximum is often contaminated (fair silent).
    WARNING: the two shipped models were TRAINED on the mask WITHOUT rule C
    (artifact_samples.csv). The mask with rule C is the EVALUATION mask
    (artifact_samples_cat.csv); --catenary-k 0 --out artifact_samples.csv
    reproduces the training mask exactly.

  Non-finite samples are also marked, though the 11 cases carrying them are
  dropped whole (one missed NaN poisons the normalizer fit).

Writes artifact_samples_cat.csv (default):
    loc,case,n_samples,n_shape,n_ceiling,n_nonfinite,n_catenary,n_bad,peak,bad_t
  n_catenary counts samples flagged by RULE C and by neither S nor M.
Runs locally in ~1 min, one memory-mapped case at a time (peak RSS ~35 MB).
"""
import argparse
import csv
import os
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

CACHE = pathlib.Path(os.environ.get("MOORING_CACHE_DIR", r"C:\Users\thano\Desktop\data\cache_npy"))
OUT = pathlib.Path(__file__).with_name("artifact_samples_cat.csv")
ART_T, ART_ISO, ART_COH, ART_FAIR = 10_000.0, 10.0, 3.0, 3.0
CEILING = 50_000.0
CAT_K, CAT_FLOOR = 1.0, 1_000.0     # RULE C; k = 0 disables it


def neighbour_max(T):
    left = np.empty_like(T)
    right = np.empty_like(T)
    left[:, 1:] = T[:, :-1]
    left[:, 0] = T[:, 1]
    right[:, :-1] = T[:, 1:]
    right[:, -1] = T[:, -2]
    return np.maximum(left, right)


def scan_case(path, ceiling, cat_k=CAT_K, cat_floor=CAT_FLOOR):
    raw = np.asarray(np.load(path, mmap_mode="r"), np.float64)
    n_samples = raw.shape[0]
    nonfin = (~np.isfinite(raw)).any(axis=1)
    T = np.where(np.isfinite(raw), raw, 0.0)

    iso = np.ones_like(T)
    iso[1:-1] = T[1:-1] / np.maximum(0.5 * (T[:-2] + T[2:]), 1.0)
    coh = T / np.maximum(neighbour_max(T), 1.0)
    fmax = float(T[:, -1].max())
    shape = ((T > ART_T) & (iso > ART_ISO)
             & ((coh > ART_COH) | (T > ART_FAIR * fmax))).any(axis=1)

    tmax = T.max(axis=1)
    ceil_hit = tmax > ceiling
    if cat_k > 0:
        num = T[:, :-1]
        cat_hit = ((num > cat_floor)
                   & (num > cat_k * np.maximum(T[:, -1:], 1.0))).any(axis=1)
    else:
        cat_hit = np.zeros(n_samples, bool)
    bad = shape | ceil_hit | nonfin | cat_hit
    return dict(
        n_samples=n_samples,
        n_shape=int(shape.sum()),
        n_ceiling=int((ceil_hit & ~shape).sum()),
        n_nonfinite=int(nonfin.sum()),
        n_catenary=int((cat_hit & ~shape & ~ceil_hit).sum()),
        n_bad=int(bad.sum()),
        bad_t=np.where(bad)[0],
        peak=float(tmax.max()),
        peak_kept=float(tmax[~bad].max()) if (~bad).any() else 0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceiling", type=float, default=CEILING)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--catenary-k", type=float, default=CAT_K,
                    help="RULE C: k x fairlead at the same sample; 0 disables "
                         "(reproduces the training mask artifact_samples.csv)")
    ap.add_argument("--catenary-floor", type=float, default=CAT_FLOOR)
    args = ap.parse_args()

    rows = []
    tot_shape = tot_ceil = tot_cat = tot_bad = 0
    worst_kept = 0.0
    for loc_dir in sorted(CACHE.glob("loc*")):
        loc = int(loc_dir.name[3:])
        for case_dir in sorted(loc_dir.glob("case_*")):
            p = case_dir / "tension.npy"
            if not p.exists():
                continue
            case = int(case_dir.name.split("_")[1])
            r = scan_case(p, args.ceiling, args.catenary_k, args.catenary_floor)
            tot_shape += r["n_shape"]
            tot_ceil += r["n_ceiling"]
            tot_cat += r["n_catenary"]
            tot_bad += r["n_bad"]
            if r["n_bad"] < r["n_samples"]:
                worst_kept = max(worst_kept, r["peak_kept"])
            if r["n_bad"]:
                rows.append((loc, case, r))
        print(f"  loc{loc:02d} done ({len(rows)} affected cases so far)")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cat_col = args.catenary_k > 0
        w.writerow(["loc", "case", "n_samples", "n_shape", "n_ceiling",
                    "n_nonfinite"] + (["n_catenary"] if cat_col else [])
                   + ["n_bad", "peak", "bad_t"])
        for loc, case, r in rows:
            w.writerow([loc, case, r["n_samples"], r["n_shape"], r["n_ceiling"],
                        r["n_nonfinite"]] + ([r["n_catenary"]] if cat_col else [])
                       + [r["n_bad"], f'{r["peak"]:.6g}',
                        " ".join(str(int(t)) for t in r["bad_t"])])

    print()
    print(f"ceiling                     : {args.ceiling:,.0f} N")
    print(f"catenary k / floor          : {args.catenary_k} / {args.catenary_floor:,.0f} N")
    print(f"cases with >=1 bad sample   : {len(rows)}")
    print(f"samples flagged by SHAPE    : {tot_shape:,}")
    print(f"samples flagged by CEILING  : {tot_ceil:,}   (shape missed these)")
    print(f"samples flagged by CATENARY : {tot_cat:,}   (shape and ceiling missed these)")
    print(f"total corrupt samples       : {tot_bad:,}")
    print(f"worst tension left in a KEPT sample : {worst_kept:,.0f} N")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
