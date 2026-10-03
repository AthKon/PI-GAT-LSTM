"""EXACT per-split snap reachability, replicating build_leakage_aware_splits.

Difference vs snap_split_accounting.py: that script treats the development
region as ONE contiguous interval [0, cut). The notebook does NOT. It chunks
the dev region into raw_block_len=2000 blocks and calls
valid_window_starts_inside_raw_interval() PER BLOCK, so a train/val window must
fit entirely inside a single 2000-sample block. With window_len=1000 that means
only starts [b_start, b_end-1000] are ever candidates, and an event within 999
samples of a block boundary can be unreachable even though the dev region is
long enough. This script models that exactly.

Config replicated (v4 run cell a04f1f5f + TrainingConfig defaults):
  window_len=1000, raw_block_len=2000, test_time_fraction=0.30,
  train_lc_ids=(1,3,4,5,6,7,8,9), test_extra_lc_ids=(10,11).

Inputs (defaults = the TRAINING-era files, whose output this script reproduced before the
arguments existed):
  --catalogue snap_catalogue.csv      the events to account for
  --mask      artifact_samples.csv    the per-sample mask
The paper's Table 3 uses --catalogue snap_catalogue_cat.csv --mask artifact_samples_cat.csv
(the catenary rule, CLAUDE.md 12.14 / 16.20). The whole-case policy then drops every
simulation the given mask touches (126 with the evaluation mask, 109 with the training one).

Policies:
  v3_gc  : whole-case exclusion of every simulation in the mask, NO per-window mask.
  v4     : whole-case exclusion of NON-FINITE (11) + TOO_SHORT (2) only,
           plus the per-sample mask (shape OR 50 kN ceiling OR non-finite).

Reachability -- "does at least one legal, clean window covering this event
exist in this split" -- is what decides whether the pipeline can ever show the
event to the model. It is NOT the same as "the event's own sample is clean",
because a 1000-sample window straddles many artifacts.
"""
import argparse
import csv
import os
import pathlib
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE
CACHE = pathlib.Path(os.environ.get("MOORING_CACHE_DIR", r"C:\Users\thano\Desktop\data\cache_npy"))
# code repository: the derived tables live in data/
TABLES = HERE if (HERE / "snap_catalogue.csv").exists() else HERE / "data"

W = 1000
BLOCK = 2000
TEST_FRAC = 0.30
DEV = {1, 3, 4, 5, 6, 7, 8, 9}
OOD = {10, 11}
CEILING = 50_000.0
TOO_SHORT = {(5, 202), (5, 262)}


def n_steps_of(loc, case):
    """Record length straight from the .npy header (no data read)."""
    p = CACHE / f"loc{loc:02d}" / f"case_{case:04d}" / "tension.npy"
    if not p.exists():
        return None
    return int(np.load(p, mmap_mode="r").shape[0])


def load_snaps(path):
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append((int(r["loc"]), int(r["case"]), int(r["t"]),
                        float(r["peak"]), int(float(r["rise"]))))
    return out


def load_mask(path):
    bad, nonfinite, ceil_cases, prov = {}, set(), set(), {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            k = (int(r["loc"]), int(r["case"]))
            bad[k] = np.array(sorted(int(x) for x in r["bad_t"].split() if x), dtype=np.int64)
            prov[k] = (int(r["n_shape"]), int(r["n_ceiling"]), int(r["n_nonfinite"]))
            if int(r["n_nonfinite"]):
                nonfinite.add(k)
            if int(r["n_ceiling"]):
                ceil_cases.add(k)
    return bad, nonfinite, ceil_cases, prov


def clean_starts(lo, hi_excl, badset):
    """Legal starts s in [lo, hi_excl-W] whose window [s, s+W) has no bad sample."""
    hi = hi_excl - W          # inclusive
    if hi < lo:
        return np.empty(0, np.int64)
    s = np.arange(lo, hi + 1, dtype=np.int64)
    if badset is None or len(badset) == 0:
        return s
    b = badset[(badset >= lo) & (badset < hi_excl)]
    if b.size == 0:
        return s
    # a start s is killed iff some bad t satisfies s <= t < s+W  <=>  t-W < s <= t
    keep = np.ones(s.size, bool)
    for t in b:
        i0 = np.searchsorted(s, t - W + 1, "left")
        i1 = np.searchsorted(s, t, "right")
        keep[i0:i1] = False
    return s[keep]


def covers(starts, te):
    """Is there a start in `starts` with s <= te < s+W ?"""
    if starts.size == 0:
        return False
    i0 = np.searchsorted(starts, te - W + 1, "left")
    i1 = np.searchsorted(starts, te, "right")
    return i1 > i0


def split_of(loc, te, n):
    if loc in OOD:
        return "test_ood"
    cut = int(np.floor((1.0 - TEST_FRAC) * n))
    return "test_temporal" if te >= cut else "train_val"


def candidates(loc, n, badset, per_window):
    """Exactly the candidate start lists build_leakage_aware_splits produces."""
    bs = badset if per_window else None
    if loc in OOD:
        return {"test_ood": clean_starts(0, n, bs)}
    cut = int(np.floor((1.0 - TEST_FRAC) * n))
    out = {"test_temporal": clean_starts(cut, n, bs)}
    dev = []
    cur = 0
    while cur < cut:                       # chunk_raw_time_range
        nxt = min(cur + BLOCK, cut)
        st = clean_starts(cur, nxt, bs)    # <-- per BLOCK, this is the key constraint
        if st.size:
            dev.append(st)
        cur = nxt
    out["train_val"] = np.concatenate(dev) if dev else np.empty(0, np.int64)
    return out


def account(snaps, bad, nonfinite, steps, policy):
    if policy == "v3_gc":
        drop_whole, per_window = set(bad), False
    else:
        drop_whole, per_window = (nonfinite | TOO_SHORT), True

    cache = {}
    res = defaultdict(lambda: [0, 0])
    for loc, case, te, peak, rise in snaps:
        k = (loc, case)
        n = steps.get(k)
        if n is None:
            continue
        sp = split_of(loc, te, n)
        res[sp][1] += 1
        if k in drop_whole or n < W:
            continue
        ck = (k, policy)
        if ck not in cache:
            cache[ck] = candidates(loc, n, bad.get(k), per_window)
        if covers(cache[ck][sp], te):
            res[sp][0] += 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", type=pathlib.Path, default=TABLES / "snap_catalogue.csv")
    ap.add_argument("--mask", type=pathlib.Path, default=TABLES / "artifact_samples.csv")
    args = ap.parse_args()
    print(f"catalogue: {args.catalogue.name} | mask: {args.mask.name}")
    snaps = load_snaps(args.catalogue)
    bad, nonfinite, ceil_cases, prov = load_mask(args.mask)

    keys = {(l, c) for l, c, *_ in snaps}
    steps = {}
    for k in sorted(keys):
        ns = n_steps_of(*k)
        if ns is not None:
            steps[k] = ns

    genuine = [s for s in snaps if s[3] <= CEILING]
    print("=" * 80)
    print("SNAP-EVENT ACCOUNTING  (exact, block-aware)")
    print("=" * 80)
    print(f"catalogued events            : {len(snaps)} in "
          f"{len({(s[0], s[1]) for s in snaps})} cases")
    print(f"  genuine (peak <= {CEILING/1e3:.0f} kN)   : {len(genuine)}")
    print(f"  above the ceiling          : {len(snaps) - len(genuine)}  "
          f"(reclassified as artifacts)")
    print(f"mask cases                   : {len(bad)}  "
          f"| non-finite {len(nonfinite)} | ceiling fires in {len(ceil_cases)}")
    only_ceiling = sum(1 for k, v in prov.items() if v[1] and not v[0] and not v[2])
    print(f"cases flagged ONLY by ceiling: {only_ceiling}")
    print()

    for label, pool in (("ALL catalogued events", snaps),
                        (f"GENUINE only (peak <= {CEILING/1e3:.0f} kN)", genuine)):
        print(f"--- {label} ---")
        r3 = account(pool, bad, nonfinite, steps, "v3_gc")
        r4 = account(pool, bad, nonfinite, steps, "v4")
        print("  %-16s %8s %14s %14s %9s" % ("split", "present", "v3_gc", "v4", "gain"))
        tt = ta = tb = 0
        for sp in ("train_val", "test_temporal", "test_ood"):
            t, a, b = r3[sp][1], r3[sp][0], r4[sp][0]
            tt += t; ta += a; tb += b
            print("  %-16s %8d %8d %4.0f%% %8d %4.0f%% %+9d"
                  % (sp, t, a, 100*a/max(t, 1), b, 100*b/max(t, 1), b - a))
        print("  %-16s %8d %8d %4.0f%% %8d %4.0f%% %+9d"
              % ("TOTAL", tt, ta, 100*ta/max(tt, 1), tb, 100*tb/max(tt, 1), tb - ta))
        print()

    # ---- ceiling sweep -------------------------------------------------
    print("--- ceiling sensitivity (v4 policy, GENUINE = peak <= that ceiling) ---")
    print("  %-9s %10s %12s %14s" % ("ceiling", "events<=c", "reachable", "reach(all)"))
    for c in (20e3, 30e3, 35e3, 50e3, 75e3, 100e3):
        pool = [s for s in snaps if s[3] <= c]
        r = account(pool, bad, nonfinite, steps, "v4")
        ra = account(snaps, bad, nonfinite, steps, "v4")
        print("  %6.0f kN %10d %12d %14d"
              % (c/1e3, len(pool), sum(v[0] for v in r.values()),
                 sum(v[0] for v in ra.values())))

    # ---- rise composition ----------------------------------------------
    print()
    print("--- rise-time composition of the genuine catalogue ---")
    comp = defaultdict(int)
    for s in genuine:
        comp[s[4]] += 1
    for k in sorted(comp):
        print(f"  rise={k} : {comp[k]:5d}  ({100*comp[k]/len(genuine):4.1f}%)")

    # ---- peak bands ----------------------------------------------------
    print()
    print("--- catalogued event peak bands ---")
    edges = [0, 10e3, 20e3, 30e3, 50e3, 75e3, 100e3, 1e12]
    names = ["<10 kN", "10-20", "20-30", "30-50", "50-75", "75-100", ">=100 kN"]
    pk = np.array([s[3] for s in snaps])
    for lo_, hi_, nm in zip(edges[:-1], edges[1:], names):
        print(f"  {nm:>10s} : {int(((pk >= lo_) & (pk < hi_)).sum()):5d}")


if __name__ == "__main__":
    main()
