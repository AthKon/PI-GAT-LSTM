"""
scan_corrupt_cases.py -- rescan every FE case for solver blowups using a
SPATIAL-COHERENCE criterion, not just a magnitude threshold.

Why this exists
---------------
The blacklist in the notebook (cell f80f8d84) excludes a case when its tension
is non-finite OR peaks above 1.5 MN. That catches the obvious divergences but
misses a whole class: a single node blowing up to a few hundred kN while its
immediate neighbours stay at normal load, or even at exactly zero. One such
case (loc10 / case0100, peak 522 kN) survived the blacklist and is the sole
source of test_peak_tension_max_abs_error = 519,912 N in BOTH the v1 and v2
FINAL runs -- i.e. it is a deterministic data artifact, not a model effect.

The physical discriminator
--------------------------
A real snap load is SPATIALLY COHERENT: the whole slack section goes taut at
once, so tension is near-uniform along it and each node is comparable to its
neighbours. A numerical blowup is NODE-LOCAL: one node departs from both of
its neighbours by a large factor. So, at every sample,

    coh[t,i] = T[t,i] / max( T[t,i-1], T[t,i+1] )

stays near 1 for physics and explodes for artifacts. Measured on the known
bad case, the offending sample reads

    0, 0, 22253, 522030, 0, 20844, 6982      -> coh = 23.5

whereas a genuine snap (loc11/case0011) has five adjacent nodes jumping
together to a uniform ~15.4 kN, coh ~ 1.0.

A second, independent axis is TEMPORAL isolation (the Finding-1 ratio),

    iso[t,i] = T[t,i] / max( (T[t-1,i] + T[t+1,i]) / 2, 1 )

which separates one-sample spikes from resolved events. Physical snaps score
high on iso but LOW on coh; artifacts score high on both. Reporting both keeps
the two populations separable instead of collapsing them into one "outlier"
bucket.

Cost
----
Reads only tension.npy per case (~1.2 MB each, memory-mapped, one at a time),
so peak RAM is a few MB regardless of dataset size. This is NOT a notebook run
-- it never builds datasets, fits normalizers, or loads a model.

Usage
-----
    python scan_corrupt_cases.py                     # scan everything
    python scan_corrupt_cases.py --locs 10,11        # a subset
    python scan_corrupt_cases.py --out scan.csv
    python scan_corrupt_cases.py --report scan.csv   # re-analyse without rescanning
"""
import argparse
import csv
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

CACHE_DIR = pathlib.Path(r"C:\Users\thano\Desktop\data\cache_npy")

# Current notebook blacklist (cell f80f8d84) -- non-finite OR peak > 1.5 MN.
CORRUPT_CASES = frozenset({
    (1, 63), (1, 202), (1, 215),
    (4, 45), (4, 133), (4, 164), (4, 270),
    (5, 31), (5, 35), (5, 46), (5, 108), (5, 194), (5, 197), (5, 202), (5, 262), (5, 264),
    (6, 22), (6, 131), (6, 134),
    (7, 16), (7, 32), (7, 83), (7, 132), (7, 137), (7, 174), (7, 212), (7, 214),
    (7, 220), (7, 221), (7, 225), (7, 229),
    (8, 37), (8, 94), (8, 136), (8, 143), (8, 247),
    (9, 43), (9, 167), (9, 193),
    (10, 22), (10, 56), (10, 77), (10, 187), (10, 188), (10, 224), (10, 234),
    (10, 239), (10, 275), (10, 278),
    (11, 3), (11, 22), (11, 61), (11, 89),
})

# Only test where the tension is actually large; a 5 N node next to a 1 N node
# is a meaningless 5x. 10 kN sits above the routine load range (case p90 peak
# = 14 kN) without hiding the moderate blowups this scan exists to find.
COH_FLOOR_N = 10_000.0

# Three conditions, ALL required. They are independent, and each alone has
# false positives that the others reject -- established by inspecting the
# flagged and retained populations sample by sample:
#
#   ISO_CUT  one-sample rise out of near-nothing. A genuine snap rises out of
#            the wave envelope (a few kN), so it scores iso ~ 1.0-1.5 even
#            though it is also a single 0.1 s sample (Finding 1). An artifact
#            rises from ~0 N, so it scores in the hundreds or thousands. This
#            is what separates the two populations, NOT spike width.
#   COH_CUT  node-local: the node departs from BOTH neighbours.
#   FAIR_CUT monotonicity violation: in a catenary the fairlead carries the
#            most load, so a node exceeding the fairlead's own maximum by a
#            large factor is non-physical. Needed because COH is fooled when
#            2+ ADJACENT nodes blow up together (loc06/case0120: nodes 0 and 1
#            at 133/121 kN with the rest of the line at exactly 0, coh = 1.1).
#
# A sample is a blowup when it is temporally isolated AND (node-local OR
# monotonicity-violating). Retained coherent events reach 45 kN; rejected
# artifacts start at 10 kN -- so this is a shape test, not a magnitude cut.
ISO_CUT = 10.0
COH_CUT = 3.0
FAIR_CUT = 3.0

FIELDS = [
    "loc", "case", "n_samples", "n_nonfinite", "peak", "peak_node", "peak_t",
    "peak_iso", "peak_coh", "peak_over_fairlead",
    "fairlead_max", "interior_max", "worst_coh", "worst_coh_T",
    "worst_coh_node", "worst_coh_t", "worst_coh_iso",
    "n_blowup", "n_blowup_severe", "blowup_max_T", "blacklisted",
]


def scan_case(loc, case, path):
    """Per-case statistics. Returns a dict, or None if the case has no tension."""
    f = path / "tension.npy"
    if not f.exists():
        return None
    T = np.asarray(np.load(f, mmap_mode="r"), dtype=np.float32)
    if T.ndim != 2:
        return None

    finite = np.isfinite(T)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite:
        T = np.where(finite, T, 0.0)

    S, N = T.shape
    flat = int(np.argmax(T))
    peak_t, peak_node = divmod(flat, N)

    # --- spatial coherence: each node against the LARGER of its two neighbours.
    # Using max() (not mean) is deliberately conservative -- a node only looks
    # incoherent if it exceeds BOTH neighbours, so a genuine taut section that
    # steps up along the line is never flagged.
    left = np.empty_like(T)
    left[:, 1:] = T[:, :-1]
    left[:, 0] = T[:, 1]            # anchor has one neighbour; reuse it
    right = np.empty_like(T)
    right[:, :-1] = T[:, 1:]
    right[:, -1] = T[:, -2]         # fairlead likewise
    coh = T / np.maximum(np.maximum(left, right), 1.0)

    # --- temporal isolation: each sample against the mean of its own node one
    # step before and after. Genuine snaps rise out of the wave envelope and
    # score ~1; blowups rise out of ~0 N and score in the hundreds.
    up = np.empty_like(T)
    up[1:] = T[:-1]
    up[0] = T[0]
    dn = np.empty_like(T)
    dn[:-1] = T[1:]
    dn[-1] = T[-1]
    iso = T / np.maximum((up + dn) * 0.5, 1.0)

    fairlead_max = float(T[:, -1].max())

    # --- the joint criterion
    big = T > COH_FLOOR_N
    blow = big & (iso > ISO_CUT) & (
        (coh > COH_CUT) | (T > FAIR_CUT * max(fairlead_max, 1.0))
    )
    n_blowup = int(blow.sum())
    blowup_max_T = float(T[blow].max()) if n_blowup else 0.0
    n_blowup_severe = int((blow & (T > 50_000.0)).sum())

    if big.any():
        masked = np.where(big, coh, 0.0)
        wflat = int(np.argmax(masked))
        wt, wn = divmod(wflat, N)
        worst_coh = float(masked.flat[wflat])
        worst_coh_T = float(T[wt, wn])
        worst_iso = float(iso[wt, wn])
    else:
        worst_coh = worst_coh_T = worst_iso = 0.0
        wt = wn = -1

    return {
        "loc": loc, "case": case, "n_samples": S,
        "n_nonfinite": n_nonfinite,
        "peak": float(T[peak_t, peak_node]),
        "peak_node": int(peak_node), "peak_t": int(peak_t),
        "peak_iso": float(iso[peak_t, peak_node]),
        "peak_coh": float(coh[peak_t, peak_node]),
        "peak_over_fairlead": float(T[peak_t, peak_node] / max(fairlead_max, 1.0)),
        "fairlead_max": fairlead_max,
        "interior_max": float(T[:, 1:-1].max()),
        "worst_coh": worst_coh, "worst_coh_T": worst_coh_T,
        "worst_coh_node": int(wn), "worst_coh_t": int(wt),
        "worst_coh_iso": worst_iso,
        "n_blowup": n_blowup, "n_blowup_severe": n_blowup_severe,
        "blowup_max_T": blowup_max_T,
        "blacklisted": int((loc, case) in CORRUPT_CASES),
    }


def scan_all(locs, out_path):
    rows = []
    t0 = time.time()
    for loc in locs:
        ldir = CACHE_DIR / f"loc{loc:02d}"
        if not ldir.is_dir():
            print(f"  loc{loc:02d}: missing, skipped")
            continue
        cases = sorted(ldir.glob("case_*"))
        for k, cdir in enumerate(cases):
            case = int(cdir.name.split("_")[1])
            r = scan_case(loc, case, cdir)
            if r is not None:
                rows.append(r)
            if (k + 1) % 100 == 0:
                print(f"  loc{loc:02d}: {k+1}/{len(cases)}  ({time.time()-t0:.0f}s)")
        print(f"loc{loc:02d} done ({len(cases)} cases, {time.time()-t0:.0f}s elapsed)")

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}  ({len(rows)} cases, {time.time()-t0:.0f}s)")
    return rows


def report(rows):
    print("\n" + "=" * 78)
    print("SCAN REPORT")
    print("=" * 78)

    n = len(rows)
    bl = [r for r in rows if r["blacklisted"]]
    ok = [r for r in rows if not r["blacklisted"]]
    print(f"{n} cases scanned | {len(bl)} already blacklisted | {len(ok)} in use")

    nf = [r for r in ok if r["n_nonfinite"] > 0]
    if nf:
        print(f"\n[!] {len(nf)} IN-USE cases contain non-finite tension:")
        for r in nf[:20]:
            print(f"      loc{r['loc']:02d} case{r['case']:04d}  {r['n_nonfinite']} samples")

    peaks = np.array([r["peak"] for r in ok])
    print(f"\nPeak tension over in-use cases [kN]:")
    for q in (50, 90, 99, 99.9):
        print(f"   p{q:<5} {np.percentile(peaks, q)/1000:10.2f}")
    print(f"   max    {peaks.max()/1000:10.2f}")

    # --- the criterion that matters
    flagged = [r for r in ok if r["n_blowup"] > 0]
    flagged.sort(key=lambda r: -r["blowup_max_T"])
    print(f"\n{'='*78}")
    print(f"SOLVER BLOWUPS among IN-USE cases")
    print(f"  T > {COH_FLOOR_N/1000:.0f} kN  AND  iso > {ISO_CUT}  AND  "
          f"(coh > {COH_CUT}  OR  T > {FAIR_CUT}x fairlead_max)")
    print("=" * 78)
    if not flagged:
        print("  none")
    else:
        print(f"  {len(flagged)} case(s) flagged\n")
        print(f"  {'case':<16} {'bad T [kN]':>11} {'iso':>10} {'coh':>9} "
              f"{'T/fair':>8} {'n_bad':>6} {'peak [kN]':>10}")
        for r in flagged:
            print(f"  loc{r['loc']:02d} case{r['case']:04d}   "
                  f"{r['blowup_max_T']/1000:11.1f} {r['peak_iso']:10.1f} "
                  f"{r['peak_coh']:9.1f} {r['peak_over_fairlead']:8.1f} "
                  f"{r['n_blowup']:6d} {r['peak']/1000:10.1f}")

    # --- the retained population: proof this is a shape test, not a magnitude cut
    keep_hi = [r for r in ok if r["n_blowup"] == 0 and r["peak"] > 20_000]
    keep_hi.sort(key=lambda r: -r["peak"])
    print(f"\nRETAINED high-load cases (coherent, temporally resolved = real "
          f"physics): {len(keep_hi)} above 20 kN")
    for r in keep_hi[:10]:
        print(f"  loc{r['loc']:02d} case{r['case']:04d}   peak "
              f"{r['peak']/1000:7.1f} kN  iso {r['peak_iso']:6.2f}  "
              f"coh {r['peak_coh']:5.2f}  T/fair {r['peak_over_fairlead']:5.2f}")

    # sanity: does the criterion re-find what the magnitude rule already caught?
    caught = sum(1 for r in bl if r["n_blowup"] > 0 or r["n_nonfinite"] > 0)
    print(f"\nSanity: {caught}/{len(bl)} already-blacklisted cases are independently "
          f"flagged by this criterion")

    kept = np.array([r["peak"] for r in ok if r["n_blowup"] == 0])
    print(f"\nPeak tension AFTER removal [kN]: p99 {np.percentile(kept,99)/1000:.2f}"
          f"  p99.9 {np.percentile(kept,99.9)/1000:.2f}  max {kept.max()/1000:.2f}")

    print(f"\nProposed blacklist additions ({len(flagged)}) -- "
          f"{100*len(flagged)/len(ok):.2f}% of in-use cases:")
    if flagged:
        s = sorted((r["loc"], r["case"]) for r in flagged)
        print("    " + ", ".join(f"({l}, {c})" for l, c in s))
    else:
        print("    (none)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locs", default="1,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--out", default="corrupt_scan.csv")
    ap.add_argument("--report", default=None,
                    help="re-analyse an existing CSV instead of rescanning")
    args = ap.parse_args()

    if args.report:
        with open(args.report, newline="", encoding="utf-8") as fh:
            rows = [{k: (float(v) if "." in v or "e" in v.lower() else int(v))
                     for k, v in r.items()} for r in csv.DictReader(fh)]
        report(rows)
        return

    locs = [int(v) for v in args.locs.replace(" ", "").split(",") if v]
    print(f"Scanning {CACHE_DIR}  locs={locs}")
    rows = scan_all(locs, pathlib.Path(args.out))
    report(rows)


if __name__ == "__main__":
    main()
