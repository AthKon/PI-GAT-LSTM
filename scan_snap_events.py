"""
scan_snap_events.py -- catalogue every SNAP-LOAD event and every SOLVER-ARTIFACT
sample in the cache, so that artifacts can be masked per sample instead of per
case, and snaps can be found and sampled deliberately.

Why this exists
---------------
`scan_corrupt_cases.py` answers "is this case contaminated?" and the pipeline
then throws the whole case away. But contamination is extremely sparse -- a
median of 4 bad samples in a 13,801-sample simulation -- and the discarded cases
also contain genuine, physically valid snap loads. This script separates the two
populations sample by sample.

Discriminating a snap from an artifact
--------------------------------------
Both are one-sample events, so DURATION cannot separate them. What separates
them is SPATIAL STRUCTURE:

  * a snap is the whole taut section going tight at once, so tension is near
    uniform along it (coh ~ 1) and never exceeds the fairlead (fair <= 1).
    This is forced by physics: the axial wave speed is sqrt(EA/rhoA) = 2871 m/s,
    which crosses the line in 0.037 s, i.e. within ONE 0.1 s output sample.
  * an artifact is node-local: one or two nodes read many times their immediate
    neighbours and many times the fairlead, which no catenary can do.

NOTE (measured, do not re-derive): the line's EXTENSION is *not* a discriminator.
Chord/L0 sits above 99.5% for 96-100% of all samples, so "the line is nearly
straight" is the normal state, not a snap signature. The slack-pay-out ->
arrest sequence is real and visible in the geometry, but it is a MECHANISM
confirmation, not a usable classifier. Coherence and fairlead monotonicity do
the actual work.

Two corrections applied 2026-08-16 (both DEFAULT ON; --legacy reproduces the
original 1611-event catalogue exactly for provenance)
-----------------------------------------------------------------------------
GUARD A -- catenary monotonicity veto (--guard-k, default 3.0).
  A whole-line one-sample artifact whose profile *decreases* monotonically
  anchor -> fairlead is spatially SMOOTH, so `coh` never fires, and the
  `T > 3 x fairlead_max` branch is dead whenever the case's own fairlead_max is
  itself contaminated. With the artifact rule silent the timestep stays eligible
  for snap classification, and the per-node test then catalogues the ONE node
  where the artifact's ramp happens to satisfy both T > 3 kN and fair <= 1.5.
  Worked example: loc10/case0224 t=7603 (24,175 N at node 0 falling to 511 N at
  node 17) was catalogued as a 4.1 kN "snap" at node 14.
  The veto is per TIMESTEP: reject the whole sample if the anchor out-pulls the
  fairlead by more than k. Measured over all 3000 cases: in never-flagged cases
  99.7% (2882/2891) have ZERO such samples, 32 in total, versus 7,049 inside the
  109 flagged cases.

GUARD B -- robust fairlead reference (--fair-ref-pct, default 99.9).
  The artifact rule's `T > 3 x fairlead_max` branch uses the max of a possibly
  CONTAMINATED series (loc11/case0022's fairlead_max IS its own 943 MN blowup,
  so the threshold became 2.8 GN and the branch could never fire). Use a high
  percentile instead. Measured: correct but nearly inert once the pipeline's
  50 kN ceiling is in place (+10 samples in 5 cases). Set --fair-ref-pct 100 to
  restore the raw max.

GUARD C -- inherit the pipeline's magnitude ceiling (--ceiling, default 50 kN).
  A sample above ARTIFACT_TENSION_CEILING_N is masked out of every built window,
  so a "snap" catalogued there can never be scored -- and the argmax fix below
  makes it actively harmful, since inside a diverged case the largest member of
  a refractory group IS the divergence. Measured: without guard C the argmax fix
  RAISES the count of catalogued events above 50 kN from 27 to 36. Non-finite
  timesteps are vetoed for the same reason. This is not a new criterion; it is
  the mask's own RULE M applied to the catalogue so the two agree.

REFRACTORY ARGMAX -- keep the strongest sample in each group (--refractory-keep).
  The 20-sample (2.0 s) refractory window groups a snap with its own aliased
  ringdown, which is correct, but the loop kept the FIRST passing timestep in
  each group rather than the largest. Measured over all 3000 cases: 644 passing
  timesteps are grouped away, and in 274 groups (17.0% of all events) a dropped
  timestep is >5% larger than the kept one (p50 1.56x, p90 5.45x, max 939x).
  Worked example: loc04/case0135 kept a 15.5 kN precursor at t=9773 and threw
  away the 22.7 kN main event at t=9783. Grouping BOUNDARIES are unchanged (they
  are still defined by the first passing timestep), so the event COUNT is
  preserved and no event can be double-counted -- only the reported timestep,
  node and magnitude move to the strongest member of the group.

GUARD D -- same-instant catenary test on EVERY station (added 2026-09-23,
  --cat-k, default 1.0; CLAUDE.md 12.14). Guard A only looked at the anchor, at
  3x the fairlead. The evaluation mask now carries RULE C of
  scan_artifact_samples.py (any station > k x the fairlead at the same sample,
  station tension > 1 kN), so a timestep it flags is an ARTIFACT and cannot be a
  snap; the snap rule's own bound (was cat <= 1.5) is tightened to the same k, so
  the two rules partition the one-sample spikes. Measured: 556 of the 1388
  catalogued events had their own station above the fairlead -- 47 % of those in
  flagged simulations, 7 % in clean ones. --pre-cat reproduces the 1388-event
  catalogue of 2026-08-16 exactly.

Outputs
-------
  snap_catalogue.csv   one row per snap event: loc, case, t, node, peak, iso,
                       coh, fair, rise (samples to rise = how well 0.1 s
                       sampling resolves it), pre-slack tension
  snap_by_case.csv     per case: counts, the non-snap fairlead ceiling (used to
                       decide whether a given WINDOW contains a snap), and the
                       artifact sample indices for per-sample masking

Cost: memory-maps one case's tension.npy at a time and reduces it immediately;
peak RSS stays a few tens of MB. This is NOT a notebook run.
"""
import argparse
import csv
import os
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

CACHE = pathlib.Path(os.environ.get("MOORING_CACHE_DIR", r"C:\Users\thano\Desktop\data\cache_npy"))
LOCS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
N_CASES = 300

# artifact rule (scan_corrupt_cases.py, unchanged)
ART_T, ART_ISO, ART_COH, ART_FAIR = 10_000.0, 10.0, 3.0, 3.0
# snap rule
SNAP_T, SNAP_ISO, SNAP_COH, SNAP_FAIR, SNAP_SLACK = 3_000.0, 3.0, 3.0, 1.5, 0.25
# guard A: reject a whole timestep whose anchor out-pulls the fairlead by > k
GUARD_K = 3.0
# guard B: percentile used as the fairlead reference in the artifact rule
FAIR_REF_PCT = 99.9
# guard C: the pipeline's own magnitude ceiling (ARTIFACT_TENSION_CEILING_N /
# scan_artifact_samples.py RULE M). A sample above it is masked out of every
# built window, so catalogueing a "snap" there is meaningless -- and the
# refractory-argmax fix makes it ACTIVELY harmful, because inside a diverged
# case the largest member of a group is the divergence. Measured: without this
# the argmax fix raises the number of catalogued events above 50 kN from 27
# to 36. Per timestep on the line max, exactly as the mask does.
CEILING_N = 50_000.0
REFRACTORY = 20          # samples; groups a snap with its own aliased ringdown
# guard D: the evaluation mask's RULE C (scan_artifact_samples.py). 0 disables.
CAT_K, CAT_FLOOR = 1.0, 1_000.0


def neighbour_max(T):
    """max of each node's two along-line neighbours, [t, n]."""
    left = np.empty_like(T)
    right = np.empty_like(T)
    left[:, 1:] = T[:, :-1]
    left[:, 0] = T[:, 1]
    right[:, :-1] = T[:, 1:]
    right[:, -1] = T[:, -2]
    return np.maximum(left, right)


def scan_case(loc, case, guard_k=GUARD_K, fair_ref_pct=FAIR_REF_PCT,
              refractory_keep="max", guard_artifacts=False, ceiling=CEILING_N,
              cat_k=CAT_K, snap_fair=None):
    d = CACHE / f"loc{loc:02d}" / f"case_{case:04d}" / "tension.npy"
    if not d.exists():
        return None
    T = np.asarray(np.load(d, mmap_mode="r"), np.float64)
    nS, nN = T.shape
    finite = np.isfinite(T)
    n_nonfinite = int((~finite).sum())
    nonfin_t = ~finite.all(axis=1)
    T = np.where(finite, T, 0.0)

    iso = np.ones_like(T)
    iso[1:-1] = T[1:-1] / np.maximum(0.5 * (T[:-2] + T[2:]), 1.0)
    coh = T / np.maximum(neighbour_max(T), 1.0)
    fmax = float(T[:, -1].max())
    fair = T / np.maximum(T[:, -1:], 1.0)

    # guard B: a contaminated fairlead_max kills the T > 3 x fairlead_max branch
    fref = (fmax if fair_ref_pct >= 100.0
            else max(float(np.percentile(T[:, -1], fair_ref_pct)), 1.0))
    # guard A: anchor out-pulling the fairlead is not a catenary. Per TIMESTEP.
    bad_grad = ((T[:, 0] > guard_k * np.maximum(T[:, -1], 1.0))
                if guard_k > 0 else np.zeros(nS, bool))

    art = (T > ART_T) & (iso > ART_ISO) & ((coh > ART_COH) | (T > ART_FAIR * fref))
    if guard_artifacts:                  # off by default: changing the MASK
        art[bad_grad, :] = True          # forces a pipeline re-run, not an eval
    # guard D: the evaluation mask's RULE C, on every station but the fairlead
    if cat_k > 0:
        _num = T[:, :-1]
        bad_cat = ((_num > CAT_FLOOR)
                   & (_num > cat_k * np.maximum(T[:, -1:], 1.0))).any(axis=1)
        art[bad_cat, :] = True           # it IS in the evaluation mask
    if snap_fair is None:
        snap_fair = cat_k if cat_k > 0 else SNAP_FAIR
    art_t = np.unique(np.where(art)[0])

    # snap candidates: isolated, coherent, never above the fairlead, and the node
    # was slack immediately before
    pre = np.full_like(T, np.inf)
    for k in range(1, 11):
        pre[k:] = np.minimum(pre[k:], T[:-k])
    snap = ((T > SNAP_T) & (iso > SNAP_ISO) & (coh < SNAP_COH)
            & (fair <= snap_fair) & (pre < SNAP_SLACK * T))
    snap[art.any(1), :] = False          # never call an artifact sample a snap
    snap[bad_grad, :] = False            # guard A
    # guard C + non-finite: align the veto with the mask the PIPELINE applies,
    # so no catalogued event can sit on a sample no window can ever contain
    snap[T.max(axis=1) > ceiling, :] = False
    snap[nonfin_t, :] = False
    snap[:12] = False
    snap[-6:] = False

    # peak of the snap-passing nodes at each candidate timestep
    ts = np.unique(np.where(snap)[0])
    tpeak = {int(t): float(T[t][snap[t]].max()) for t in ts}

    # Group with the ORIGINAL boundary rule (a group opens at its first passing
    # timestep and absorbs everything within REFRACTORY of it) so the event
    # COUNT is identical; then report the strongest member of the group.
    groups, cur, last = [], [], -99
    for t in ts:
        t = int(t)
        if t - last < REFRACTORY:
            cur.append(t)
            continue
        if cur:
            groups.append(cur)
        cur, last = [t], t
    if cur:
        groups.append(cur)

    events = []
    for g in groups:
        t = g[0] if refractory_keep == "first" else max(g, key=lambda u: tpeak[u])
        n = int(np.where(snap[t])[0][np.argmax(T[t][snap[t]])])
        base = float(np.median(T[t - 10:t - 4, n]))
        exc = T[t - 4:t + 1, n] - base
        rise = int((exc > 0.25 * max(exc[-1], 1.0)).sum())
        events.append(dict(loc=loc, case=case, t=int(t), node=n,
                           peak=float(T[t, n]), fairlead=float(T[t, -1]),
                           iso=float(iso[t, n]), coh=float(coh[t, n]),
                           fair=float(fair[t, n]), rise=rise,
                           pre_slack=float(pre[t, n]),
                           t_first=int(g[0]), n_grouped=len(g),
                           peak_first=float(tpeak[g[0]])))

    # fairlead ceiling excluding snap and artifact samples: a test window whose
    # true fairlead peak exceeds this MUST contain one of those events
    bad = np.zeros(nS, bool)
    bad[art_t] = True
    for e in events:
        bad[max(0, e["t"] - 2):e["t"] + 3] = True
    quiet = float(T[~bad, -1].max()) if (~bad).any() else fmax

    return dict(loc=loc, case=case, n_samples=nS, n_nonfinite=n_nonfinite,
                n_artifact_samples=len(art_t), n_snaps=len(events),
                fairlead_max=fmax, quiet_fairlead_max=quiet,
                artifact_t=" ".join(str(int(v)) for v in art_t)), events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-events", default="snap_catalogue_cat.csv")
    ap.add_argument("--out-cases", default="snap_by_case_cat.csv")
    ap.add_argument("--cat-k", type=float, default=CAT_K,
                    help="guard D: veto (and mark as artifact) a timestep where any "
                         "station above 1 kN exceeds k x the fairlead; the snap "
                         "rule's own bound follows it. 0 disables.")
    ap.add_argument("--pre-cat", action="store_true",
                    help="reproduce the 1388-event catalogue of 2026-08-16 "
                         "(no guard D, snap bound 1.5) into snap_catalogue.csv")
    ap.add_argument("--locs", default="")
    ap.add_argument("--guard-k", type=float, default=GUARD_K,
                    help="guard A: veto a timestep whose anchor exceeds k x the "
                         "fairlead. 0 disables. Calibration (verified-genuine "
                         "events lost / 231): k=1.0 -> 16, 2.0 -> 8, 3.0 -> 3.")
    ap.add_argument("--fair-ref-pct", type=float, default=FAIR_REF_PCT,
                    help="guard B: percentile of the fairlead series used as the "
                         "artifact rule's reference. 100 = the raw max (legacy).")
    ap.add_argument("--refractory-keep", choices=("max", "first"), default="max",
                    help="which timestep of a refractory group to catalogue")
    ap.add_argument("--ceiling", type=float, default=CEILING_N,
                    help="guard C: veto any timestep whose line max exceeds this, "
                         "matching the pipeline mask (scan_artifact_samples.py "
                         "RULE M). inf disables.")
    ap.add_argument("--guard-artifacts", action="store_true",
                    help="ALSO mask guard-A timesteps as artifacts. This changes "
                         "the exclusion MASK, so it requires re-measuring "
                         "reachability and re-running FINAL -- not eval-only.")
    ap.add_argument("--legacy", action="store_true",
                    help="reproduce the original catalogue (no guards, keep the "
                         "first timestep of each refractory group)")
    a = ap.parse_args()
    if a.legacy:
        a.guard_k, a.fair_ref_pct, a.refractory_keep = 0.0, 100.0, "first"
        a.guard_artifacts, a.ceiling = False, float("inf")
    if a.legacy or a.pre_cat:
        a.cat_k = 0.0
        if a.out_events == "snap_catalogue_cat.csv":
            a.out_events, a.out_cases = "snap_catalogue.csv", "snap_by_case.csv"
    locs = [int(v) for v in a.locs.split(",")] if a.locs else LOCS
    print(f"[cfg] guard_k={a.guard_k} fair_ref_pct={a.fair_ref_pct} "
          f"refractory_keep={a.refractory_keep} ceiling={a.ceiling:,.0f} "
          f"guard_artifacts={a.guard_artifacts} cat_k={a.cat_k}")

    rows, evs = [], []
    for loc in locs:
        got = 0
        for case in range(1, N_CASES + 1):
            r = scan_case(loc, case, guard_k=a.guard_k,
                          fair_ref_pct=a.fair_ref_pct,
                          refractory_keep=a.refractory_keep,
                          guard_artifacts=a.guard_artifacts,
                          ceiling=a.ceiling, cat_k=a.cat_k)
            if r is None:
                continue
            row, e = r
            rows.append(row)
            evs.extend(e)
            got += 1
        print(f"  loc{loc:02d}: {got} cases, "
              f"{sum(x['n_snaps'] for x in rows if x['loc'] == loc)} snaps, "
              f"{sum(x['n_artifact_samples'] for x in rows if x['loc'] == loc)} "
              f"artifact samples")

    with open(a.out_cases, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(a.out_events, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(evs[0].keys()))
        w.writeheader()
        w.writerows(evs)

    ns = sum(r["n_samples"] for r in rows)
    na = sum(r["n_artifact_samples"] for r in rows)
    print(f"\n{len(rows)} cases | {ns:,} samples")
    print(f"  artifact samples : {na:,} ({100 * na / ns:.5f}%) in "
          f"{sum(1 for r in rows if r['n_artifact_samples'])} cases")
    print(f"  snap events      : {len(evs):,} in "
          f"{len({(e['loc'], e['case']) for e in evs})} cases")
    print(f"  wrote {a.out_events}, {a.out_cases}")


if __name__ == "__main__":
    main()
