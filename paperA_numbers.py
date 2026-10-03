"""Generate paper_A_manuscript/numbers.tex -- every result macro the manuscript quotes.

Implements the "recompute before quoting" rule (CLAUDE.md, top of file): no result
number is ever typed into the prose by hand. Each macro carries a comment naming the
file and column it came from, plus its full-precision value.

Local only. Reads CSV / npz artefacts and the snap catalogue. Never executes a
notebook (memory feedback_never_run_notebook_locally). Runs in ~30 s, most of which is
exact_split_accounting.py re-deriving snap reachability from the npy cache.

Sources
  S1  Results/checkpoints_P1_matched_v5/                     Stage 1, v5 ep32
  SF  Results/eval_v5_snapfix/                               corrected snap catalogue
  S2  Results/SR_RESULTS/checkpoints_P2_final_warm(run2)/    Stage 2, ep76

Coverage: the abstract (plan Step 6.9). Extend section by section as drafting goes on
-- add a build_* function and register it in MACRO_GROUPS.

Inputs are found through paperA_paths.py: the development layout under Results/ by
default, or the three unpacked 4TU.ResearchData archives when PAPERA_DATA is set
(https://doi.org/10.4121/e3f167ac-2925-4815-956e-5ef3d883f1a5); see that module.

Usage
  python paperA_numbers.py                # regenerate numbers.tex
  python paperA_numbers.py --check        # exit 1 if numbers.tex is out of date
  python paperA_numbers.py --allow-stale  # keep the recorded reachability figures if
                                          # the data drive (cache_npy) is not mounted
  python paperA_numbers.py --write-derived  # development layout: also store the results
                                          # that need non-deposited inputs in data/
"""
import argparse
import collections
import functools
import csv
import pathlib
import pickle
import re
import statistics
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

import paperA_paths as P
from paperA_paths import derived, npload

ROOT = pathlib.Path(__file__).parent
OUT = P.OUT_DIR / "numbers.tex"

S1 = ROOT / "Results" / "checkpoints_P1_matched_v5"
SF = ROOT / "Results" / "eval_v5_snapfix"
S2 = ROOT / "Results" / "SR_RESULTS" / "checkpoints_P2_final_warm(run2)"
# The final evaluation job (EVAL_CLEAN, patch sr-22, job 402698): both shipped models
# re-scored on their published test draws minus every window covering a sample of the
# evaluation mask (artifact_samples_cat.csv). Absent until the job is pulled down.
EVAL_CLEAN = ROOT / "Results" / "SR_RESULTS" / "checkpoints_P2_eval_clean"
# Every TEST-side read goes to that job (landed 2026-09-24, all in-job gates OK): T1 = Stage 1
# on its unpaired draw minus 487 windows, T2 = Stage 2 on its paired draw minus 47 windows x 10
# layouts, with the noise levels on the same clean Stage-2 draw. TRAINING-side reads
# (config.json, metrics_history.csv, the validation metrics_by_node_count.csv, checkpoints)
# stay on S1/S2. EVAL_CLEAN/published/ and stage1/published/ reproduce the published draws.
T1 = EVAL_CLEAN / "stage1"
T2 = EVAL_CLEAN

CACHE = P.CACHE          # MOORING_CACHE_DIR, or the development default

# Solver settings, read from the raw batch metadata (gnl_aout.dat) on 2026-09-15.
# Identical in every case sampled across all ten sites. The raw batch directory is
# not always mounted, so these are recorded rather than re-read on each run.
SOLVER = dict(dt=0.1, sim_s=1380.0, ramp_s=30.0, nw=1441, wc=7.5,
              elem_m=0.75, nx_short=100, nx_long=140)

# Line properties, Santjer et al. (2025) Table 2 -- verified against the PDF 2026-09-16.
# EA in N, mass per unit length (rho*A) in kg/m.
LINE = dict(EA=232.7e6, rhoA=28.23, area=0.003619)
RHO_W, GRAV = 1025.0, 9.81     # sea-water density and gravity used by the notebook (cell 66cf027f)
# submerged weight per unit length, w_s = (rho*A - rho_w * A_disp) g
W_SUB = (LINE["rhoA"] - RHO_W * LINE["area"]) * GRAV

# Design suspended catenary length per site [m], Santjer et al. (2025) Table 3. Our ten
# sites were matched onto their twelve by fairlead x-coordinate from gnl_aout.dat
# (exact to 0.01 m in nine cases, 0.04 m for loc01/loc08 which sit 54.46/54.42 against
# their 54.5/54.4); the two omitted sites are their locations 1 and 4.
L_SUSP = {"loc01": 30.27, "loc03": 36.46, "loc04": 46.54, "loc05": 40.43,
          "loc06": 12.79, "loc07": 42.15, "loc08": 30.36, "loc09": 34.16,
          "loc10": 13.40, "loc11": 39.78}

# Stage-2 shipped epoch, and the output resolution the paper reports.
EP = 76
N_OUT = 21

# The catalogue and mask Data 2.5 reports: the catenary rule (rule C, k = 1.0) in both.
# The models were TRAINED on artifact_samples.csv (no rule C); the evaluation uses the
# _cat mask. The EVAL_CLEAN dumps join on the _cat catalogue (in-job gate: S1 523 rows /
# 154 events, S2 500 / 45).
SNAP_CAT = ROOT / "snap_catalogue_cat.csv"
MASK_EVAL = ROOT / "artifact_samples_cat.csv"
MASK_TRAIN = ROOT / "artifact_samples.csv"

# Used ONLY with --allow-stale, when cache_npy is not mounted. Recorded from the live
# run of exact_split_accounting.py --catalogue snap_catalogue_cat.csv
# --mask artifact_samples_cat.csv on 2026-09-23.
STALE_REACH = {"train_val": (517, 115, 319), "test_temporal": (194, 44, 133),
               "test_ood": (191, 40, 100), "TOTAL": (902, 199, 552)}
_REACH = {}          # filled by build_snaps(); read by table_snapaccounting()


# --------------------------------------------------------------------------- io
def rows(path):
    with P.open_(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def kv(path):
    """A two-column metric CSV (name,value) as a dict."""
    out = {}
    for r in rows(path):
        cols = list(r)
        out[r[cols[0]]] = r[cols[1]]
    return out


def grid_cell(rs, n_in, n_out, key):
    for r in rs:
        if int(r["N_in"]) == n_in and int(r["N_out"]) == n_out:
            return float(r[key])
    raise KeyError((n_in, n_out, key))


def pooled_row(rs, n_out):
    for r in rs:
        if int(r["N_out"]) == n_out:
            return r
    raise KeyError(n_out)


def per_event_capture(ev, true, pred, n_in=None):
    """Median over DISTINCT events of the median pred/true within each event.

    snap_ev rows are [loc, case, N_in, window_start, event_t]: one physical event is
    drawn by many windows, so a per-row statistic over-weights high-multiplicity
    events (CLAUDE.md 13.3). Collapse on (loc, case, t) first.
    """
    groups = collections.defaultdict(list)
    for (lc, case, nin, _start, t), a, b in zip(ev, true, pred):
        if a <= 0 or (n_in is not None and int(nin) != n_in):
            continue
        groups[(int(lc), int(case), int(t))].append(b / a)
    per_event = [statistics.median(v) for v in groups.values()]
    return statistics.median(per_event), len(per_event)


# ----------------------------------------------------------------- macro groups
def build_scope():
    """Dataset and coverage, verified against the artefacts where possible."""
    layouts = sorted({int(r["node_count"])
                      for r in rows(T1 / "test_metrics_by_node_count.csv")})
    n_in_grid = sorted({int(r["N_in"]) for r in rows(T2 / "superres_grid.csv")})
    assert layouts == n_in_grid, (layouts, n_in_grid)

    dump = npload(T2 / "test_timeseries_dump.npz")
    locs = sorted({int(m[0]) for m in dump["meta_all"]})

    return [
        ("nSites", f"{len(locs)}",
         "S2 test_timeseries_dump.npz meta_all, distinct loc ids"),
        ("nSitesHeld", "2", "loc10 and loc11, withheld from training"),
        ("nLayouts", f"{len(layouts)}", "S2 superres_grid.csv, distinct N_in"),
        ("nStationsMin", f"{min(layouts)}", "same"),
        ("nStationsMax", f"{max(layouts)}", "same = the native FE resolution"),
        ("nSeaStates", "3000", "in-use FE cases (CLAUDE.md corrupt-case scan)"),
        ("sampleRate", "10", "FE export interval 0.1 s"),
        ("depthMin", "14", "southern North Sea sites (thesis Ch. 2)"),
        ("depthMax", "45", "same"),
        ("tpMin", "2.8", "cache_npy env.npy col 2 over all 3000 cases, 2.76 s"),
        ("tpMax", "17.8", "same, 17.77 s"),
        ("windowSeconds", "100", "window_len 1000 samples at 0.1 s"),
    ]


def build_stage1():
    """Stage-1 matched-resolution test headline (v5 ep32), clean draw."""
    m = kv(T1 / "test_metrics.csv")
    r2 = float(m["test_global_R2_tension"])
    mae = float(m["test_MAE_tension"])
    mape = float(m["test_MAPE_tension"])
    return [
        ("sOneRsq", f"{r2:.4f}", f"S1 test_metrics.csv test_global_R2_tension = {r2!r}"),
        ("sOneMAE", f"{mae:.1f}", f"S1 test_metrics.csv test_MAE_tension = {mae!r}"),
        ("sOneMAPE", f"{mape:.2f}", f"S1 test_metrics.csv test_MAPE_tension = {mape!r}"),
    ]


def build_stage1_regions():
    """Stage-1 contact-regime R2 on the test set, a uniform sample over all nodes.

    Also emits a floor over the three regimes, which is what the abstract quotes:
    one honest bound is cheaper than three medians where there is no baseline to
    beat, and the discriminating regional result is Stage 2's touchdown cell.
    """
    m = kv(T1 / "test_metrics.csv")
    out, vals = [], []
    for regime, tag in (("grounded", "Grounded"), ("touchdown", "Touchdown"),
                        ("suspended", "Suspended")):
        v = float(m[f"test_contact_{regime}_R2_median"])
        vals.append(v)
        out.append((f"sOne{tag}Rsq", f"{v:.3f}",
                    f"S1 test_metrics.csv test_contact_{regime}_R2_median = {v!r}"))
    floor = int(min(vals) * 100) / 100.0
    out.append(("sOneRegionMinRsq", f"{floor:.2f}",
                f"floor over the three regime medians (min = {min(vals)!r}, "
                f"suspended); quote as 'above'"))
    return out


def build_stage1_calibration():
    """The snap / non-snap split of the peak-tension tails (CLAUDE.md 15.5).

    Fairlead peak over the clean Stage-1 test windows, on the evaluation catalogue. The
    threshold is a quantile of TRUE peak over all windows; the tail is then split by
    whether the window contains a catalogued snap event. up = share of tail entries
    predicted low, bias = mean signed error (prediction minus truth).
    """
    z = npload(T1 / "test_timeseries_dump.npz")
    meta, pt, pp = z["meta_all"], z["peak_true_all"], z["peak_pred_all"]
    snap_keys = {(int(a), int(b), int(c), int(d)) for a, b, c, d, _t in z["snap_ev"]}
    is_snap = np.array([(int(m[0]), int(m[1]), int(m[2]), int(m[3])) in snap_keys
                        for m in meta])
    signed = pp - pt

    out = [("snapWindowShare", f"{100 * is_snap.mean():.2f}",
            f"{int(is_snap.sum())} of {len(meta)} test windows contain a catalogued "
            f"snap event")]
    for q, band in ((0.90, "Ninety"), (0.99, "NinetyNine")):
        thr = float(np.quantile(pt, q))
        sel = pt >= thr
        out.append((f"sOneThr{band}", f"{thr:,.0f}".replace(",", "\\,"),
                    f"S1 clean dump, p{int(q * 100)} of true fairlead peak = "
                    f"{thr!r} N over {len(meta)} windows"))
        out.append((f"sOneSnapShare{band}", f"{100 * is_snap[sel].mean():.0f}",
                    f"share of the p{int(q * 100)} tail that is snap-composed "
                    f"(n = {int(sel.sum())})"))
        for pop, mask in (("NonSnap", sel & ~is_snap), ("Snap", sel & is_snap)):
            s = signed[mask]
            up = 100.0 * (s < 0).mean()
            bias = float(s.mean())
            out.append((f"sOne{pop}Up{band}", f"{up:.1f}",
                        f"{pop} p{int(q * 100)} tail, n = {int(mask.sum())}, "
                        f"underprediction rate = {up!r} %"))
            out.append((f"sOne{pop}Bias{band}", f"{bias:+.0f}".replace("+", "+"),
                        f"same population, mean signed error = {bias!r} N"))
    return out


def build_stage2():
    """Stage-2 headline pooled over all N_in sensor layouts (N_out = 21), test set.

    Unlike build_stage2_sparse (a single grid cell), this is the aggregate
    test_metrics.csv Stage 2 itself writes -- every window scored at every one of
    the 10 layouts, N_out = 21 throughout, on the clean (screened) test draw.
    """
    m = kv(T2 / "test_metrics.csv")
    r2 = float(m["test_global_R2_tension"])
    mae = float(m["test_MAE_tension"])
    mape = float(m["test_MAPE_tension"])
    return [
        ("sTwoRsq", f"{r2:.4f}", f"S2 test_metrics.csv test_global_R2_tension = {r2!r}"),
        ("sTwoMAE", f"{mae:.1f}", f"S2 test_metrics.csv test_MAE_tension = {mae!r}"),
        ("sTwoMAPE", f"{mape:.2f}", f"S2 test_metrics.csv test_MAPE_tension = {mape!r}"),
    ]


def build_stage2_sparse():
    """The headline: four measurement stations -> the 21-station field."""
    g = rows(T2 / "superres_grid.csv")
    gb = rows(T2 / "superres_grid_baseline.csv")

    mae_d4 = grid_cell(g, 4, N_OUT, "MAE_tension")
    mae_b4 = grid_cell(gb, 4, N_OUT, "MAE_tension")
    diag_d = grid_cell(g, N_OUT, N_OUT, "MAE_tension")
    diag_b = grid_cell(gb, N_OUT, N_OUT, "MAE_tension")
    r2_d4 = grid_cell(g, 4, N_OUT, "global_R2_tension")
    r2_b4 = grid_cell(gb, 4, N_OUT, "global_R2_tension")

    # Validation penalty at the shipped epoch: in-distribution, so not diluted by the
    # withheld site loc10, whose error does not depend on N_in (CLAUDE.md 16.15.3).
    val = {int(r["node_count"]): float(r["MAE_tension"])
           for r in rows(S2 / "metrics_by_node_count.csv") if int(r["epoch"]) == EP}
    pen_val = val[4] / val[N_OUT]

    return [
        ("srMAEfour", f"{mae_d4:.1f}",
         f"S2 superres_grid.csv N_in=4 N_out=21 MAE_tension = {mae_d4!r}"),
        ("srMAEbaseFour", f"{mae_b4:.1f}",
         f"S2 superres_grid_baseline.csv N_in=4 N_out=21 MAE_tension = {mae_b4!r}"),
        ("srMAEfull", f"{diag_d:.1f}",
         f"S2 superres_grid.csv N_in=21 N_out=21 MAE_tension = {diag_d!r}"),
        ("srPenTest", f"{mae_d4 / diag_d:.2f}",
         f"S2 grid MAE(4,21)/MAE(21,21) = {mae_d4 / diag_d!r}"),
        ("srPenVal", f"{pen_val:.2f}",
         f"S2 metrics_by_node_count.csv epoch {EP}, MAE N_in=4 / N_in=21 = {pen_val!r}"),
        ("srPenBase", f"{mae_b4 / diag_b:.2f}",
         f"S2 baseline grid MAE(4,21)/MAE(21,21) = {mae_b4 / diag_b!r}"),
        ("srRsqFour", f"{r2_d4:.4f}",
         f"S2 superres_grid.csv N_in=4 N_out=21 global_R2_tension = {r2_d4!r}"),
        ("srRsqBaseFour", f"{r2_b4:.4f}",
         f"S2 baseline N_in=4 N_out=21 global_R2_tension = {r2_b4!r}"),
    ]


def build_full_sensing_anchor():
    """The matched-resolution model scored at N_out = 21 with all 21 stations.

    This is cell (21, 21) of the baseline surface, where interp_nodes is the
    IDENTITY -- so it is the Stage-1 model itself, with no interpolation involved,
    on the same paired test windows as the decoder. It is therefore the honest
    apples-to-apples anchor for "four stations versus full sensing", and it does
    not lean on the interpolation baseline.

    Also emits the share of test windows drawn from the two withheld sites.
    """
    gb = rows(T2 / "superres_grid_baseline.csv")
    mae = grid_cell(gb, N_OUT, N_OUT, "MAE_tension")
    r2 = grid_cell(gb, N_OUT, N_OUT, "global_R2_tension")

    # The SAME checkpoint (v5, ep32) scored on ITS OWN test draw (Table 6 / build_results_matched),
    # not the Stage-2 paired draw above. The two draws are independent samples of the same test
    # population, so this differs from `mae` by a few percent even though the predictions at
    # N = 21 are identical -- see the Results Sec. 4 intro. Quantified here, not left implicit,
    # so the gap is never mistaken for a discrepancy between the model and itself.
    per_n_s1 = {int(r["node_count"]): r for r in rows(T1 / "test_metrics_by_node_count.csv")}
    mae_s1_own = float(per_n_s1[N_OUT]["MAE_tension"])
    gap_pct = 100 * abs(mae_s1_own - mae) / mae_s1_own

    dump = npload(T2 / "test_timeseries_dump.npz")
    locs = np.array([int(m[0]) for m in dump["meta_all"]])
    ood = np.isin(locs, (10, 11))

    return [
        ("sOneFullMAE", f"{mae:.1f}",
         f"S2 superres_grid_baseline.csv N_in=21 N_out=21 MAE_tension = {mae!r} "
         f"-- the Stage-1 model, identity interpolation, paired test windows"),
        ("sOneFullRsq", f"{r2:.4f}",
         f"same cell, global_R2_tension = {r2!r}"),
        ("sOneFullMAETable", f"{mae_s1_own:.1f}",
         f"S1 test_metrics_by_node_count.csv N=21 MAE_tension = {mae_s1_own!r} "
         f"-- same checkpoint, Stage-1's own (non-paired) test draw; = Table 6's N=21 row"),
        ("sOneFullGapPct", f"{gap_pct:.1f}",
         f"relative gap between sOneFullMAE and sOneFullMAETable = {gap_pct!r} %, "
         f"sampling variation between the two draws"),
        ("oodTestShare", f"{100 * ood.mean():.0f}",
         f"{int(ood.sum())} of {len(locs)} test windows come from loc10/loc11, "
         f"withheld from training ({100 * ood.mean():.2f} %)"),
    ]


def build_stage2_regions():
    """Contact-regime R2 at N_out = 21, pooled over N_in: the touchdown result."""
    p = pooled_row(rows(T2 / "superres_grid_pooled.csv"), N_OUT)
    b = pooled_row(rows(T2 / "superres_grid_pooled_baseline.csv"), N_OUT)
    out = []
    for regime, tag in (("touchdown", "Touchdown"), ("grounded", "Grounded"),
                        ("suspended", "Suspended")):
        col = f"contact_{regime}_R2_median"
        vd, vb = float(p[col]), float(b[col])
        out += [
            (f"sr{tag}Rsq", f"{vd:.3f}",
             f"S2 superres_grid_pooled.csv N_out=21 {col} = {vd!r}"),
            (f"sr{tag}RsqBase", f"{vb:.3f}",
             f"S2 superres_grid_pooled_baseline.csv N_out=21 {col} = {vb!r}"),
        ]
    return out


@derived
def snap_reachability():
    """Reachable snap events per split (exact_split_accounting.py, reads the npy cache)."""
    proc = subprocess.run([sys.executable, str(ROOT / "exact_split_accounting.py"),
                           "--catalogue", str(P.src(SNAP_CAT)), "--mask", str(P.src(MASK_TRAIN))],
                          capture_output=True, text=True, timeout=1800, cwd=ROOT)
    block = proc.stdout.split("--- ALL catalogued events ---")[1].split("---")[0]
    got = {}
    for sp in ("train_val", "test_temporal", "test_ood", "TOTAL"):
        m = re.search(sp + r"\s+(\d+)\s+(\d+)\s+\d+%\s+(\d+)\s+\d+%", block)
        if not m:
            raise RuntimeError(f"row {sp} not found")
        got[sp] = tuple(int(m.group(j)) for j in (1, 2, 3))
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-400:])
    note = (f"exact_split_accounting.py (block-aware) --catalogue {SNAP_CAT.name} "
            f"--mask {MASK_TRAIN.name}, TOTAL row")
    return got, note


def build_snaps(allow_stale=False):
    """Snap-event reachability under the two exclusion policies, and per-event capture."""
    split_rows = dict(STALE_REACH)
    note = "STALE -- cache_npy not mounted; re-run with the data drive attached"
    try:
        split_rows, note = snap_reachability()
    except Exception as exc:                                             # noqa: BLE001
        if not allow_stale:
            sys.exit(f"[numbers] exact_split_accounting.py failed: {exc}\n"
                     f"[numbers] attach the data drive, or pass --allow-stale.")
        print(f"[numbers] WARNING: using stale reachability ({exc})")
    for sp in ("train_val", "test_temporal", "test_ood"):
        assert split_rows[sp][0] > 0
    assert tuple(sum(split_rows[sp][j] for sp in ("train_val", "test_temporal", "test_ood"))
                 for j in range(3)) == split_rows["TOTAL"], split_rows
    n_cat = len(rows(SNAP_CAT))
    assert split_rows["TOTAL"][0] == n_cat, (split_rows["TOTAL"], n_cat)
    _REACH.update(split_rows)
    reach = dict(zip(("present", "before", "after"), split_rows["TOTAL"]))

    before = 100.0 * reach["before"] / reach["present"]
    after = 100.0 * reach["after"] / reach["present"]

    # Per-event snap capture on the clean draws: Stage 1, and Stage 2 by N_in.
    z1 = npload(T1 / "test_timeseries_dump.npz")
    cat = {(int(r["loc"]), int(r["case"]), int(r["t"]))
           for r in rows(SNAP_CAT)}
    miss = sum(1 for a, b, _, _, t in z1["snap_ev"]
               if (int(a), int(b), int(t)) not in cat)
    assert miss == 0, f"{miss} dump events absent from {SNAP_CAT.name} -- wrong version"
    cap1, n1 = per_event_capture(z1["snap_ev"], z1["snap_true"], z1["snap_pred"])

    z2 = npload(T2 / "test_timeseries_dump.npz")
    cap4, n4 = per_event_capture(z2["snap_ev"], z2["snap_true"], z2["snap_pred"], n_in=4)
    cap21, _ = per_event_capture(z2["snap_ev"], z2["snap_true"], z2["snap_pred"], n_in=N_OUT)

    return [
        ("snapEvents", f"{reach['present']}", f"{SNAP_CAT.name}; {note}"),
        ("snapEventSims", f"{len({(r['loc'], r['case']) for r in rows(SNAP_CAT)})}",
         f"distinct simulations holding a catalogued event, {SNAP_CAT.name}"),
        ("snapReachBefore", f"{before:.0f}",
         f"{reach['before']}/{reach['present']} reachable under whole-case exclusion"),
        ("snapReachAfter", f"{after:.0f}",
         f"{reach['after']}/{reach['present']} reachable under per-window exclusion "
         f"with the training mask ({MASK_TRAIN.name}, the catenary condition is a snap condition only)"),
        ("snapCaptureSone", f"{cap1:.2f}",
         f"S1 clean dump, per-event median pred/true over {n1} distinct events = {cap1!r}"),
        ("snapCapturePct", f"{100 * cap1:.0f}",
         "same, as a percentage of the recorded peak magnitude"),
        ("snapCaptureFour", f"{cap4:.2f}",
         f"S2 clean dump, N_in=4, per-event median over {n4} events = {cap4!r}"),
        ("snapCaptureFull", f"{cap21:.2f}", f"S2 clean dump, N_in=21 = {cap21!r}"),
    ]


@derived
def build_campaign():
    """The simulation campaign: forcing ranges, record geometry, solver cost.

    Hs comes from env.npy col 1 over all 3000 cases and L0 from the node spacing
    in reference.npy. Falls back to the recorded values if cache_npy is absent.
    """
    hs_lo = hs_hi = None
    if CACHE.exists():
        hs, l0 = [], set()
        for d in sorted(CACHE.iterdir()):
            cases = sorted(d.iterdir())
            for c in cases:
                hs.append(float(npload(c / "env.npy")[1]))
            ref = npload(cases[0] / "reference.npy", mmap_mode="r")
            l0.add(round(float(ref[0, 1] - ref[0, 0]) * 20))
        hs_lo, hs_hi = min(hs), max(hs)
        assert l0 == {75, 105}, l0
    if hs_lo is None:                       # recorded 2026-09-15 from the cache
        hs_lo, hs_hi = 0.415, 8.415

    # nw components span [0, wc] inclusive => nw-1 intervals. Verified against the
    # measured autocorrelation lag of 12064 samples (1206.4 s) in 30/30 cases.
    repeat_s = 2 * np.pi / (SOLVER["wc"] / (SOLVER["nw"] - 1))
    return [
        ("hsMin", f"{hs_lo:.1f}", f"cache env.npy col 1 over all 3000 cases, {hs_lo!r} m"),
        ("hsMax", f"{hs_hi:.1f}", f"same, {hs_hi!r} m"),
        ("lineShort", "75", "cache reference.npy node spacing x 20, six sites"),
        ("lineLong", "105", "same, four sites"),
        ("simSeconds", f"{SOLVER['sim_s']:.0f}", "gnl_aout.dat simT"),
        ("simMinutes", f"{SOLVER['sim_s'] / 60:.0f}", "same"),
        ("recordSamples", "13801", "rows of gnl_data1.dat = tension.npy length"),
        ("solverStep", "0.1",
         "gnl_aout.dat simDt -- the INTEGRATION step, NOT an export decimation"),
        ("rampSeconds", f"{SOLVER['ramp_s']:.0f}", "gnl_aout.dat StartRamp"),
        ("elemSize", "0.75", f"L0/nx: 75/{SOLVER['nx_short']} = 105/{SOLVER['nx_long']} m"),
        ("repeatSeconds", f"{repeat_s:.1f}",
         f"2*pi/(wc/(nw-1)), wc={SOLVER['wc']} rad/s, nw={SOLVER['nw']} = {repeat_s!r} s; "
         "measured autocorrelation lag 12064 samples in 30/30 cases"),
        ("solverHoursMed", "37",
         "gnl_aout.dat [TIM] Total Time, 30 cases sampled across all sites: 25.8-59.6 h"),
        ("solverHoursMin", "26", "same"),
        ("solverHoursMax", "60", "same"),
    ]


def build_axial():
    """Axial wave speed and the frequency limits that bound a recorded snap peak.

    All analytic, from LINE and SOLVER -- nothing here is fitted or measured from a
    run. Three bands matter and they must not be conflated:

      * the integrator's dissipation threshold 1/(10 dt). agarwal2026finitestrain
        integrates with Generalised-alpha under asymptotic annihilation and states
        that response above this frequency is removed from the solution.
      * the record's Nyquist frequency 1/(2 dt).
      * the line's axial band, c/(4L). Over the FULL line (L0 = 75 or 105 m) this is
        the lowest it can be; the suspended segment, which is what rings after an
        arrest, is shorter and therefore higher.

    CLAUDE.md's "6.8-29 Hz" is wrong in both directions -- it pairs the full-line
    lower bound with a suspended-length estimate that undershoots the real one.
    """
    c = (LINE["EA"] / LINE["rhoA"]) ** 0.5
    dt = SOLVER["dt"]
    full = {75.0: c / (4 * 75.0), 105.0: c / (4 * 105.0)}
    susp = {k: c / (4 * v) for k, v in L_SUSP.items()}
    ls_lo, ls_hi = min(L_SUSP.values()), max(L_SUSP.values())
    return [
        ("axialSpeed", f"{c:.0f}",
         f"sqrt(EA/rhoA) = sqrt({LINE['EA']:.4g}/{LINE['rhoA']}) = {c!r} m/s"),
        ("axialCrossLong", f"{105.0 / c:.3f}",
         f"L0/c for the 105 m lines = {105.0 / c!r} s"),
        ("axialCrossShort", f"{75.0 / c:.3f}",
         f"L0/c for the 75 m lines = {75.0 / c!r} s"),
        ("ringFullLo", f"{full[105.0]:.1f}",
         f"c/(4*105) = {full[105.0]!r} Hz, full-line quarter-wave, long lines"),
        ("ringFullHi", f"{full[75.0]:.1f}",
         f"c/(4*75) = {full[75.0]!r} Hz, full-line quarter-wave, short lines"),
        ("ringSuspLo", f"{c / (4 * ls_hi):.0f}",
         f"c/(4*{ls_hi}) = {c / (4 * ls_hi)!r} Hz, longest suspended segment (loc04)"),
        ("ringSuspHi", f"{c / (4 * ls_lo):.0f}",
         f"c/(4*{ls_lo}) = {c / (4 * ls_lo)!r} Hz, shortest suspended segment (loc06); "
         f"per-site range {min(susp.values())!r}-{max(susp.values())!r} Hz"),
        ("dissipFreq", f"{1.0 / (10 * dt):.0f}",
         f"1/(10*dt) at dt={dt} s; agarwal2026finitestrain state that response above "
         "this frequency is dissipated out of the solution (Chung and Hulbert 1993)"),
        ("nyquistFreq", f"{1.0 / (2 * dt):.0f}",
         f"1/(2*dt) at dt={dt} s, the archived record's Nyquist frequency"),
    ]


def build_curation():
    """Artifact mask size (measured). artifact_samples.csv, one row per flagged simulation.

    The wave-train repeat check moved to build_repeat() (per-entry metrics, both stages).
    """
    mask = rows(MASK_TRAIN)
    n_bad = sum(int(r["n_bad"]) for r in mask)
    ev = rows(MASK_EVAL)
    n_ev = sum(int(r["n_bad"]) for r in ev)
    n_ceiling_cases = sum(1 for r in ev if int(r["n_ceiling"]) > 0)
    only_ceiling = sum(1 for r in ev if int(r["n_ceiling"]) and not int(r["n_shape"])
                       and not int(r["n_nonfinite"]) and not int(r.get("n_catenary", 0)))
    assert n_ceiling_cases == sum(1 for r in mask if int(r["n_ceiling"]) > 0)
    assert only_ceiling == 0, only_ceiling
    # The ceiling adds SAMPLES, not simulations: the samples it flags in the training mask, and
    # the check that every simulation it fires in was already flagged by the shape or the
    # non-finite rule (so it condemns no simulation of its own).
    n_ceiling_samples = sum(int(r["n_ceiling"]) for r in mask)
    assert all(int(r["n_shape"]) or int(r["n_nonfinite"]) for r in mask if int(r["n_ceiling"]))
    total_samples = 3000 * 13801

    # Non-circular bound for the 50 kN ceiling: the largest tension recorded anywhere in a
    # simulation in which neither of the other artifact rules fires (shape, non-finite;
    # ceiling_derivation.csv) AND the fairlead remains the most heavily loaded station at every
    # sample (no fairlead-condition sample, n_catenary in the evaluation mask). None of these
    # depends on the ceiling. Without the last condition the reference would be loc08/0229 at
    # 32.8 kN -- the ANCHOR at 3.07 x the fairlead, a sample the fairlead condition rejects
    # (user, 2026-09-24: step 4 of the 2.5 review).
    ceil_rows = rows(ROOT / "ceiling_derivation.csv")
    fair_viol = {(int(r["loc"]), int(r["case"])) for r in ev if int(r["n_catenary"]) > 0}
    clean = [r for r in ceil_rows if int(r["n_shape"]) == 0 and int(r["n_nonfinite"]) == 0
             and (int(r["loc"]), int(r["case"])) not in fair_viol]
    top = max(clean, key=lambda r: float(r["peak_all"]))
    clean_max_n = float(top["peak_all"])
    assert (int(top["loc"]), int(top["case"])) == (11, 296), (top["loc"], top["case"])

    out = [
        ("maskSamples", f"{n_bad}",
         f"{MASK_TRAIN.name} (the TRAINING mask, no catenary rule), sum of n_bad over "
         f"{len(mask)} flagged simulations"),
        ("maskShare", f"{100 * n_bad / total_samples:.4f}",
         f"{n_bad} of 3000 x 13801 = {total_samples} archived samples"),
        ("maskSims", f"{len(mask)}", f"simulations the training mask touches, {MASK_TRAIN.name}"),
        ("maskEvalSamples", f"{n_ev}",
         f"{MASK_EVAL.name} (the EVALUATION mask, with the catenary rule), sum of n_bad"),
        ("maskEvalSims", f"{len(ev)}", f"simulations the evaluation mask touches"),
        ("maskEvalShare", f"{100 * n_ev / total_samples:.3f}",
         f"{n_ev} of {total_samples} archived samples"),
        ("ceilingRefSims", f"{len(clean)}",
         "simulations with no shape, non-finite or fairlead-condition sample (use with \\num)"),
        ("ceilingCleanMax", f"{clean_max_n / 1000.0:.1f}",
         f"ceiling_derivation.csv: largest tension recorded anywhere in the {len(clean)} "
         f"simulations where no shape, non-finite or fairlead-condition sample occurs = "
         f"{clean_max_n!r} N, loc{int(top['loc']):02d}/case{int(top['case']):04d}"),
        ("ceilingRatio", f"{50_000.0 / clean_max_n:.1f}", "50 kN over that reference"),
        ("ceilingCases", f"{n_ceiling_cases}",
         "simulations in which the 50 kN ceiling fires; none is flagged by it alone"),
        ("ceilingSamples", f"{n_ceiling_samples}",
         f"samples the 50 kN ceiling flags in {MASK_TRAIN.name} (sum of n_ceiling); "
         f"a sample may also satisfy the shape rule, so the rule counts overlap"),
    ]

    return out


CAT_STUDY = ROOT / "cat_study"


@derived
def build_catenary():
    """Data 2.5: why no station may out-pull the fairlead at the same sample (rule C).

    Reads the two per-case scans of cat_study/ (python scan_cat.py; python snapcand.py,
    ~2 min each, CLAUDE.md 12.14) and the two masks. Every quantity is recomputed here.
    """
    for f in ("scan_cat.pkl", "snapcand.pkl"):
        if not (CAT_STUDY / f).exists():
            sys.exit(f"[numbers] cat_study/{f} missing -- run python {f[:-4]}.py in cat_study/")
    R = pickle.loads((CAT_STUDY / "scan_cat.pkl").read_bytes())
    cand = pickle.loads((CAT_STUDY / "snapcand.pkl").read_bytes())
    tr = rows(MASK_TRAIN)
    drop = {(int(r["loc"]), int(r["case"])) for r in tr if int(r["n_nonfinite"])}
    drop |= {(5, 202), (5, 262)}                    # too short for one window

    # 1. static relation over the clean simulations (shape and non-finite never fire)
    cl = [v for v in R.values() if v["clean"] and "h_med" in v]
    h = np.array([v["h_med"] for v in cl])
    d = np.array([v["dTa_med"] for v in cl])
    slope, icpt = np.linalg.lstsq(np.vstack([h, np.ones_like(h)]).T, d, rcond=None)[0]
    corr = float(np.corrcoef(h, d)[0, 1])
    assert abs(slope - W_SUB) / W_SUB < 0.01, (slope, W_SUB)
    mmin = np.array([v["ma_q"][0] for v in cl])
    mmed = np.array([v["ma_q"][4] for v in cl])
    margin_p1 = float(np.percentile(mmin, 1))

    # 2. station above the fairlead at the same sample (k = 1.0, numerator > 1 kN)
    key = ("any", 1000.0, 1.0)
    trm = {(int(r["loc"]), int(r["case"])) for r in tr}
    c_samp = c_sims = f_samp = f_sims = f_inuse = 0
    c_total = 0
    for k, v in R.items():
        if k in drop:
            continue
        n_hit = int(v[key].size)
        if v["clean"]:
            c_total += v["n"]
            c_samp += n_hit
            c_sims += n_hit > 0
        else:
            f_inuse += 1
            f_samp += n_hit
            f_sims += n_hit > 0
    assert c_sims <= c_samp and f_samp > 100 * c_samp

    # 3. snap candidates (no cat bound) by the largest station/fairlead ratio at the sample
    mx = np.array([o[7] for o in cand])
    isc = np.array([o[8] for o in cand])
    below = int(((mx >= 0.9) & (mx < 1.0) & isc).sum())
    above = int(((mx >= 1.0) & (mx < 1.1) & isc).sum())
    assert below > 10 * max(above, 1)

    # 4. axial momentum residual at those candidates, from the archived motion (mom.py)
    mu, dt = LINE["rhoA"], SOLVER["dt"]
    bycase = collections.defaultdict(list)
    for o in cand:
        bycase[(o[0], o[1])].append(o)
    res = []
    for (l, c), lst in bycase.items():
        p = CACHE / f"loc{l:02d}" / f"case_{c:04d}"
        T = npload(p / "tension.npy", mmap_mode="r")
        x = npload(p / "x_abs.npy", mmap_mode="r")
        z = npload(p / "z_abs.npy", mmap_mode="r")
        m = np.full(21, mu * float(npload(p / "reference.npy")[0, -1]) / 20)
        m[[0, -1]] *= 0.5
        for o in lst:
            t = o[2]
            if t < 1 or t >= T.shape[0] - 1:
                continue
            Tt = np.asarray(T[t], float)
            X = np.asarray(x[t - 1:t + 2], float)
            Z = np.asarray(z[t - 1:t + 2], float)
            ax = (X[2] - 2 * X[1] + X[0]) / dt ** 2
            az = (Z[2] - 2 * Z[1] + Z[0]) / dt ** 2
            tx, tz = np.gradient(X[1]), np.gradient(Z[1])
            nn = np.hypot(tx, tz)
            inert = float((m * (ax * tx / nn + az * tz / nn)).sum())
            res.append((o[7], abs((Tt[-1] - Tt[0]) - W_SUB * (Z[1, -1] - Z[1, 0]) - inert)))
    res = np.array(res)
    mom_lo = float(np.median(res[res[:, 0] < 1.0, 1]))
    mom_hi = float(np.median(res[res[:, 0] >= 1.5, 1]))
    bands = [float(np.median(res[(res[:, 0] >= a) & (res[:, 0] < b), 1]))
             for a, b in ((1.0, 1.2), (1.2, 1.5), (1.5, 3.0), (3.0, 1e9))]
    assert mom_lo < bands[0] < bands[1] < bands[2] < bands[3], (mom_lo, bands)

    # 5. what rule C costs: training/validation windows, clean-simulation snap events
    import exact_split_accounting as E                                    # noqa: E402
    bad_tr = {k: np.array(sorted(v), np.int64) for k, v in _mask_by_case(MASK_TRAIN).items()}
    bad_ev = {k: np.array(sorted(v), np.int64) for k, v in _mask_by_case(MASK_EVAL).items()}
    n_tv = {"tr": 0, "ev": 0}
    for p in sorted(CACHE.glob("loc*/case_*/tension.npy")):
        k = (int(p.parent.parent.name[3:]), int(p.parent.name[5:]))
        if k in drop or k[0] not in REPEAT_DEV:
            continue
        n = int(npload(p, mmap_mode="r").shape[0])
        for tag, bad in (("tr", bad_tr), ("ev", bad_ev)):
            n_tv[tag] += E.candidates(k[0], n, bad.get(k), True)["train_val"].size
    pool_share = 100.0 * (1 - n_tv["ev"] / n_tv["tr"])

    cdr = {(int(r["loc"]), int(r["case"])): r for r in rows(ROOT / "ceiling_derivation.csv")}
    shape_clean = {k for k, r in cdr.items()
                   if int(r["n_shape"]) == 0 and int(r["n_nonfinite"]) == 0}
    ev_old = sum(1 for r in rows(ROOT / "snap_catalogue.csv")
                 if (int(r["loc"]), int(r["case"])) in shape_clean)
    ev_new = sum(1 for r in rows(SNAP_CAT) if (int(r["loc"]), int(r["case"])) in shape_clean)

    # 6. the published test draws: windows covering a sample the catenary rule adds
    ext = {k: np.array(sorted(v - _mask_by_case(MASK_TRAIN).get(k, set())), np.int64)
           for k, v in _mask_by_case(MASK_EVAL).items()}

    def covered(r):
        b = ext.get((int(r["lc_id"]), int(r["case_id"])))
        w = int(r["start_idx"])
        if b is None or not b.size:
            return False
        j = np.searchsorted(b, w)
        return j < b.size and b[j] < w + REPEAT_W

    s1 = rows(SNAPROWS / "stage1" / "test_per_window_stats.csv")
    s1_hit = [r for r in s1 if covered(r)]
    s2 = {(r["lc_id"], r["case_id"], r["start_idx"]): r
          for r in rows(SNAPROWS / "test_per_window_stats.csv")}
    s2_hit = [r for r in s2.values() if covered(r)]
    assert (len(s1), len(s1_hit), len(s2), len(s2_hit)) == (50000, 487, 5000, 47), \
        (len(s1), len(s1_hit), len(s2), len(s2_hit))
    ten1 = sum(1 for r in s1_hit if int(r["lc_id"]) == 10)
    ten2 = sum(1 for r in s2_hit if int(r["lc_id"]) == 10)

    return [
        ("wSub", f"{W_SUB:.1f}",
         f"(rho A - rho_w A) g = ({LINE['rhoA']} - {RHO_W} x {LINE['area']}) x {GRAV} "
         f"= {W_SUB!r} N/m"),
        ("catCleanSims", f"{len(cl)}",
         "simulations where the shape and non-finite rules never fire (scan_cat.pkl)"),
        ("catStaticSlope", f"{slope:.1f}",
         f"least squares over them, per-case median (T_f - T_anchor) against median "
         f"(z_f - z_anchor) = {slope!r} N/m"),
        ("catStaticIcpt", f"{-icpt:.0f}", f"same fit, minus the intercept = {-icpt!r} N"),
        ("catStaticCorr", f"{corr:.4f}", f"correlation of the same = {corr!r}"),
        ("catMarginMed", f"{float(np.median(mmed)):.2f}",
         "median over them of the per-case median of (T_f - T_a)/(w_s h)"),
        ("catMarginPone", f"{margin_p1:.2f}",
         f"1st percentile over them of the per-case minimum of the same = {margin_p1!r}"),
        ("catCleanSamples", f"{c_samp}",
         "samples of those simulations with a station other than the fairlead above 1 kN "
         "and above the fairlead at the same sample (k = 1.0)"),
        ("catCleanHitSims", f"{c_sims}", "simulations holding them"),
        ("catCleanTotalM", f"{c_total / 1e6:.0f}",
         f"archived samples of those simulations, in millions = {c_total}"),
        ("catFlagSamples", f"{f_samp}",
         f"the same count over the {f_inuse} shape/non-finite-flagged simulations in use "
         "(masked or not)"),
        ("catFlagSims", f"{f_inuse}", "flagged simulations still in use"),
        ("catCandBelow", f"{below}",
         "snap candidates (no cat bound, snapcand.py) in clean simulations whose largest "
         "station/fairlead ratio at the sample is in [0.9, 1.0)"),
        ("catCandAbove", f"{above}", "same, ratio in [1.0, 1.1)"),
        ("catMomBelow", f"{mom_lo / 1000:.1f}",
         f"median |axial momentum residual| at snap candidates with ratio < 1, all "
         f"simulations, accelerations from second differences of the archived positions "
         f"= {mom_lo!r} N"),
        ("catMomAbove", f"{mom_hi / 1000:.0f}",
         f"same, ratio >= 1.5 = {mom_hi!r} N; bands 1.0-1.2/1.2-1.5/1.5-3/3+ = "
         + "/".join(f"{b:.0f}" for b in bands) + " N (monotone, asserted)"),
        ("catPoolShare", f"{pool_share:.2f}",
         f"block-aware training/validation candidate windows lost when the evaluation mask "
         f"replaces the training mask, {n_tv['tr']} -> {n_tv['ev']}"),
        ("catEventsCleanOld", f"{ev_old}",
         "snap_catalogue.csv (bound 1.5, no rule C) events in shape-clean simulations"),
        ("catEventsCleanNew", f"{ev_new}", f"{SNAP_CAT.name}, the same"),
        ("catDropSone", f"{len(s1_hit)}",
         "Stage-1 published test windows covering a catenary-rule sample (of 50000)"),
        ("catDropSoneTen", f"{ten1}", "of them at loc10"),
        ("catDropStwo", f"{len(s2_hit)}",
         "Stage-2 distinct published test windows covering one (of 5000; x10 layouts)"),
        ("catDropStwoTen", f"{ten2}", "of them at loc10"),
    ]


# ---------------------------------------------------------------------------------
# The wave-train repeat (Data 2.4), on the per-entry metrics of BOTH shipped models.
#
# Replaces the fairlead-peak-only check (paperA_repeat_check.py, kept as a script). The
# per-window statistics (test_per_window_stats.csv, patch sr-18) carry MAE, MAPE and S per
# window, plus the denominator of S_w, <|dT|>_w, the mean absolute step change of the TRUE
# tension ("activity") -- a property of the ground truth alone.
#   * the affected windows (covering sample >= 12064) DO score better, at both stages;
#   * they are calmer at every one of the eight training sites (the withheld sites are not part
#     of this step: activity involves no model, so they add nothing to it);
#   * at matched activity (deciles within site x layout) it shrinks to under 1 % (MAE -0.5 % S1,
#     -0.7 % S2; 20 bins -0.3/-0.7 %, a log-log fit per stratum -0.0/+0.9 % -- the decile residual
#     is coarse-bin confounding). The withheld-site DiD is NOT used: tight for Stage 1
#     (-0.7 % [-3.5, +2.2]) but inconclusive for Stage 2 (-5.9 % [-12.9, +0.1], loc11 alone +1.0 %).
# Population = the CLEAN test draw (evaluation mask). Read from the EVAL_CLEAN job when it is
# on disk; until then reproduced offline from the published-draw per-window files of job
# 287893 (EVAL_SNAPROWS, which reproduced every published metric) by dropping each window
# that covers an evaluation-mask sample -- the same rows the job writes (sr-22 test T5).
REPEAT_W = 1000
REPEAT_DEV = (1, 3, 4, 5, 6, 7, 8, 9)
REPEAT_DRAWS = (   # tag, sub-directory, pairs the evaluation mask removes (sr-22 references)
    ("Sone", "stage1", 487),
    ("Stwo", "", 470),
)


def _mask_by_case(path):
    return {(int(r["loc"]), int(r["case"])): {int(x) for x in r["bad_t"].split()}
            for r in rows(path)}


@functools.lru_cache(maxsize=None)
def _eval_extra():
    """Evaluation-mask samples that the training mask lacks, per simulation (sorted)."""
    ev, tr = _mask_by_case(MASK_EVAL), _mask_by_case(MASK_TRAIN)
    return {k: np.array(sorted(v - tr.get(k, set())), np.int64) for k, v in ev.items()}


def eval_dropped(lc, case, start):
    """True if the published-draw window [start, start + W) covers an evaluation-mask sample
    the training mask lacks, i.e. one the EVAL_CLEAN job removed (sr-22 semantics)."""
    b = _eval_extra().get((int(lc), int(case)))
    if b is None or not b.size:
        return False
    j = np.searchsorted(b, int(start))
    return bool(j < b.size and b[j] < int(start) + REPEAT_W)


def perwindow_clean(sub, n_drop):
    """Per-window statistics of a shipped model on its clean test draw, and their source."""
    f_job = EVAL_CLEAN / sub / "test_per_window_stats.csv"
    if P.exists(f_job):
        out = rows(f_job)
        assert len(out) == 50000 - n_drop, (f_job, len(out))
        return out, f"EVAL_CLEAN job, {f_job.relative_to(ROOT)}"
    f_pub = SNAPROWS / sub / "test_per_window_stats.csv"
    pub = rows(f_pub)
    assert len(pub) == 50000, (f_pub, len(pub))
    ev = _mask_by_case(ROOT / "artifact_samples_cat.csv")
    tr = _mask_by_case(ROOT / "artifact_samples.csv")
    extra = {k: np.array(sorted(v - tr.get(k, set())), np.int64) for k, v in ev.items()}
    out = []
    for r in pub:
        b = extra.get((int(r["lc_id"]), int(r["case_id"])))
        w = int(r["start_idx"])
        if b is not None and b.size:
            j = np.searchsorted(b, w)
            if j < b.size and b[j] < w + REPEAT_W:
                continue
        out.append(r)
    assert len(pub) - len(out) == n_drop, (f_pub, len(pub) - len(out), n_drop)
    return out, (f"OFFLINE PREVIEW, {f_pub.relative_to(ROOT)} minus the {n_drop} windows "
                 "covering an evaluation-mask sample -- switches to EVAL_CLEAN when it lands")


def _repeat_analysis(rs, n_boot=2000, seed=7):
    """Raw and activity-matched repeat effects, cluster bootstrap over distinct windows."""
    def g(k, t=float):
        return np.array([t(r[k]) for r in rs])
    lc, case, st, nin = g("lc_id", int), g("case_id", int), g("start_idx", int), g("n_in", int)
    st = st.astype(np.int64)
    Y = {"MAE": g("mae"), "MAPE": g("mape"), "S": g("td_skill")}
    act = g("td_mae_flat")
    repeat_idx = int(round(2 * np.pi / (SOLVER["wc"] / (SOLVER["nw"] - 1)) / SOLVER["dt"]))
    rep = np.clip(st + REPEAT_W - repeat_idx, 0, REPEAT_W) > 0
    dev = np.isin(lc, REPEAT_DEV) & np.isfinite(act) & (act > 0)
    # the span of the record the activity comparison is restricted to: the temporal test region
    lo = int(np.percentile(st[dev], 1))

    # activity deciles within each (site, layout) stratum of the development windows
    gid = np.full(len(rs), -1, np.int64)
    for s_ in np.unique((lc * 100 + nin)[dev]):
        ii = np.nonzero(dev & (lc * 100 + nin == s_))[0]
        q = np.quantile(act[ii], np.linspace(0, 1, 11))
        gid[ii] = s_ * 10 + np.clip(np.searchsorted(q, act[ii], side="right") - 1, 0, 9)
    _, gidx = np.unique(gid[gid >= 0], return_inverse=True)
    G = np.zeros(len(rs), np.int64)
    G[gid >= 0] = gidx
    ng = int(gidx.max()) + 1

    def raw(y, sel, w):
        f = sel & np.isfinite(y)
        yy = np.nan_to_num(y)
        return 100 * (((w * yy)[f & rep].sum() / w[f & rep].sum())
                      / ((w * yy)[f & ~rep].sum() / w[f & ~rep].sum()) - 1)

    def matched(y, w):
        f = dev & np.isfinite(y) & (gid >= 0)
        yy = np.nan_to_num(y)
        wt, wc = w * (f & rep), w * (f & ~rep)
        Nt, Nc = np.bincount(G, wt, ng), np.bincount(G, wc, ng)
        St, Sc = np.bincount(G, wt * yy, ng), np.bincount(G, wc * yy, ng)
        k = (Nt >= 3) & (Nc >= 3)
        lr = np.log((St[k] / Nt[k]) / (Sc[k] / Nc[k]))
        return 100 * (np.exp(np.average(lr, weights=Nt[k])) - 1)

    # the activity comparison at each of the eight training sites, over that span of the record.
    # The withheld sites are deliberately NOT part of it: activity involves no model, so they add
    # nothing to the argument, which concerns the windows the training could have seen.
    sites = tuple(REPEAT_DEV)
    span = np.isfinite(act) & (act > 0) & (st >= lo)
    late = {s_: np.nonzero(span & (lc == s_) & rep)[0] for s_ in sites}
    early = {s_: np.nonzero(span & (lc == s_) & ~rep)[0] for s_ in sites}

    def site_gap(s_, w):
        return 100 * (((w[late[s_]] * act[late[s_]]).sum() / w[late[s_]].sum())
                      / ((w[early[s_]] * act[early[s_]]).sum() / w[early[s_]].sum()) - 1)

    one = np.ones(len(rs))
    res = {"share": 100 * rep[dev].sum() / dev.sum(), "n_dev": int(dev.sum()),
           "lo": lo, "act_dev": raw(act, dev, one),
           "site_gap": {s_: site_gap(s_, one) for s_ in sites},
           "site_n": {s_: (late[s_].size, early[s_].size) for s_ in sites}}
    for m, y in Y.items():
        res[f"clean_{m}"] = float(y[dev & ~rep & np.isfinite(y)].mean())
        res[f"rep_{m}"] = float(y[dev & rep & np.isfinite(y)].mean())
        res[f"raw_{m}"] = raw(y, dev, one)
        res[f"match_{m}"] = matched(y, one)

    # cluster bootstrap: a window scored at several layouts (the paired Stage-2 draw) is ONE draw
    key = lc.astype(np.int64) * 100_000_000 + case.astype(np.int64) * 100_000 + st
    _, cid = np.unique(key, return_inverse=True)
    ncl = int(cid.max()) + 1
    res["n_distinct_dev"] = int(np.unique(cid[dev]).size)
    rng = np.random.default_rng(seed)
    boot = collections.defaultdict(list)
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, ncl, ncl), minlength=ncl)[cid].astype(float)
        boot["act_dev"].append(raw(act, dev, w))
        for s_ in sites:
            boot[f"site{s_}"].append(site_gap(s_, w))
        for m, y in Y.items():
            boot[f"raw_{m}"].append(raw(y, dev, w))
            boot[f"match_{m}"].append(matched(y, w))
    res["ci"] = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                 for k, v in boot.items()}
    return res


def build_repeat():
    """The wave-train repeat on per-entry metrics, both stages, clean test draws."""
    R, src = {}, {}
    for tag, sub, n_drop in REPEAT_DRAWS:
        rs, src[tag] = perwindow_clean(sub, n_drop)
        R[tag] = _repeat_analysis(rs)
    a, b = R["Sone"], R["Stwo"]

    def ci(r, k):
        return "95 %% CI [%+.3f, %+.3f] %%" % r["ci"][k]

    out = [
        ("repeatShare", f"{a['share']:.0f}",
         f"Stage-1 clean draw: {a['share']!r} % of {a['n_dev']} development test windows cover "
         f"a sample >= the repeat index (Stage-2 draw {b['share']:.1f} %); {src['Sone']}"),
    ]
    for tag, r in (("Sone", a), ("Stwo", b)):
        out += [
            (f"repeat{tag}MAEClean", f"{r['clean_MAE']:.1f}",
             f"{tag} dev windows NOT covering the repeat, mean per-window MAE = {r['clean_MAE']!r} N"),
            (f"repeat{tag}MAERep", f"{r['rep_MAE']:.1f}",
             f"{tag} dev windows covering the repeat, mean per-window MAE = {r['rep_MAE']!r} N; "
             f"change {r['raw_MAE']:+.2f} % ({ci(r, 'raw_MAE')}); MAPE {r['raw_MAPE']:+.2f} % "
             f"({ci(r, 'raw_MAPE')}), S {r['raw_S']:+.2f} % ({ci(r, 'raw_S')}); "
             f"{r['n_distinct_dev']} distinct windows; source {src[tag]}"),
            (f"repeat{tag}MAEDrop", f"{-r['raw_MAE']:.1f}",
             f"{tag}: reduction [%] of the mean per-window MAE, repeat vs non-repeat dev windows"),
        ]
    # The activity <|dT|>_w (denominator of S_w, true tension only) is lower in the repeat windows
    # at EACH of the eight training sites. The text states it for the Stage-1 draw (about 300
    # simulations per site); the Stage-2 draw has about 40 per site and some of its intervals
    # include zero, so it is reported here and not relied on. The generator refuses to emit the
    # claim if it fails.
    gaps = a["site_gap"]
    site_hi = {s_: a["ci"][f"site{s_}"][1] for s_ in gaps}
    assert all(g < 0 for g in gaps.values()), gaps
    assert all(h < 0 for h in site_hi.values()), site_hi     # every 95 % interval excludes zero
    per_site = "; ".join(f"loc{s_:02d} {gaps[s_]:+.1f} % [{a['ci'][f'site{s_}'][0]:+.1f}, "
                         f"{a['ci'][f'site{s_}'][1]:+.1f}]" for s_ in gaps)
    pooled = (f"pooled over the eight sites {a['act_dev']:+.2f} % ({ci(a, 'act_dev')}); "
              f"Stage-2 draw {b['act_dev']:+.2f} %")
    out += [
        ("repeatActSiteMin", f"{min(-g for g in gaps.values()):.0f}",
         f"smallest reduction [%] of the mean <|dT|>_w in the repeat windows over the eight training sites, "
         f"Stage-1 clean draw, span start >= {a['lo']}; every 95 % cluster-bootstrap interval "
         f"excludes zero (asserted); {per_site}; {pooled}"),
        ("repeatActSiteMax", f"{max(-g for g in gaps.values()):.0f}",
         "largest reduction [%] of the mean <|dT|>_w over the eight training sites, same comparison"),
        ("repeatActPooled", f"{-a['act_dev']:.1f}",
         f"pooled reduction [%] of the mean <|dT|>_w in the repeat windows, magnitude of act_dev "
         f"over the eight training sites, Stage-1 clean draw; {pooled}"),
        ("repeatActPooledLo", f"{-a['ci']['act_dev'][1]:.1f}",
         "pooled activity reduction, 95 % CI bound nearest zero [%], magnitude of the upper "
         "(less negative) raw CI end"),
        ("repeatActPooledHi", f"{-a['ci']['act_dev'][0]:.1f}",
         "pooled activity reduction, 95 % CI bound farthest from zero [%], magnitude of the "
         "lower (more negative) raw CI end"),
        ("repeatSpanStart", f"{a['lo']}",
         "first sample at which a Stage-1 test window may start to enter the per-site activity "
         "comparison: the 1st percentile of the training-site test starts, i.e. the final 30 % of "
         "a full 13801-sample record"),
    ]
    for tag, r in (("Sone", a), ("Stwo", b)):
        out.append((f"repeatMatch{tag}MAE", f"{r['match_MAE']:.1f}",
                    f"{tag}: repeat vs non-repeat windows of the same site, layout and activity "
                    f"decile, MAE ratio - 1 [%] (weighted by repeat windows per group) = "
                    f"{r['match_MAE']!r} ({ci(r, 'match_MAE')}); MAPE {r['match_MAPE']:+.3f} "
                    f"({ci(r, 'match_MAPE')}); S {r['match_S']:+.3f} ({ci(r, 'match_S')})"))
    other = max(abs(r[f"match_{m}"]) for r in (a, b) for m in ("MAPE", "S"))
    out.append(("repeatMatchOther", f"{other:.1f}",
                "largest |activity-matched difference| [%] of MAPE and S over both stages"))
    mx = max(abs(r[f"match_{m}"]) for r in (a, b) for m in ("MAE", "MAPE", "S"))
    out.append(("repeatMatchMax", f"{mx:.1f}",
                "largest |activity-matched difference| [%] over MAE, MAPE and S, both stages"))
    lows = {f"{t} {m}": r["ci"][f"match_{m}"][0] for t, r in (("Sone", a), ("Stwo", b))
            for m in ("MAE", "MAPE", "S")}
    worst = min(lows, key=lows.get)
    out.append(("repeatEquivBound", f"{max(0.0, -lows[worst]):.1f}",
                "largest advantage [%] not excluded at 95 % by ANY activity-matched CI, both "
                "stages x (MAE, MAPE, S): lower CI ends "
                + ", ".join(f"{k} {v:+.2f}" for k, v in lows.items())
                + f"; binding: {worst}"))
    return out


@functools.lru_cache(maxsize=None)
@derived
def pinball_exceedance():
    """Share of per-entry tension above the pinball threshold, by site (Method 3.5, Results 4.1.1).

    The eighth loss term acts only on entries whose TRUE tension exceeds peak_pinball_thr_n, one
    absolute threshold for every site. Training sites: the first 70 % of each record (the
    train+val region the term is trained on); loc10, which has no training data: the whole record.
    The 21 native stations, minus the artifact mask the models were trained with (MASK_TRAIN),
    with the simulations dropped whole (non-finite, or shorter than one window) skipped, as in the
    pipeline. Also returns the pooled training share, whose complement is the percentile
    of the threshold (Method 3.5 says 85th; asserted).
    """
    import json
    cfg = json.loads(P.read_text(S1 / "config.json"))
    thr, win = float(cfg["peak_pinball_thr_n"]), int(cfg["window_len"])
    mask, dropped = {}, set()
    for r in rows(MASK_TRAIN):
        key = (int(r["loc"]), int(r["case"]))
        mask[key] = [int(t) for t in r["bad_t"].split()]
        if int(r["n_nonfinite"]) > 0:
            dropped.add(key)
    share, n_all, n_above = {}, 0, 0
    for loc in DEV_SITES + (10,):
        above = total = 0
        for cdir in sorted((CACHE / f"loc{loc:02d}").glob("case_*")):
            key = (loc, int(cdir.name.split("_")[1]))
            f = cdir / "tension.npy"
            if key in dropped or not f.exists():
                continue
            T = npload(f, mmap_mode="r")
            if T.shape[0] < win:
                continue
            cut = int(0.7 * T.shape[0]) if loc in DEV_SITES else T.shape[0]
            X = np.asarray(T[:cut], dtype=np.float64)
            bad = [b for b in mask.get(key, []) if b < cut]
            if bad:
                keep = np.ones(cut, dtype=bool)
                keep[bad] = False
                X = X[keep]
            assert np.isfinite(X).all(), key
            above += int((X > thr).sum())
            total += X.size
        share[loc] = 100.0 * above / total
        if loc in DEV_SITES:
            n_all += total
            n_above += above
    pooled = 100.0 * n_above / n_all
    assert abs(100.0 - pooled - 85.0) < 0.1, pooled          # \pinballPct{}
    return dict(thr=thr, site=share, pooled=pooled, n=n_all)


def build_sites():
    """Per-site transfer: the two withheld locations, as a fairlead-peak level offset."""
    out = []
    for tag, path in (("Sone", T1), ("Stwo", T2)):
        z = npload(path / "test_timeseries_dump.npz")
        by_loc = collections.defaultdict(list)
        for m, a, b in zip(z["meta_all"], z["peak_true_all"], z["peak_pred_all"]):
            if a > 0:
                by_loc[int(m[0])].append(b / a)
        for lc, name in ((10, "Ten"), (11, "Eleven")):
            md = statistics.median(by_loc[lc])
            pct = 100 * (md - 1)
            out.append((f"loc{name}Offset{tag}", f"{abs(pct):.1f}",
                        f"{path.name} dump, loc{lc} median fairlead peak pred/true = "
                        f"{md!r} ({pct:+.2f} %), magnitude only -- state the sign in prose"))

    # The peak (pinball) term acts only on entries above one absolute threshold; at the two
    # shallow sites almost none exceed it (Results 4.1.1).
    sh = pinball_exceedance()
    others = {k: v for k, v in sh["site"].items() if k in DEV_SITES and k != 6}
    assert sh["site"][6] < 0.1 and sh["site"][10] < 0.2 and min(others.values()) > 5.0, sh
    src = (f"cache_npy tension, share of per-entry values above {sh['thr']:.0f} N (the pinball "
           f"threshold); training sites over the first 70 % of the record, loc10 over the whole "
           f"record; pooled training share {sh['pooled']:.2f} %")
    out += [
        ("pinShareSix", f"{sh['site'][6]:.2f}", f"loc06 (the only shallow training site) = "
                                                f"{sh['site'][6]!r} %; {src}"),
        ("pinShareTen", f"{sh['site'][10]:.2f}", f"loc10 = {sh['site'][10]!r} %; {src}"),
        ("pinShareTrainLo", f"{min(others.values()):.0f}",
         "smallest share [%] over the other seven training sites: "
         + ", ".join(f"loc{k:02d} {v:.2f}" for k, v in others.items())),
        ("pinShareTrainHi", f"{max(others.values()):.0f}", "largest share [%] of the same seven sites"),
    ]
    return out


def build_architecture():
    """Section 3 -- the model as built and trained.

    Every hyperparameter is read from the shipped config.json of the run that
    produced the corresponding model, and every parameter count is counted from
    that run's own checkpoint state_dict, so the prose cannot drift from the
    artefact. Reading a checkpoint is a plain torch.load onto the CPU (~68 MB),
    not a notebook run.
    """
    import json
    import torch

    c1 = json.loads(P.read_text(S1 / "config.json"))
    c2 = json.loads(P.read_text(S2 / "config.json"))

    # The two runs share every architecture and optimiser setting; the query
    # decoder and the batch split are the only intended differences. Assert it
    # rather than trust it, because Section 3 presents them as one architecture.
    for k in ("gat_hidden_dim", "gat_out_dim", "num_heads", "lstm_hidden_dim",
              "num_lstm_layers", "lstm_bidirectional", "lstm_dropout", "head_hidden_dim",
              "head_dropout", "window_len", "learning_rate", "weight_decay", "grad_clip_max_norm",
              "scheduler_factor", "scheduler_patience", "scheduler_min_lr",
              "peak_pinball_tau", "peak_pinball_thr_n", "use_global_context",
              "sel_r2_delta", "node_counts"):
        assert c1[k] == c2[k], (k, c1[k], c2[k])
    assert c1["use_query_decoder"] is False and c2["use_query_decoder"] is True
    assert (c1["batch_size"] * c1["accumulation_steps"]
            == c2["batch_size"] * c2["accumulation_steps"])

    # Feature widths, derived from the column-index lists the runs actually used.
    n_dyn_node = len(c1["node_continuous_idx"]) + len(c1["node_binary_idx"])
    n_dyn_edge = len(c1["edge_continuous_idx"]) + len(c1["edge_binary_idx"])
    n_stat_node, n_stat_edge = 12, 10   # build_mooring_graph() docstring, fixed layout
    assert n_dyn_node == 22 and n_dyn_edge == 4

    counts = {}
    for tag, path in (("One", S1), ("Two", S2)):
        ck = torch.load(P.src(path / "best_mooring_gat_lstm_seed_42.pt"),
                        map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        counts[tag] = dict(
            epoch=int(ck["epoch"]),
            total=sum(v.numel() for v in sd.values()),
            decoder=sum(v.numel() for k, v in sd.items()
                        if k.startswith("query_decoder.")),
            globalmlp=sum(v.numel() for k, v in sd.items() if "global_mlp" in k),
            # the target standardiser the loss was trained with (fitted on training windows)
            t_mean=float(ck["criterion_state_dict"]["t_mean"]),
            t_std=float(ck["criterion_state_dict"]["t_std"]),
        )
        del ck
    assert counts["Two"]["total"] - counts["One"]["total"] == counts["Two"]["decoder"]
    # Stage 2 refits its standardiser; it must agree with Stage 1's at the precision printed
    for k in ("t_mean", "t_std"):
        assert (f"{counts['One'][k] / 1000:.2f}" == f"{counts['Two'][k] / 1000:.2f}"), \
            (k, counts["One"][k], counts["Two"][k])

    # Actual epochs completed, from the training log -- distinct from config num_epochs,
    # which is only the cap. Stage 1 stopped early (best epoch + patience); Stage 2 ran to
    # its configured cap without early stopping firing, so the two coincide there.
    s1_last_epoch = int(rows(S1 / "metrics_history.csv")[-1]["epoch"])
    s2_last_epoch = int(rows(S2 / "metrics_history.csv")[-1]["epoch"])
    assert c1["early_stopping_patience"] == c2["early_stopping_patience"]
    assert s1_last_epoch == counts["One"]["epoch"] + c1["early_stopping_patience"]
    assert s2_last_epoch == c2["num_epochs"]

    def grp(n):
        return f"{n:,}".replace(",", "\\,")

    heads, hid = c1["num_heads"], c1["gat_hidden_dim"]
    lstm_out = c1["lstm_hidden_dim"] * (2 if c1["lstm_bidirectional"] else 1)
    gc_pct = 100 * counts["One"]["globalmlp"] / counts["One"]["total"]

    return [
        # ----- feature widths
        ("nStatNodeFeat", f"{n_stat_node}",
         "build_mooring_graph(): static node matrix [N x 12]"),
        ("nDynNodeFeat", f"{n_dyn_node}",
         "config node_continuous_idx + node_binary_idx"),
        ("nNodeFeat", f"{n_stat_node + n_dyn_node}", "concatenated node input width"),
        ("nStatEdgeFeat", f"{n_stat_edge}",
         "build_mooring_graph(): static edge matrix [E x 10]"),
        ("nDynEdgeFeat", f"{n_dyn_edge}",
         "config edge_continuous_idx + edge_binary_idx"),
        ("nEdgeFeat", f"{n_stat_edge + n_dyn_edge}", "concatenated edge input width"),
        ("nEnvChannels", "13",
         "h0, Hs, Tp and the five-point current profile d1..5, v1..5"),
        ("windowSamples", f"{c1['window_len']}", "config window_len [samples]"),
        # ----- spatial encoder
        ("gatLayers", "2", "MooringGATEncoder: gat1 + gat2, fixed"),
        ("gatHeads", f"{heads}",
         "config num_heads (layer 1; layer 2 is single-head)"),
        ("gatHidden", f"{hid}", "config gat_hidden_dim, per head"),
        ("gatConcat", f"{hid * heads}",
         f"concatenated layer-1 width = {hid} x {heads}"),
        ("gatOut", f"{c1['gat_out_dim']}", "config gat_out_dim"),
        ("gatDropout", f"{c1['gat_dropout']:g}", "config gat_dropout"),
        # ----- temporal encoder
        ("lstmLayers", f"{c1['num_lstm_layers']}", "config num_lstm_layers"),
        ("lstmHidden", f"{c1['lstm_hidden_dim']}",
         "config lstm_hidden_dim, per direction"),
        ("lstmOut", f"{lstm_out}",
         f"bidirectional: {c1['lstm_hidden_dim']} x 2 per step"),
        ("lstmDropout", f"{c1['lstm_dropout']:g}",
         "config lstm_dropout, between the stacked layers and after the layer norm"),
        # ----- head and query decoder
        ("headHidden", f"{c1['head_hidden_dim']}", "config head_hidden_dim"),
        ("headDropout", f"{c1['head_dropout']:g}",
         "config head_dropout, between the two layers of the head"),
        ("decDim", f"{c2['gat_out_dim']}", "decoder d_model = gat_out_dim"),
        ("decHeads", f"{c2['query_decoder_heads']}", "config query_decoder_heads"),
        ("decFFN", f"{c2['query_decoder_ffn_dim']}", "config query_decoder_ffn_dim"),
        ("decBands", f"{c2['query_decoder_fourier_bands']}",
         "config query_decoder_fourier_bands"),
        ("decDropout", f"{c2['query_decoder_dropout']:g}",
         "config query_decoder_dropout, on the attention weights and inside the feed-forward block"),
        # ----- parameter counts, from the shipped checkpoints
        ("paramsSone", grp(counts["One"]["total"]),
         f"S1 checkpoint state_dict, ep{counts['One']['epoch']} = "
         f"{counts['One']['total']}"),
        ("paramsStwo", grp(counts["Two"]["total"]),
         f"S2 checkpoint state_dict, ep{counts['Two']['epoch']} = "
         f"{counts['Two']['total']}"),
        ("paramsDecoder", grp(counts["Two"]["decoder"]),
         f"S2 checkpoint, query_decoder.* = {counts['Two']['decoder']}"),
        ("paramsGlobal", grp(counts["One"]["globalmlp"]),
         f"S1 checkpoint, spatial_encoder.global_mlp.* = "
         f"{counts['One']['globalmlp']}"),
        ("paramsGlobalPct", f"{gc_pct:.1f}",
         f"global readout as a share of the Stage-1 model = {gc_pct!r} %"),
        # ----- optimisation
        ("optLr", "1 \\times 10^{-3}", f"config learning_rate = {c1['learning_rate']}"),
        ("optWd", "1 \\times 10^{-4}", f"config weight_decay = {c1['weight_decay']}"),
        ("optClip", f"{c1['grad_clip_max_norm']:g}", "config grad_clip_max_norm"),
        ("optBatch", f"{c1['batch_size'] * c1['accumulation_steps']}",
         f"effective batch: S1 {c1['batch_size']}x{c1['accumulation_steps']}, "
         f"S2 {c2['batch_size']}x{c2['accumulation_steps']}"),
        ("schedFactor", f"{c1['scheduler_factor']:g}", "config scheduler_factor"),
        ("schedPatience", f"{c1['scheduler_patience']}", "config scheduler_patience"),
        ("schedFloor", "5 \\times 10^{-5}",
         f"config scheduler_min_lr = {c1['scheduler_min_lr']}"),
        # ----- evaluation metrics
        ("mapeThrN", "1000",
         "compute_physical_metrics_per_target() default mape_min=1000.0 N, "
         "the threshold-gated MAPE's near-slack exclusion"),
        # ----- loss and selection
        ("nLossTerms", "8", "ReconstructionKendallLoss: 8 Kendall-weighted terms"),
        ("pinballTau", f"{c1['peak_pinball_tau']:g}", "config peak_pinball_tau"),
        ("pinballThr", f"{c1['peak_pinball_thr_n'] / 1000:.2f}",
         f"config peak_pinball_thr_n = {c1['peak_pinball_thr_n']} N, the 85th "
         f"percentile of per-entry training tension"),
        ("pinballPct", "85",
         "percentile of per-entry training tension at the pinball threshold"),
        ("tensionMean", f"{counts['One']['t_mean'] / 1000:.2f}",
         f"criterion_state_dict t_mean of the shipped checkpoints [kN]: Stage 1 "
         f"{counts['One']['t_mean']!r} N, Stage 2 {counts['Two']['t_mean']!r} N"),
        ("tensionStd", f"{counts['One']['t_std'] / 1000:.2f}",
         f"criterion_state_dict t_std of the shipped checkpoints [kN]: Stage 1 "
         f"{counts['One']['t_std']!r} N, Stage 2 {counts['Two']['t_std']!r} N; "
         f"the sigma_T of Method 3.5"),
        ("selRsqDelta", f"{c1['sel_r2_delta']:.4f}",
         "config sel_r2_delta, the Gate-A accuracy floor"),
        ("selUpCap", f"{c1['sel_up90_cap']:g}",
         f"config sel_up90_cap, Stage 1 only; Stage 2 sets it to "
         f"{c2['sel_up90_cap']}"),
        ("sOneEpochsMax", f"{c1['num_epochs']}",
         "config num_epochs, Stage 1 -- the cap, not the epochs actually run"),
        ("sOneEpochsRun", f"{s1_last_epoch}",
         "metrics_history.csv row count, Stage 1 -- early stopping fired here"),
        ("sOneSelEpoch", f"{counts['One']['epoch']}",
         "epoch recorded in the shipped S1 checkpoint"),
        ("sTwoEpochs", f"{c2['num_epochs']}",
         "config num_epochs, Stage 2 -- ran to this cap, early stopping did not fire"),
        ("sTwoSelEpoch", f"{counts['Two']['epoch']}",
         "epoch recorded in the shipped S2 checkpoint"),
        ("selEarlyStopPatience", f"{c1['early_stopping_patience']}",
         "config early_stopping_patience, shared by both stages"),
    ]


# =============================================================== Section 4 (Results)
# Added 2026-09-18 for the Results-and-discussion chapter (Paper_A_Results_Plan.md).
# The builders below emit prose macros; the table_* functions further down emit the
# tabular bodies that sections/06_results.tex \input's from tables/.

# The noise levels were re-run on the clean Stage-2 draw by the EVAL_CLEAN job, with the level ids of
# job 236311, so every surviving window carries the realisation it had there. Same file names.
SN = T2
HPO = ROOT / "Results" / "Results_Paper_HPO"
TABLES_DIR = P.OUT_DIR / "tables"
DEV_SITES = (1, 3, 4, 5, 6, 7, 8, 9)
LAYOUTS = (4, 5, 6, 7, 8, 10, 12, 15, 18, 21)


def undermass(mae, bias):
    """Expected shortfall per tail entry, (MAE - bias) / 2 (Method, eq. metrics_tail)."""
    return (mae - bias) / 2.0


@functools.lru_cache(maxsize=None)
@derived
def site_geometry():
    """Water depth h0 and unstretched length L0 per site, from the npy cache."""
    geo = {}
    for d in sorted(CACHE.iterdir()):
        lc = int(d.name[3:])
        c0 = sorted(d.iterdir())[0]
        h0 = float(npload(c0 / "env.npy")[0])
        ref = npload(c0 / "reference.npy", mmap_mode="r")
        geo[lc] = (h0, round(float(ref[0, 1] - ref[0, 0]) * 20))
    return geo


def snap_events(z, n_in=None):
    """Collapse snap rows on the distinct event (loc, case, t).

    Returns {event: (median true, median pred, median of the per-row pred/true)}.
    One physical event is drawn by many windows (and layouts), so per-row statistics
    over-weight high-multiplicity events (CLAUDE.md 13.3); everything is per event.
    """
    g = collections.defaultdict(list)
    for (lc, case, nin, _s, t), a, b in zip(z["snap_ev"], z["snap_true"], z["snap_pred"]):
        if a <= 0 or (n_in is not None and int(nin) != n_in):
            continue
        g[(int(lc), int(case), int(t))].append((float(a), float(b)))
    return {k: (statistics.median(x for x, _ in v), statistics.median(y for _, y in v),
                statistics.median(y / x for x, y in v)) for k, v in g.items()}


def snap_stats(ev):
    keys = sorted(ev)
    T = np.array([ev[k][0] for k in keys])
    P = np.array([ev[k][1] for k in keys])
    R = np.array([ev[k][2] for k in keys])

    def rank(a):
        return np.argsort(np.argsort(a))

    q1, q3 = np.quantile(R, [0.25, 0.75])
    return dict(n=len(keys), median=float(np.median(R)), iqr=float(q3 - q1),
                corr=float(np.corrcoef(T, P)[0, 1]),
                spearman=float(np.corrcoef(rank(T), rank(P))[0, 1]),
                slope=float(np.polyfit(T, P, 1)[0]),
                atleast=100.0 * float(np.mean(R >= 1.0)),
                m50=float(np.quantile(1.0 / R, 0.50)),
                m75=float(np.quantile(1.0 / R, 0.75)),
                m90=float(np.quantile(1.0 / R, 0.90)))


@functools.lru_cache(maxsize=None)
def snap_comparison():
    """Stage 1 on all its events, and the three models on the events common to both draws."""
    ev1 = snap_events(npload(T1 / "test_timeseries_dump.npz"))
    z2 = npload(T2 / "test_timeseries_dump.npz")
    ev21, ev4 = snap_events(z2, N_OUT), snap_events(z2, 4)
    common = set(ev1) & set(ev4) & set(ev21)
    sub = lambda ev: {k: ev[k] for k in common}                          # noqa: E731
    return ev1, dict(all1=snap_stats(ev1), c1=snap_stats(sub(ev1)),
                     c21=snap_stats(sub(ev21)), c4=snap_stats(sub(ev4)))


def tail_split(z):
    """Fairlead-peak tail split by snap membership, p90 and p99 of TRUE peak."""
    meta, pt, pp = z["meta_all"], z["peak_true_all"], z["peak_pred_all"]
    keys = {(int(a), int(b), int(c), int(d)) for a, b, c, d, _t in z["snap_ev"]}
    is_snap = np.array([(int(m[0]), int(m[1]), int(m[2]), int(m[3])) in keys for m in meta])
    e = pp - pt
    out = {}
    for q in (90, 99):
        thr = float(np.quantile(pt, q / 100.0))
        tail = pt >= thr
        for pop, mask in (("all", tail), ("nonsnap", tail & ~is_snap), ("snap", tail & is_snap)):
            s = e[mask]
            out[(q, pop)] = dict(thr=thr, n=int(mask.sum()), share=100.0 * mask.sum() / tail.sum(),
                                 up=100.0 * float((s < 0).mean()), bias=float(s.mean()),
                                 um=undermass(float(np.abs(s).mean()), float(s.mean())))
    return out


@functools.lru_cache(maxsize=None)
def site_rows():
    """Per-site accuracy on the fairlead peak, identical statistics for both stages.

    Stage-1 dumps hold per-window PEAK scalars only, so both stages are compared on the
    fairlead peak, each pooled over its ten sensor layouts. The Stage-2 field metrics `f2`
    (noise job, clean level, N_in = 4) are kept only as the join-integrity control below.
    """
    geo = site_geometry()
    z1 = npload(T1 / "test_timeseries_dump.npz")
    m1, t1, p1 = z1["meta_all"], z1["peak_true_all"], z1["peak_pred_all"]
    z2 = npload(T2 / "test_timeseries_dump.npz")
    m2, t2, p2 = z2["meta_all"], z2["peak_true_all"], z2["peak_pred_all"]

    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows(SN / "test_per_window_stats.csv"):
        if int(r["n_in"]) != 4:
            continue
        lc = int(r["lc_id"])
        n = float(r["n_pts"])
        for g in (lc, "dev" if lc in DEV_SITES else None, "all"):
            if g is None:
                continue
            a = acc[g]
            a["n"] += n; a["ss"] += float(r["ss_res"]); a["ys"] += float(r["y_sum"])
            a["yq"] += float(r["y_sq"]); a["ae"] += float(r["mae"]) * n
            a["w"] += 1; a["ape"] += float(r["mape"])

    def field(g):
        a = acc[g]
        sst = a["yq"] - a["ys"] ** 2 / a["n"]
        # MAE, MAPE (and the skill score) are per-window values averaged over windows,
        # which is how evaluate() reports them; R2 is pooled from sums over the split.
        return dict(mae=a["ae"] / a["n"], r2=1.0 - a["ss"] / sst, mape=a["ape"] / a["w"])

    # The rebuild must reproduce the grid cell it came from, or the join is wrong.
    g = rows(T2 / "superres_grid.csv")
    f_all = field("all")
    for key, col in (("mae", "MAE_tension"), ("r2", "global_R2_tension"), ("mape", "MAPE_tension")):
        ref = grid_cell(g, 4, N_OUT, col)
        assert abs(f_all[key] - ref) <= 1e-6 * max(1.0, abs(ref)), (key, f_all[key], ref)

    def peak_stats(t, p, sel):
        """Fairlead-peak statistics of Eq. (metrics_fairlead): identical for both stages."""
        e = p[sel] - t[sel]
        assert t[sel].min() > 1000.0, "a fairlead peak below T_min = mapeThrN"   # gate never binds
        return dict(ratio=float(np.median(p[sel] / t[sel])),
                    relerr=100.0 * float(np.mean(np.abs(e) / t[sel])),
                    up=100.0 * float((e < 0).mean()), n=int(sel.sum()))

    out = []
    for lc in list(DEV_SITES) + ["dev", 10, 11]:
        if lc == "dev":
            sel1 = np.isin(m1[:, 0], DEV_SITES)
            sel2 = np.isin(m2[:, 0], DEV_SITES)
            hl = None
        else:
            sel1 = m1[:, 0] == lc
            sel2 = m2[:, 0] == lc
            hl = geo[lc][0] / geo[lc][1]
        # both stages pooled over the ten sensor layouts, so the columns are comparable
        out.append(dict(site=lc, hl=hl, h0=None if lc == "dev" else geo[lc][0],
                        L0=None if lc == "dev" else geo[lc][1],
                        s1=peak_stats(t1, p1, sel1), s2=peak_stats(t2, p2, sel2),
                        f2=field(lc)))
    return out


@functools.lru_cache(maxsize=None)
@derived
def gc_ablation():
    """Global-context readout off vs on: one-variable HPO bake-off, VALIDATION set.

    Both arms: 60 epochs, same budget, seed and pipeline; only use_global_context
    differs. Reported as the mean over the last ten epochs, to damp epoch noise.
    """
    out = {}
    for arm in ("GCoff", "GCon"):
        rs = rows(HPO / f"checkpoints_P1_HPO_{arm}" / "metrics_history.csv")
        assert len(rs) == 60, (arm, len(rs))
        last = rs[-10:]
        out[arm] = {k: statistics.mean(float(r[f"val_{k}"]) for r in last) for k in (
            "contact_grounded_R2_median", "contact_grounded_R2_fracneg",
            "contact_touchdown_R2_median", "contact_suspended_R2_median",
            "global_R2_tension", "MAE_tension", "temporal_diff_skill_tension")}
    return out


def build_results_matched():
    """Section 4.1 prose macros: layouts, sites, regions, ablation, extremes, snaps."""
    per_n = {int(r["node_count"]): r for r in rows(T1 / "test_metrics_by_node_count.csv")}
    f = lambda n, k: float(per_n[n][k])                                  # noqa: E731
    r2 = [f(n, "global_R2_tension") for n in LAYOUTS]
    sk = [f(n, "temporal_diff_skill_tension") for n in LAYOUTS]
    mae = [f(n, "MAE_tension") for n in LAYOUTS]
    pooled = kv(T1 / "test_metrics.csv")
    um99 = {n: undermass(f(n, "peak_tension_MAE_p99"), f(n, "peak_tension_bias_p99"))
            for n in LAYOUTS}
    um99_pooled = undermass(float(pooled["test_peak_tension_MAE_p99"]),
                            float(pooled["test_peak_tension_bias_p99"]))
    # Results 4.1.1: "u99 is higher at four stations than at fifteen, yet the p99 under-mass is
    # lower: the misses of the four-station layout are more frequent but smaller"
    assert f(4, "peak_tension_underpred_rate_p99") > f(15, "peak_tension_underpred_rate_p99")
    assert um99[4] < um99[15], (um99[4], um99[15])
    assert f(4, "peak_tension_undermag_p99") < f(15, "peak_tension_undermag_p99")
    out = [
        ("resRsqMin", f"{min(r2):.4f}", f"S1 per-layout global_R2_tension min = {min(r2)!r}"),
        ("resRsqMax", f"{max(r2):.4f}", f"same, max = {max(r2)!r}"),
        ("resSkillMin", f"{min(sk):.2f}", f"S1 per-layout temporal_diff_skill min = {min(sk)!r}"),
        ("resSkillMax", f"{max(sk):.2f}", f"same, max = {max(sk)!r}"),
        ("resMAEFour", f"{f(4, 'MAE_tension'):.1f}", "S1 N=4 MAE_tension"),
        ("resMAEFine", f"{min(mae[5:]):.1f}", "S1 smallest MAE over N >= 10"),
        ("resMAEFineHi", f"{max(mae[5:]):.1f}", "S1 largest MAE over N >= 10"),
        ("resUpNNFour", f"{f(4, 'peak_tension_underpred_rate_p99'):.1f}",
         "S1 N=4 peak_tension_underpred_rate_p99 [%]"),
        ("resUpNNPooled", f"{float(pooled['test_peak_tension_underpred_rate_p99']):.1f}",
         "S1 pooled test_peak_tension_underpred_rate_p99 [%]"),
        ("resUpNNLo", f"{min(f(n, 'peak_tension_underpred_rate_p99') for n in LAYOUTS[1:]):.1f}",
         "S1 smallest per-layout u99 over N >= 5"),
        ("resUpNNHi", f"{max(f(n, 'peak_tension_underpred_rate_p99') for n in LAYOUTS[1:]):.1f}",
         "S1 largest per-layout u99 over N >= 5"),
        ("resUndermagFour", f"{f(4, 'peak_tension_undermag_p99'):,.0f}".replace(",", "\\,"),
         "S1 N=4 peak_tension_undermag_p99 [N], mean miss given a miss"),
        ("resUndermagTwentyOne", f"{f(21, 'peak_tension_undermag_p99'):,.0f}".replace(",", "\\,"),
         "S1 N=21 peak_tension_undermag_p99 [N]"),
        ("resUMNNFour", f"{um99[4]:,.0f}".replace(",", "\\,"), "S1 N=4 p99 under-mass [N]"),
        ("resUpNNFifteen", f"{f(15, 'peak_tension_underpred_rate_p99'):.1f}",
         "S1 N=15 peak_tension_underpred_rate_p99 [%]"),
        ("resUMNNFifteen", f"{um99[15]:,.0f}".replace(",", "\\,"),
         "S1 N=15 p99 under-mass [N]; above the N=4 value although u99 is lower (asserted)"),
        ("resUMNNTwentyOne", f"{um99[21]:,.0f}".replace(",", "\\,"), "S1 N=21 p99 under-mass [N]"),
        ("resUMNNPooled", f"{um99_pooled:,.0f}".replace(",", "\\,"),
         "S1 pooled p99 under-mass [N]; the text contrasts it with N=4 (u99 higher, under-mass "
         "similar, misses smaller: asserted)"),
        ("resThrNNFour", f"{f(4, 'peak_tension_thr_p99') / 1000:.1f}",
         "S1 N=4 peak_tension_thr_p99 [kN], quantile taken within the layout"),
        ("resThrNNPooled", f"{float(pooled['test_peak_tension_thr_p99']) / 1000:.1f}",
         "S1 pooled test_peak_tension_thr_p99 [kN]"),
    ]

    # ---- sites
    for s in site_rows():
        tag = {"dev": "Dev", 6: "Six", 10: "Ten", 11: "Eleven"}.get(s["site"])
        if tag is None:
            continue
        out += [
            (f"site{tag}RatioSone", f"{s['s1']['ratio']:.3f}",
             f"S1 clean dump, site {s['site']}, median fairlead peak pred/true, all layouts"),
            (f"site{tag}ErrSone", f"{s['s1']['relerr']:.2f}",
             "same windows, peak percentage error of the fairlead peak [%]"),
            (f"site{tag}UpSone", f"{s['s1']['up']:.1f}",
             "same windows, share of fairlead peaks underpredicted [%]"),
            (f"site{tag}RatioStwo", f"{s['s2']['ratio']:.3f}",
             f"S2 clean dump, site {s['site']}, median fairlead peak pred/true, all layouts"),
            (f"site{tag}ErrStwo", f"{s['s2']['relerr']:.2f}",
             "same windows, peak percentage error of the fairlead peak [%]"),
            (f"site{tag}UpStwo", f"{s['s2']['up']:.1f}",
             "same windows, share of fairlead peaks underpredicted [%]"),
        ]
    dev_up = [s["s1"]["up"] for s in site_rows() if s["site"] in DEV_SITES and s["site"] != 6]
    out += [("siteDevUpLo", f"{min(dev_up):.1f}", "S1 clean dump, lowest per-site u over training sites except loc06"),
            ("siteDevUpHi", f"{max(dev_up):.1f}", "same, highest")]
    # depth-to-length ratio: loc10's neighbours among the training sites
    hl = {s["site"]: s["hl"] for s in site_rows() if s["hl"] is not None}
    next_hl = min(v for k, v in hl.items() if k in DEV_SITES and k != 6)
    out += [("siteSixHL", f"{hl[6]:.3f}", "h0/L0 of loc06, env.npy col 0 over 20x reference spacing"),
            ("siteTenHL", f"{hl[10]:.3f}", "h0/L0 of loc10"),
            ("siteElevenHL", f"{hl[11]:.3f}", "h0/L0 of loc11"),
            ("siteNextHL", f"{next_hl:.3f}", "smallest h0/L0 among the training sites other than loc06")]

    # ---- regions: n and % negative (the medians already exist as sOne*Rsq / sr*Rsq)
    p2 = pooled_row(rows(T2 / "superres_grid_pooled.csv"), N_OUT)
    b2 = pooled_row(rows(T2 / "superres_grid_pooled_baseline.csv"), N_OUT)
    for regime, tag in (("grounded", "Grounded"), ("touchdown", "Touchdown"),
                        ("suspended", "Suspended")):
        out += [
            (f"sOne{tag}Neg", f"{float(pooled[f'test_contact_{regime}_R2_fracneg']):.1f}",
             f"S1 test_contact_{regime}_R2_fracneg -- already a PERCENT despite the name"),
            (f"sr{tag}Neg", f"{float(p2[f'contact_{regime}_R2_fracneg']):.1f}", "S2 pooled, N_out=21"),
            (f"sr{tag}NegBase", f"{float(b2[f'contact_{regime}_R2_fracneg']):.1f}",
             "S2 pooled baseline, N_out=21"),
        ]
    n_s1 = [int(pooled[f"test_contact_{r}_n"]) for r in ("grounded", "touchdown", "suspended")]
    n_s2 = [int(p2[f"contact_{r}_n"]) for r in ("grounded", "touchdown", "suspended")]
    out += [("sOneTouchdownShare", f"{100 * n_s1[1] / sum(n_s1):.1f}",
             f"S1 touchdown node-windows {n_s1[1]} of {sum(n_s1)} [%]"),
            ("srTouchdownShare", f"{100 * n_s2[1] / sum(n_s2):.1f}",
             f"S2 touchdown node-windows {n_s2[1]} of {sum(n_s2)} [%]; labels come from the "
             "input contact flag interpolated to the output stations")]

    # ---- global-context ablation (validation, HPO budget)
    gc = gc_ablation()
    for arm, tag in (("GCoff", "Off"), ("GCon", "On")):
        a = gc[arm]
        out += [
            (f"gc{tag}Grounded", f"{a['contact_grounded_R2_median']:.2f}",
             f"HPO {arm} metrics_history.csv, mean of the last 10 of 60 epochs, "
             f"val grounded median R2 = {a['contact_grounded_R2_median']!r}"),
            (f"gc{tag}GroundedNeg", f"{a['contact_grounded_R2_fracneg']:.0f}",
             "same, share of grounded node-windows with R2 < 0 [%]"),
            (f"gc{tag}Touchdown", f"{a['contact_touchdown_R2_median']:.3f}", "same, touchdown"),
            (f"gc{tag}Suspended", f"{a['contact_suspended_R2_median']:.3f}", "same, suspended"),
            (f"gc{tag}MAE", f"{a['MAE_tension']:.0f}", "same, val MAE_tension [N]"),
        ]

    # ---- extremes: fairlead-peak tail split with under-mass
    ts = tail_split(npload(T1 / "test_timeseries_dump.npz"))
    for q, band in ((90, "Ninety"), (99, "NinetyNine")):
        for pop, tag in (("nonsnap", "NonSnap"), ("snap", "Snap"), ("all", "All")):
            out.append((f"sOne{tag}UM{band}", f"{ts[(q, pop)]['um']:,.0f}".replace(",", "\\,"),
                         f"S1 clean dump, p{q} fairlead tail, {pop}, under-mass (MAE-bias)/2 [N], "
                         f"n = {ts[(q, pop)]['n']}"))
        out.append((f"sOneAllUp{band}", f"{ts[(q, 'all')]['up']:.1f}",
                     f"S1 clean dump, p{q} fairlead tail, pooled underprediction rate [%]"))

    # ---- snaps: per-event statistics, by site, by rise, by magnitude, along the line
    ev1, cmp_ = snap_comparison()
    a1 = cmp_["all1"]
    out += [
        ("snapNEvents", f"{a1['n']}", "S1 clean dump, distinct snap events in the Stage-1 test windows"),
        ("snapIQR", f"{a1['iqr']:.3f}", "per-event capture, interquartile range"),
        ("snapCorr", f"{a1['corr']:.3f}", "per-event Pearson correlation, true vs predicted peak"),
        ("snapSlope", f"{a1['slope']:.2f}", "OLS slope of predicted on true snap peak"),
        ("snapShrinkPct", f"{100 * (1 - a1['slope']):.0f}", "1 - OLS slope, as a percentage"),
        ("snapAtLeastPct", f"{a1['atleast']:.0f}", "share of events captured at >= the truth [%]"),
        ("snapMultFifty", f"{a1['m50']:.2f}", "multiplier covering 50 % of events"),
        ("snapMultSeventyFive", f"{a1['m75']:.2f}", "multiplier covering 75 % of events"),
        ("snapMultNinety", f"{a1['m90']:.2f}", "multiplier covering 90 % of events"),
    ]
    cat = {(int(r["loc"]), int(r["case"]), int(r["t"])): r
           for r in rows(SNAP_CAT)}
    by = collections.defaultdict(list)
    for k, (_t, _p, r) in ev1.items():
        by["dev" if k[0] in DEV_SITES else k[0]].append(r)
        by[f"rise{min(int(cat[k]['rise']), 2)}"].append(r)
    for g, tag in (("dev", "Dev"), (10, "Ten"), (11, "Eleven"), ("rise1", "RiseOne"),
                   ("rise2", "RiseTwo")):
        out += [(f"snapCap{tag}", f"{statistics.median(by[g]):.3f}",
                 f"S1 clean dump, per-event median capture, group {g}"),
                (f"snapN{tag}", f"{len(by[g])}", f"events in group {g}")]
    # What the capture statistic measures: snap_true/snap_pred are the LINE MAXIMUM at the
    # event sample. The snap rule of Data 2.5 (no station other than the fairlead above
    # 1 kN and the fairlead) makes the fairlead the highest station at every catalogued
    # event, so that maximum is the FAIRLEAD tension and a snap never sets it (the only
    # catalogued events at or above the fairlead ARE at the fairlead). Asserted, not assumed.
    below = [k for k in ev1 if float(cat[k]["peak"]) < float(cat[k]["fairlead"])]
    assert all(int(cat[k]["node"]) == 20 for k in ev1 if k not in below), "snap above the fairlead"
    ratio_node = [float(cat[k]["peak"]) / ev1[k][0] for k in ev1]
    out += [
        ("snapFairSetPct", f"{100 * len(below) / len(ev1):.0f}",
         f"{len(below)} of {len(ev1)} Stage-1 test events: catalogued snap peak below the "
         "concurrent fairlead tension; the rest peak AT the fairlead"),
        ("snapNodeOverMaxPct", f"{100 * statistics.median(ratio_node):.0f}",
         "median of catalogued snap peak / true line maximum at the event [%]"),
    ]
    # The snap's OWN station exists at matched resolution only when the layout has an
    # output station at the catalogued node; the other events are reported as not
    # available (Table 11). Availability depends only on snap_ev and the catalogue.
    n_av, n_all = snap_station_availability(npload(T1 / "test_timeseries_dump.npz"), stage2=False)
    out += [("snapStationAvailSone", f"{n_av}",
             f"Stage-1 events whose layout has an output station at the snap (of {n_all})"),
            ("snapStationNASone", f"{n_all - n_av}",
             "Stage-1 events without one: reported as not available")]
    # Figures 6 (Stage 1) and 9 (Stage 2): the named illustrative panels, and the size of
    # the pool of qualifying windows they were drawn from. These describe the PANELS only.
    # The capture distribution is the full-test-set one below, reported in Table 11.
    import paperA_figures
    # Spelled out, because these are read as prose ("in all six", "nine of the sixteen"),
    # which is how the rest of the manuscript writes counts below twenty.
    _words = ("zero one two three four five six seven eight nine ten eleven twelve "
              "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()
    for tag, s2 in (("", False), ("Two", True)):
        panels, n_pool = paperA_figures.snap_panels(s2)
        if s2:                       # Figure 9 is four sensors only: count that pool
            n_pool = len({(k[0], k[1]) for k in paperA_figures.snap_panels.pool if k[2] == 4})
        cap = [r["capture"] for r in panels]
        stage = "Stage 2" if s2 else "Stage 1"
        out += [(f"snapStation{tag}N", _words[len(panels)],
                 f"{stage} snap-station panels shown"),
                (f"snapStation{tag}PoolN", _words[n_pool],
                 f"{stage} simulations with a qualifying snap-station window" + (" at four sensors" if s2 else "")),
                (f"snapStation{tag}Median", f"{statistics.median(cap):.2f}",
                 "median capture over those panels (NOT the test-set median)"),
                (f"snapStation{tag}Lo", f"{min(cap):.2f}", "smallest"),
                (f"snapStation{tag}Hi", f"{max(cap):.2f}", "largest")]
    # The Fig. 9 pool at four sensors is loc11 only under the evaluation mask and catalogue
    # (paperA_figures, the Figure 9 comment); no loc10 event qualifies, so none is reported.
    panels2, _n = paperA_figures.snap_panels(True)
    assert all(k[0] == 11 for k in paperA_figures.snap_panels.pool if k[2] == 4)
    assert all(r["loc"] == 11 for r in panels2)          # Results 4.2.3: "all ... from loc11"

    # Full-test-set own-station capture (cluster job EVAL_SNAPROWS, patch sr-20). The two
    # stages are scored on DIFFERENT event pools (Stage 1 draws windows per node count,
    # Stage 2 draws them paired), so the columns of Table 11 are not a controlled
    # comparison; the events common to both, paired, are what settles a stage difference.
    _sr = [T1 / "test_timeseries_dump.npz", T2 / "test_timeseries_dump.npz"]
    if all(P.exists(q) for q in _sr):
        e_one = own_station_events(npload(_sr[0]))
        e_full = own_station_events(npload(_sr[1]), 21)
        e_four = own_station_events(npload(_sr[1]), 4)
        for tag, ev in (("Sone", e_one), ("StwoFull", e_full), ("StwoFour", e_four)):
            c = _capture_summary([v[0] for v in ev.values()])
            out += [(f"snapStation{tag}", f"{c['median']:.3f}",
                     f"own-station capture, per-event median over the full test set (n = {c['n']})"),
                    (f"snapStationIqr{tag}", f"{c['iqr']:.3f}", "interquartile width of the above")]
            if tag in ("Sone", "StwoFour"):               # the abstract quotes both as a percent
                out.append((f"snapStation{tag}Pct", f"{100 * c['median']:.0f}",
                            "the same median as a percentage (abstract, Conclusions)"))
        both = sorted(set(e_one) & set(e_full))
        d = [e_full[k][0] - e_one[k][0] for k in both]
        out += [("snapStationCommonN", f"{len(both)}",
                 "events with an exact own station in BOTH test draws"),
                ("snapStationCommonDiff", f"{statistics.median(d):+.3f}",
                 "paired Stage-1 -> Stage-2 own-station capture, median difference"),
                ("snapStationCommonPct", f"{100 * sum(1 for x in d if x > 0) / len(d):.0f}",
                 "share of those events where Stage 2 captures more [%]")]
        # by site, the same numbers as the last three rows of Table 11
        for (_lbl, in_site), stag in zip(SITE_GROUPS[:3], ("Dev", "Ten", "Eleven")):
            for tag, ev in (("Full", e_full), ("Four", e_four)):
                v = [c[0] for k, c in ev.items() if in_site(k[0])]
                out.append((f"snapStationSite{stag}{tag}", f"{statistics.median(v):.3f}",
                            f"own-station capture, Stage 2, N_in = {'21' if tag == 'Full' else '4'}, "
                            f"site group {stag} (n = {len(v)}); Table 11 rows"))
            # how many events each site median of Table 11 rests on (Stage 2: the same events
            # at N_in = 4 and 21, the paired draw)
            n1 = sum(1 for k in e_one if in_site(k[0]))
            n2 = sum(1 for k in e_full if in_site(k[0]))
            assert n2 == sum(1 for k in e_four if in_site(k[0]))
            out += [(f"snapStationSiteN{stag}Sone", f"{n1}",
                     f"Stage-1 events behind the Table 11 site row {stag}"),
                    (f"snapStationSiteN{stag}Stwo", f"{n2}",
                     f"Stage-2 events behind the Table 11 site row {stag} (both N_in columns)")]

    node = np.array([int(r["node"]) for r in cat.values()])
    out += [("snapLowerQuarterPct", f"{100 * np.mean(node <= 5):.0f}",
             f"{SNAP_CAT.name}, events peaking at stations 0-5 of 21, of {len(node)}"),
            ("snapFairleadEndPct", f"{100 * np.mean(node >= 16):.0f}",
             "events peaking at stations 16-20 of 21")]

    # ---- learned loss weights (Kendall sigma) at the selected epochs
    for path, ep, tag in ((S1, 32, "Sone"), (S2, EP, "Stwo")):
        row = [r for r in rows(path / "metrics_history.csv") if int(r["epoch"]) == ep][0]
        for term, t in (("catenary", "Cat"), ("momentum", "Mom"), ("constitutive", "Const")):
            v = float(row[f"sigma_{term}"])
            out.append((f"sigma{t}{tag}", f"{v:.3f}" if v < 10 else f"{v:.0f}",
                        f"{path.name} metrics_history.csv epoch {ep} sigma_{term} = {v!r}"))
    return out


@functools.lru_cache(maxsize=None)
def site_group_sweep():
    """Stage-2 MAE (window-averaged), R2 and MAPE per site group and N_in, at N_out = 21.

    From the noise job's clean-level per-window sufficient statistics, which reproduce
    the shipped ep76 model to 0.0000 %; every N_in scores the SAME 5000 windows (the
    sr-02 paired draw), which the assert checks.
    """
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    wins = collections.defaultdict(set)
    for r in rows(SN / "test_per_window_stats.csv"):
        lc, n = int(r["lc_id"]), int(r["n_in"])
        wins[n].add((lc, int(r["case_id"]), int(r["start_idx"])))
        for grp in ("dev" if lc in DEV_SITES else lc, "all"):
            a = acc[(grp, n)]
            a["n"] += float(r["n_pts"]); a["ss"] += float(r["ss_res"])
            a["ys"] += float(r["y_sum"]); a["yq"] += float(r["y_sq"])
            a["mae"] += float(r["mae"]); a["mape"] += float(r["mape"]); a["w"] += 1
    assert all(wins[n] == wins[4] for n in wins), "per-window rows are not paired over N_in"
    out = {}
    for (grp, n), a in acc.items():
        out[(grp, n)] = dict(mae=a["mae"] / a["w"], mape=a["mape"] / a["w"], w=int(a["w"]),
                             r2=1.0 - a["ss"] / (a["yq"] - a["ys"] ** 2 / a["n"]))
    g = rows(T2 / "superres_grid.csv")
    for n in LAYOUTS:                     # the rebuild must reproduce the grid cell
        ref = grid_cell(g, n, N_OUT, "MAE_tension")
        assert abs(out[("all", n)]["mae"] - ref) <= 1e-6 * ref, (n, out[("all", n)]["mae"], ref)
    return out


@functools.lru_cache(maxsize=None)
@derived
def true_contact_regions():
    """Stage-2 per-station R2 by region with TRUE 21-station contact labels.

    The grid's regional R2 labels regions from the input contact flag interpolated to
    the output stations. This check relabels from the finite-element z at all 21
    stations (npy cache), on the dump's unbiased 'typical' reservoir windows (ty, pooled
    and per layout), deduplicated on (loc, case, N_in, start).
    """
    import plot_tension_along_line as ALONG
    z = npload(T2 / "test_timeseries_dump.npz", allow_pickle=True)
    res, seen = collections.defaultdict(list), set()
    for key in z.files:
        if not re.match(r"^(?:nc\d+_)?ty\d+_meta$", key):
            continue
        stem = key[:-len("_meta")]
        loc, case, nin, start = (int(v) for v in z[key])
        if (loc, case, nin, start) in seen:
            continue
        seen.add((loc, case, nin, start))
        tr, pr = z[stem + "_true"], z[stem + "_pred"]
        _inc, frac = ALONG.contact_fraction(loc, case, start, tr.shape[0], N_OUT)
        for i in range(N_OUT):
            t = tr[:, i]
            sst = float(((t - t.mean()) ** 2).sum())
            if sst <= 0:
                continue
            reg = ("grounded" if frac[i] >= 0.98 else
                   "suspended" if frac[i] <= 0.02 else "touchdown")
            res[reg].append(1.0 - float(((t - pr[:, i]) ** 2).sum()) / sst)
    return len(seen), {k: (len(v), statistics.median(v)) for k, v in res.items()}


@derived
def checkpoint_dump(ep):
    """The arrays Results 4.2.3 reads from one Stage-2 per-checkpoint test dump (the pinned
    windows of the published draw); these dumps are not in the archives."""
    z = npload(S2 / f"test_timeseries_dump_epoch{ep:04d}.npz")
    return {k: z[k] for k in ("meta_all", "snap_ev", "peak_true_all", "peak_pred_all",
                               "snap_true", "snap_pred")}


@derived
def four_sensor_station(stem):
    """The touchdown station of one Fig. 8 window (paperA_figures.four_sensor_window): its
    traces and seabed contact. The contact comes from the raw simulations."""
    import paperA_figures
    w = paperA_figures.four_sensor_window(stem)
    i = w["stations"][1]
    return dict(loc=w["loc"], hs=w["hs"], i=i, frac=w["frac"][i], true=w["true"][:, i],
                pred=w["pred"][:, i], interp=w["interp"][:, i], inc=w["inc"][:, i])


def build_results_sparse():
    """Section 4.2 prose macros: sensor-count sweep, touchdown, tail and snaps."""
    g = rows(T2 / "superres_grid.csv")
    gb = rows(T2 / "superres_grid_baseline.csv")
    d = lambda n, k, nout=N_OUT: grid_cell(g, n, nout, k)                # noqa: E731
    b = lambda n, k, nout=N_OUT: grid_cell(gb, n, nout, k)               # noqa: E731
    mae_d = [d(n, "MAE_tension") for n in LAYOUTS]
    out = [
        ("srMAELo", f"{min(mae_d):.1f}", "S2 grid N_out=21, smallest decoder MAE over N_in"),
        ("srMAEHi", f"{max(mae_d):.1f}", "same, largest (N_in = 4)"),
        ("srPenBaseSix", f"{b(6, 'MAE_tension') / b(21, 'MAE_tension'):.2f}",
         "baseline MAE(6,21)/MAE(21,21)"),
        ("srPenBaseFive", f"{b(5, 'MAE_tension') / b(21, 'MAE_tension'):.2f}",
         "baseline MAE(5,21)/MAE(21,21)"),
        ("srSkillFour", f"{d(4, 'temporal_diff_skill_tension'):.3f}", "decoder S at (4,21)"),
        ("srSkillBaseFour", f"{b(4, 'temporal_diff_skill_tension'):.3f}", "baseline S at (4,21)"),
        ("srSkillFull", f"{d(21, 'temporal_diff_skill_tension'):.3f}", "decoder S at (21,21)"),
        ("sOneFullSkill", f"{b(21, 'temporal_diff_skill_tension'):.3f}",
         "baseline S at (21,21) = the Stage-1 model itself, identity interpolation"),
        ("srSkillPenBase", f"{b(4, 'temporal_diff_skill_tension') / b(21, 'temporal_diff_skill_tension'):.2f}",
         "baseline S(4,21)/S(21,21): the interpolation's temporal penalty"),
        ("srFullGainPct", f"{100 * (1 - d(21, 'MAE_tension') / b(21, 'MAE_tension')):.0f}",
         "1 - decoder/Stage-1 MAE at (21,21) [%]"),
    ]
    # R2 and MAPE against the baseline at the same N_in
    r2_wins = sum(d(n, "global_R2_tension") > b(n, "global_R2_tension") for n in LAYOUTS)
    # Results 4.2.1 says the decoder's R2 exceeds the baseline's at EVERY layout, by the
    # smallest margin at full sensing (on the clean draw; on the published one N_in = 21 tied)
    margin = {n: d(n, "global_R2_tension") - b(n, "global_R2_tension") for n in LAYOUTS}
    assert r2_wins == len(LAYOUTS) and min(margin, key=margin.get) == N_OUT, margin
    mape_cross = min(n for n in LAYOUTS if d(n, "MAPE_tension") > b(n, "MAPE_tension"))
    assert all(d(n, "MAPE_tension") > b(n, "MAPE_tension") for n in LAYOUTS if n >= mape_cross)
    out += [("srRsqWins", f"{r2_wins}", "N_in (of 10) at which decoder global R2 > baseline"),
            ("srMAPECross", f"{mape_cross}",
             "smallest N_in from which the decoder's MAPE exceeds the baseline's (at every larger N_in too)")]
    # grid-wide counts
    up = [(i, o) for i in LAYOUTS for o in LAYOUTS if i < o]
    out += [("srGridUpper", f"{len(up)}", "cells with N_in < N_out"),
            ("srGridMAEWins", f"{sum(d(i, 'MAE_tension', o) < b(i, 'MAE_tension', o) for i, o in up)}",
             "upper-triangle cells where the decoder MAE is lower"),
            ("srGridSkillWins",
             f"{sum(d(i, 'temporal_diff_skill_tension', o) < b(i, 'temporal_diff_skill_tension', o) for i in LAYOUTS for o in LAYOUTS)}",
             "cells of all 100 where the decoder skill score is lower (better)")]

    # ---- site groups: where the sensing penalty lives
    sg = site_group_sweep()
    for grp, tag in (("dev", "Dev"), (10, "Ten"), (11, "Eleven")):
        out += [(f"srPen{tag}", f"{sg[(grp, 4)]['mae'] / sg[(grp, N_OUT)]['mae']:.2f}",
                 f"SN clean per-window, group {grp}: MAE N_in=4 / N_in=21"),
                (f"srMAEFour{tag}", f"{sg[(grp, 4)]['mae']:.1f}", f"group {grp}, MAE at N_in=4 [N]"),
                (f"srMAEFull{tag}", f"{sg[(grp, N_OUT)]['mae']:.1f}", f"group {grp}, MAE at N_in=21 [N]"),
                (f"srMAPEFull{tag}", f"{sg[(grp, N_OUT)]['mape']:.2f}",
                 f"group {grp}, MAPE at N_in=21 [%]")]
    ten_share = sg[(10, N_OUT)]["w"] / sg[("all", N_OUT)]["w"]
    ten_mape_share = (sg[(10, N_OUT)]["mape"] * sg[(10, N_OUT)]["w"]
                      / (sg[("all", N_OUT)]["mape"] * sg[("all", N_OUT)]["w"]))
    out += [("srTenWinPct", f"{100 * ten_share:.0f}", "loc10 share of the test windows [%]"),
            ("srTenMAPEPct", f"{100 * ten_mape_share:.0f}",
             "loc10 share of the pooled decoder MAPE at N_in=21 [%]")]
    z2 = npload(T2 / "test_timeseries_dump.npz")
    m2, t2, p2 = z2["meta_all"], z2["peak_true_all"], z2["peak_pred_all"]
    # ---- query-position limit: output columns on / off the native 21-station grid
    col = {o: statistics.mean(d(i, "MAE_tension", o) for i in LAYOUTS) for o in LAYOUTS}
    on = [o for o in LAYOUTS if 20 % (o - 1) == 0]
    off = [o for o in LAYOUTS if o not in on]
    pool = {int(r["N_out"]): float(r["MAE_tension"]) for r in rows(T2 / "superres_grid_pooled.csv")}
    assert all(abs(pool[o] - col[o]) < 1e-6 * col[o] for o in LAYOUTS)
    out += [("srOnGridList", ", ".join(map(str, on[:-1])) + f" and {on[-1]}",
             "output layouts whose stations all lie on the native 21-station grid"),
            ("srOnGridLo", f"{min(col[o] for o in on):.1f}", "decoder MAE pooled over N_in, on-grid N_out, min"),
            ("srOnGridHi", f"{max(col[o] for o in on):.1f}", "same, max"),
            ("srOffGridLo", f"{min(col[o] for o in off):.1f}", "off-grid N_out, min"),
            ("srOffGridHi", f"{max(col[o] for o in off):.1f}", "off-grid N_out, max"),
            ("srOffCellD", f"{d(21, 'MAE_tension', 15):.1f}", "decoder MAE at (21,15)"),
            ("srOffCellB", f"{b(21, 'MAE_tension', 15):.1f}", "baseline MAE at (21,15)"),
            ("decFinestPeriod", f"{2 ** (int(kv_config()['query_decoder_fourier_bands']) - 2)}",
             "finest Fourier band sin(2^(K-1) pi s) has period 1/2^(K-2) of the line")]

    # ---- touchdown: true-contact check on the unbiased reservoir windows
    nw, tc = true_contact_regions()
    out += [("srTrueWin", f"{nw}", "S2 clean dump ty windows, all N_in, deduplicated")]
    for reg, tag in (("grounded", "Gr"), ("touchdown", "Td"), ("suspended", "Su")):
        out += [(f"srTrue{tag}Rsq", f"{tc[reg][1]:.3f}",
                 f"median per-station R2, TRUE contact label {reg}"),
                (f"srTrue{tag}N", f"{tc[reg][0]}", f"station-windows labelled {reg}")]
    # the two windows of Fig. 8, one per withheld site (paperA_figures.FOUR_WINDOWS). Both rows
    # carry the comparison against the true tension at the four sensed stations, interpolated.
    import paperA_figures
    r2f = lambda t, p: 1.0 - float(((t - p) ** 2).sum()) / float(((t - t.mean()) ** 2).sum())  # noqa: E731
    mae = lambda t, p: float(np.abs(t - p).mean())                                             # noqa: E731
    # R2 inside [0, 1] to two decimals, a negative one to one -- its magnitude is the message
    _r2fmt = lambda v: f"{v:.2f}" if 0.0 <= v <= 1.0 else f"{v:.1f}"                           # noqa: E731
    for stem, tag in ((paperA_figures.FOUR_WINDOWS[0], ""),
                      (paperA_figures.FOUR_WINDOWS[1], "B")):
        w = four_sensor_station(stem)
        i = w["i"]                                 # the touchdown station of that row
        row = "b" if tag else "a"
        out += [(f"figFour{tag}Site", f"loc{w['loc']:02d}", f"Fig. 8 {row} row, site"),
                (f"figFour{tag}Hs", f"{w['hs']:.1f}", "its Hs [m]"),
                (f"figFour{tag}Station", f"{i}", "its touchdown station"),
                (f"figFour{tag}TdPct", f"{100 * w['frac']:.0f}", "share of the window on the seabed [%]"),
                (f"figFour{tag}TdRsq", f"{r2f(w['true'], w['pred']):.3f}", "decoder R2 there"),
                (f"figFour{tag}TdMAE", f"{mae(w['true'], w['pred']):.0f}", "decoder MAE there [N]"),
                (f"figFour{tag}TdRsqInterp", _r2fmt(r2f(w['true'], w['interp'])),
                 "R2 of the TRUE tension at the four sensed stations, interpolated to it"),
                (f"figFour{tag}TdMAEInterp", f"{mae(w['true'], w['interp']):.0f}",
                 "interpolated-truth MAE there [N]"),
                (f"figFour{tag}TdArc", f"{i / 20:.2f}",
                 "its normalised arc length (sensors sit at 0, 1/3, 2/3, 1)"),
                (f"figFour{tag}TdLifts", str(int(np.sum(np.diff((~w['inc']).astype(np.int8)) == 1)
                                                + (1 if not w['inc'][0] else 0))),
                 "times that station leaves the seabed inside the window")]

    # ---- the tail: Stage-2 fairlead peaks split by snap membership
    ts = tail_split(z2)
    for q, band in ((90, "Ninety"), (99, "NinetyNine")):
        out.append((f"sTwoThr{band}", f"{ts[(q, 'all')]['thr']:,.0f}".replace(",", "\\,"),
                    f"S2 clean dump, p{q} of true fairlead peak [N], N_out=21, pooled over N_in"))
        for pop, tag in (("nonsnap", "NonSnap"), ("snap", "Snap")):
            t_ = ts[(q, pop)]
            out += [(f"sTwo{tag}Up{band}", f"{t_['up']:.1f}", f"{pop}, n = {t_['n']}, u [%]"),
                    (f"sTwo{tag}Bias{band}", f"{t_['bias']:+.0f}", "same, mean signed error [N]"),
                    (f"sTwo{tag}UM{band}", f"{t_['um']:,.0f}".replace(",", "\\,"), "same, under-mass [N]")]
    dev = np.isin(m2[:, 0], DEV_SITES)
    out += [("sTwoDevRatio", f"{float(np.median(p2[dev] / t2[dev])):.3f}",
             "S2 clean dump, training sites, median fairlead peak pred/true, all N_in"),
            ("sTwoDevUp", f"{100 * float(np.mean(p2[dev] < t2[dev])):.0f}",
             "same windows, share underpredicted [%]")]
    # under-mass against the interpolation baseline, every cell of the grid
    um = lambda rs, i, o, q: undermass(grid_cell(rs, i, o, f"peak_tension_MAE_p{q}"),   # noqa: E731
                                       grid_cell(rs, i, o, f"peak_tension_bias_p{q}"))
    for q, band in ((90, "Ninety"), (99, "NinetyNine")):
        rat = [um(g, i, o, q) / um(gb, i, o, q) for i in LAYOUTS for o in LAYOUTS]
        out += [(f"srUMCells{band}", f"{sum(r > 1 for r in rat)}",
                 f"cells of 100 where the decoder p{q} under-mass exceeds the baseline's"),
                (f"srUMRatio{band}", f"{float(np.median(rat)):.2f}",
                 f"median decoder/baseline p{q} under-mass over the 100 cells")]

    # ---- the epoch-33 alternative, on the pinned windows of the per-checkpoint dumps
    za, zb = checkpoint_dump(33), checkpoint_dump(EP)
    assert (za["meta_all"] == zb["meta_all"]).all() and (za["snap_ev"] == zb["snap_ev"]).all()
    # The pinned windows are drawn from the PUBLISHED draw: keep those the evaluation mask
    # leaves, and snap rows of events that are in the evaluation catalogue. These dumps joined
    # the earlier catalogue, so an event the new one moved to another sample is absent here.
    keep = np.array([not eval_dropped(m[0], m[1], m[3]) for m in za["meta_all"]])
    cat_ev = {(int(r["loc"]), int(r["case"]), int(r["t"])) for r in rows(SNAP_CAT)}
    rkeep = np.array([(not eval_dropped(a_, b_, s_)) and (int(a_), int(b_), int(t_)) in cat_ev
                      for a_, b_, _n, s_, t_ in za["snap_ev"]], bool)
    pa = za["peak_true_all"][keep]

    def _pin(zz):
        return dict(meta_all=zz["meta_all"][keep], peak_pred_all=zz["peak_pred_all"][keep],
                    snap_ev=zz["snap_ev"][rkeep], snap_true=zz["snap_true"][rkeep],
                    snap_pred=zz["snap_pred"][rkeep])
    za, zb = _pin(za), _pin(zb)
    hist = {int(r["epoch"]): r for r in rows(S2 / "metrics_history.csv")}
    out += [("altNWin", f"{len(pa):,}".replace(",", "\\,"),
             f"pinned windows per per-checkpoint dump kept by the evaluation mask "
             f"({int((~keep).sum())} removed)"),
            ("altValMAE", f"{float(hist[33]['val_MAE_tension']):.1f}", "val MAE at epoch 33 [N]"),
            ("selValMAE", f"{float(hist[EP]['val_MAE_tension']):.1f}", f"val MAE at epoch {EP} [N]")]
    for zz, tag in ((za, "alt"), (zb, "sel")):
        e = zz["peak_pred_all"] - pa
        for q, band in ((90, "Ninety"), (99, "NinetyNine")):
            tail = pa >= np.quantile(pa, q / 100.0)
            s = e[tail]
            out += [(f"{tag}Up{band}", f"{100 * float((s < 0).mean()):.1f}",
                     f"per-ckpt dump, fairlead p{q} tail, u [%]"),
                    (f"{tag}UM{band}", f"{undermass(float(np.abs(s).mean()), float(s.mean())):,.0f}".replace(",", "\\,"),
                     f"same, under-mass [N]")]
        ten = zz["meta_all"][:, 0] == 10
        out += [(f"{tag}TenRatio", f"{float(np.median(zz['peak_pred_all'][ten] / pa[ten])):.3f}",
                 "loc10 median fairlead peak pred/true"),
                (f"{tag}SnapCap", f"{snap_stats(snap_events(zz))['median']:.3f}",
                 "per-event snap capture on the pinned windows")]

    # ---- snaps from sparse sensing
    _ev, c = snap_comparison()
    for k, tag in (("c1", "One"), ("c21", "TwentyOne"), ("c4", "Four")):
        out += [(f"snapCom{tag}Med", f"{c[k]['median']:.3f}", f"common events, {k}, median capture"),
                (f"snapCom{tag}IQR", f"{c[k]['iqr']:.3f}", "same, interquartile range"),
                (f"snapCom{tag}Ninety", f"{c[k]['m90']:.2f}", "same, multiplier for 90 % coverage")]
    out.append(("snapComN", f"{c['c1']['n']}", "events common to the Stage-1 and Stage-2 test draws"))
    # Results 4.2.3: "from four sensors ..., above the ... at matched resolution"
    assert c["c4"]["median"] > c["c1"]["median"], (c["c4"]["median"], c["c1"]["median"])
    ev4, ev21 = snap_events(z2, 4), snap_events(z2, N_OUT)
    out.append(("snapNStwo", f"{len(ev4)}", "distinct snap events in the Stage-2 test windows"))
    for grp, tag in (("dev", "Dev"), (10, "Ten"), (11, "Eleven")):
        sel = (lambda k: k[0] in DEV_SITES) if grp == "dev" else (lambda k, g_=grp: k[0] == g_)
        for ev, nt in ((ev4, "Four"), (ev21, "Full")):
            r = [v[2] for k, v in ev.items() if sel(k)]
            out.append((f"snapCap{tag}{nt}", f"{statistics.median(r):.3f}",
                        f"S2 clean dump, group {grp}, N_in={'4' if nt == 'Four' else '21'}, "
                        f"per-event median capture, n = {len(r)}"))
        out.append((f"snapNStwo{tag}", f"{sum(1 for k in ev4 if sel(k))}", f"events in group {grp}"))
    # Results 4.2.3 calls loc10's Stage-2 snap "a single event" (clean draw): asserted
    assert sum(1 for k in ev4 if k[0] == 10) == 1 and sum(1 for k in ev21 if k[0] == 10) == 1
    return out


@functools.lru_cache(maxsize=None)
def kv_config():
    import json
    return json.loads(P.read_text(S2 / "config.json"))


# ------------------------------------------------------------------ table bodies
def _row(cells):
    return " & ".join(cells) + " \\\\\n"


def table_layouts():
    """Table 6: Stage-1 test metrics per sensor layout, plus the pooled row."""
    per_n = {int(r["node_count"]): r for r in rows(T1 / "test_metrics_by_node_count.csv")}
    p = kv(T1 / "test_metrics.csv")

    def cells(get, label):
        um90 = undermass(get("peak_tension_MAE_p90"), get("peak_tension_bias_p90"))
        um99 = undermass(get("peak_tension_MAE_p99"), get("peak_tension_bias_p99"))
        return [label, f"{get('global_R2_tension'):.4f}", f"{get('MAE_tension'):.1f}",
                f"{get('MAPE_tension'):.2f}", f"{get('temporal_diff_skill_tension'):.3f}",
                f"{get('peak_tension_MAE'):.1f}",
                f"{get('peak_tension_thr_p90') / 1000:.2f}",
                f"{get('peak_tension_underpred_rate_p90'):.1f}", f"{um90:.0f}",
                f"{get('peak_tension_thr_p99') / 1000:.2f}",
                f"{get('peak_tension_underpred_rate_p99'):.1f}", f"{um99:.0f}"]

    s = ("\\begin{tabular}{@{}S[table-format=2.0]S[table-format=1.4]S[table-format=2.1]"
         "S[table-format=1.2]S[table-format=1.3]S[table-format=3.1]S[table-format=2.2]"
         "S[table-format=2.1]S[table-format=3.0]S[table-format=2.2]S[table-format=2.1]"
         "S[table-format=4.0]@{}}\n\\toprule\n"
         " & \\multicolumn{4}{c}{Tension field} & {Peaks} & \\multicolumn{3}{c}{$p_{90}$ peak tail}"
         " & \\multicolumn{3}{c}{$p_{99}$ peak tail} \\\\\n"
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}\n")
    s += _row(["{$N$}", "{$R^2$}", "{MAE}", "{MAPE}", "{$S$}", "{MAE}", "{$T_{90}$}",
               "{$u_{90}$}", "{Under-mass}", "{$T_{99}$}", "{$u_{99}$}", "{Under-mass}"])
    s += _row(["", "", "{[\\si{\\newton}]}", "{[\\si{\\percent}]}", "", "{[\\si{\\newton}]}",
               "{[\\si{\\kilo\\newton}]}", "{[\\si{\\percent}]}", "{[\\si{\\newton}]}",
               "{[\\si{\\kilo\\newton}]}", "{[\\si{\\percent}]}", "{[\\si{\\newton}]}"])
    s += "\\midrule\n"
    for n in LAYOUTS:
        s += _row(cells(lambda k: float(per_n[n][k]), f"{n}"))
    s += "\\midrule\n"
    s += _row(cells(lambda k: float(p[f"test_{k}"]), "{All}"))
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


def table_sites():
    """Table 7: per-site fairlead-peak statistics, the same three for Stage 1 and Stage 2."""
    s = ("\\begin{tabular}{@{}l S[table-format=1.3] S[table-format=1.3] S[table-format=1.2]"
         " S[table-format=2.1] S[table-format=1.3] S[table-format=1.2] S[table-format=2.1]@{}}\n"
         "\\toprule\n"
         " & & \\multicolumn{3}{c}{Stage~1} & \\multicolumn{3}{c}{Stage~2} \\\\\n"
         "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n")
    s += _row(["{Site}", "{$h_0/L_0$}", "{$\\tilde{\\rho}$}", "{$\\mathrm{MAPE}^{\\mathrm{peak}}$}",
               "{$u^{\\mathrm{peak}}$}", "{$\\tilde{\\rho}$}", "{$\\mathrm{MAPE}^{\\mathrm{peak}}$}",
               "{$u^{\\mathrm{peak}}$}"])
    s += _row(["", "", "", "{[\\si{\\percent}]}", "{[\\si{\\percent}]}", "", "{[\\si{\\percent}]}",
               "{[\\si{\\percent}]}"])
    s += "\\midrule\n"
    for r in site_rows():
        if r["site"] == 10:
            s += "\\midrule\n"
        name = "Training, pooled" if r["site"] == "dev" else f"loc{r['site']:02d}"
        hl = "{--}" if r["hl"] is None else f"{r['hl']:.3f}"
        a, b = r["s1"], r["s2"]
        s += _row([name, hl, f"{a['ratio']:.3f}", f"{a['relerr']:.2f}", f"{a['up']:.1f}",
                   f"{b['ratio']:.3f}", f"{b['relerr']:.2f}", f"{b['up']:.1f}"])
        if r["site"] == 9:
            s += "\\addlinespace\n"
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


def table_regions():
    """Table 8: median per-station R2 by contact region, Stage 1 / Stage 2 / interpolation."""
    p1 = kv(T1 / "test_metrics.csv")
    p2 = pooled_row(rows(T2 / "superres_grid_pooled.csv"), N_OUT)
    b2 = pooled_row(rows(T2 / "superres_grid_pooled_baseline.csv"), N_OUT)
    s = ("\\begin{tabular}{@{}l S[table-format=1.3] S[table-format=1.1] S[table-format=1.3]"
         " S[table-format=1.1] S[table-format=1.3] S[table-format=2.1]@{}}\n\\toprule\n"
         " & \\multicolumn{2}{c}{Stage~1} & \\multicolumn{2}{c}{Stage~2} & "
         "\\multicolumn{2}{c}{Interpolation} \\\\\n"
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n")
    s += _row(["{Region}", "{Median}", "{$<0$ [\\si{\\percent}]}", "{Median}",
               "{$<0$ [\\si{\\percent}]}", "{Median}", "{$<0$ [\\si{\\percent}]}"])
    s += "\\midrule\n"
    for regime in ("grounded", "touchdown", "suspended"):
        s += _row([regime.capitalize(),
                   f"{float(p1[f'test_contact_{regime}_R2_median']):.3f}",
                   f"{float(p1[f'test_contact_{regime}_R2_fracneg']):.1f}",
                   f"{float(p2[f'contact_{regime}_R2_median']):.3f}",
                   f"{float(p2[f'contact_{regime}_R2_fracneg']):.1f}",
                   f"{float(b2[f'contact_{regime}_R2_median']):.3f}",
                   f"{float(b2[f'contact_{regime}_R2_fracneg']):.1f}"])
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


def table_tails():
    """Table 9: fairlead-peak tails split by snap membership, Stage 1 then Stage 2.

    Each stage has its own thresholds (quantiles of its own test draw), so they are given in
    the stage's header row rather than in the caption."""
    s = ("\\begin{tabular}{@{}l l S[table-format=4.0] S[table-format=2.1] S[table-format=2.1]"
         " S[table-format=+4.0] S[table-format=4.0]@{}}\n\\toprule\n")
    # units on their own row, as in Table 6, so the table fits one column
    s += _row(["{Tail}", "{Window}", "{$n$}", "{Share}", "{$u$}", "{Bias}", "{Under-mass}"])
    s += _row(["", "", "", "{[\\si{\\percent}]}", "{[\\si{\\percent}]}", "{[\\si{\\newton}]}",
               "{[\\si{\\newton}]}"])
    for stage, z in (("Stage 1", npload(T1 / "test_timeseries_dump.npz")),
                     ("Stage 2", npload(T2 / "test_timeseries_dump.npz"))):
        ts = tail_split(z)
        t90, t99 = ts[(90, "all")]["thr"], ts[(99, "all")]["thr"]
        s += "\\midrule\n"
        s += (f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{stage}}}, $T_{{90}} = "
              f"\\SI{{{t90:.0f}}}{{\\newton}}$, $T_{{99}} = \\SI{{{t99:.0f}}}{{\\newton}}$}} \\\\\n")
        s += "\\addlinespace\n"
        for q in (90, 99):
            for i, (pop, label) in enumerate((("nonsnap", "no snap"), ("snap", "snap"))):
                d = ts[(q, pop)]
                s += _row([f"$p_{{{q}}}$" if i == 0 else "", label, f"{d['n']}", f"{d['share']:.1f}",
                           f"{d['up']:.1f}", f"{d['bias']:+.0f}", f"{d['um']:.0f}"])
            if q == 90:
                s += "\\addlinespace\n"
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


def _capture_summary(caps):
    """n, median, interquartile width and share captured in full, of per-event captures."""
    a = np.asarray(caps, dtype=float)
    q1, q3 = np.quantile(a, [0.25, 0.75])
    return dict(n=int(a.size), median=float(np.median(a)), iqr=float(q3 - q1),
                atleast=100.0 * float(np.mean(a >= 1.0)))


SITE_GROUPS = (("Training sites", lambda lc: lc in DEV_SITES), ("loc10", lambda lc: lc == 10),
               ("loc11", lambda lc: lc == 11), ("All", lambda lc: True))


def table_snaps():
    """Table 10: Stage-1 snap capture of the LINE MAXIMUM per distinct event, per site group.

    The table used to split the events by whether the snap peak reaches the concurrent
    fairlead tension. Under the evaluation catalogue (fairlead condition, cat <= 1) no event
    does -- the line maximum is the fairlead tension at every event (asserted in
    build_results_matched) -- so only the site grouping remains."""
    ev1, _c = snap_comparison()
    s = ("\\begin{tabular}{@{}l S[table-format=3.0] S[table-format=1.3]"
         " S[table-format=2.1]@{}}\n\\toprule\n")
    s += _row(["", "{Events}", "{Median capture}", "{In full [\\si{\\percent}]}"])
    s += "\\midrule\n"
    for site_label, in_site in SITE_GROUPS:
        caps = [v[2] for k, v in ev1.items() if in_site(k[0])]
        if not caps:
            continue
        d = _capture_summary(caps)
        if site_label == "All":
            s += "\\addlinespace\n"
        s += _row([site_label, f"{d['n']}", f"{d['median']:.3f}", f"{d['atleast']:.1f}"])
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


SNAPROWS = ROOT / "Results" / "SR_RESULTS" / "checkpoints_P2_eval_snaprows"


def own_station_events(z, n_in=None):
    """Snap capture at the snap's OWN station, per distinct event (patch sr-20 dumps).

    snap_nbhd_true/pred hold every output station for the event sample +-2. The
    snap's station is the catalogued native node; it is taken only where it IS an
    output station of the row's layout (always at N_out = 21; at matched
    resolution only when node*(N-1)/20 is an integer). Returns
    {event: (capture at the event sample, capture of the +-2-sample maxima,
             line-maximum capture of the same rows)}.
    """
    cat = {(int(r["loc"]), int(r["case"]), int(r["t"])): int(r["node"])
           for r in rows(SNAP_CAT)}
    offs = [int(o) for o in z["snap_nbhd_offsets"]]
    i0 = offs.index(0)
    g = collections.defaultdict(list)
    for k, (lc, case, nin, _s, t) in enumerate(z["snap_ev"].tolist()):
        if n_in is not None and int(nin) != n_in:
            continue
        nout = int(z["snap_row_nout"][k])
        x = cat[(lc, case, t)] * (nout - 1) / 20.0
        if abs(x - round(x)) > 1e-9:
            continue
        i = int(round(x))
        tt, pp = z["snap_nbhd_true"][k, :, i], z["snap_nbhd_pred"][k, :, i]
        if not tt[i0] > 0:
            continue
        g[(lc, case, t)].append((float(pp[i0] / tt[i0]),
                                 float(np.nanmax(pp) / np.nanmax(tt)),
                                 float(z["snap_pred"][k] / z["snap_true"][k])))
    return {key: tuple(statistics.median(v[j] for v in vals) for j in range(3))
            for key, vals in g.items()}


def snap_station_availability(z, stage2, n_in=None):
    """(events with the snap's own station among the outputs, all distinct events).

    Output stations per row: 21 for Stage 2; N for Stage 1 (snap_ev column 2), or
    snap_row_nout where the dump carries it (patch sr-20)."""
    node = {(int(r["loc"]), int(r["case"]), int(r["t"])): int(r["node"])
            for r in rows(SNAP_CAT)}
    nout = z["snap_row_nout"] if "snap_row_nout" in z.files else None
    allev, avail = set(), set()
    for k, (lc, case, nin, _s, t) in enumerate(z["snap_ev"].tolist()):
        if n_in is not None and int(nin) != n_in:
            continue
        key = (lc, case, t)
        allev.add(key)
        n = int(nout[k]) if nout is not None else (21 if stage2 else int(nin))
        if (node[key] * (n - 1)) % 20 == 0:
            avail.add(key)
    return len(avail), len(allev)


def table_snapstation():
    """Table 11: snap capture at the station where the snap occurs, per distinct event,
    over the FULL test sets (cluster job EVAL_SNAPROWS, patch sr-20). Until that job's
    dumps are in Results/SR_RESULTS/checkpoints_P2_eval_snaprows/ a placeholder is
    written with the final layout."""
    cols = (("{Stage~1}", T1 / "test_timeseries_dump.npz", None),
            ("{$N_{\\mathrm{in}}=21$}", T2 / "test_timeseries_dump.npz", 21),
            ("{$N_{\\mathrm{in}}=4$}", T2 / "test_timeseries_dump.npz", 4))
    have = all(P.exists(p) for _l, p, _n in cols)
    evs = [own_station_events(npload(p), n) for _l, p, n in cols] if have else None
    # availability depends only on the event rows, which the job must reproduce exactly
    # (its in-job gate), so it is read from the published dumps until the job lands
    pub = (T1 / "test_timeseries_dump.npz", T2 / "test_timeseries_dump.npz",
           T2 / "test_timeseries_dump.npz")
    avail = [snap_station_availability(npload(p if have else q), stage2=(n is not None), n_in=n)
             for (_l, p, n), q in zip(cols, pub)]
    if have:
        assert [len(ev) for ev in evs] == [a for a, _t in avail], "own-station events != availability"
    s = ("\\begin{tabular}{@{}l S[table-format=3.3] S[table-format=3.3] S[table-format=3.3]@{}}\n"
         "\\toprule\n & & \\multicolumn{2}{c}{Stage~2} \\\\\n\\cmidrule(lr){3-4}\n")
    s += _row([""] + [c[0] for c in cols])
    s += "\\midrule\n"

    def cells(fn):
        if not have:
            return ["{--}"] * len(cols)
        return [fn(ev) for ev in evs]

    def med(ev, j=0, site=None):
        v = [c[j] for k, c in ev.items() if site is None or site(k[0])]
        return f"{statistics.median(v):.3f}" if v else "{--}"

    s += _row(["Events"] + [f"{a}" for a, _t in avail])
    s += _row(["Not available (sensor layout)"] + [f"{t - a}" for a, t in avail])
    s += _row(["Median capture"] + cells(lambda ev: med(ev, 0)))
    s += _row(["Captured in full [\\si{\\percent}]"] + cells(
        lambda ev: f"{_capture_summary([c[0] for c in ev.values()])['atleast']:.1f}"))
    s += _row(["Median, $\\pm 2$ samples"] + cells(lambda ev: med(ev, 1)))
    s += _row(["Median, line maximum"] + cells(lambda ev: med(ev, 2)))
    s += "\\addlinespace\n\\multicolumn{4}{@{}l}{Median capture by site} \\\\\n"
    for site_label, in_site in SITE_GROUPS[:3]:
        s += _row([f"\\quad {site_label}"] + cells(lambda ev, f=in_site: med(ev, 0, f)))
    if not have:
        s += ("\\addlinespace\n\\multicolumn{4}{@{}l}{\\emph{Pending: cluster job "
              "EVAL\\_SNAPROWS (patch sr-20).}} \\\\\n")
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


def table_sweep():
    """Table 12: Stage 2 against the interpolated Stage-1 model, per N_in, at N_out = 21.

    The final row pools the same five metrics over every sensor layout (superres_grid_pooled*.csv
    at N_out = 21, the file build_stage2/build_stage2_regions also read -- CLAUDE.md 16.15.3/16.20.2).
    """
    g = rows(T2 / "superres_grid.csv")
    gb = rows(T2 / "superres_grid_baseline.csv")
    p = pooled_row(rows(T2 / "superres_grid_pooled.csv"), N_OUT)
    pb = pooled_row(rows(T2 / "superres_grid_pooled_baseline.csv"), N_OUT)
    spec = (("MAE_tension", "{:.1f}"), ("global_R2_tension", "{:.4f}"), ("MAPE_tension", "{:.2f}"),
            ("temporal_diff_skill_tension", "{:.3f}"), ("peak_tension_MAE", "{:.1f}"))
    s = ("\\begin{tabular}{@{}S[table-format=2.0] S[table-format=2.1] S[table-format=3.1]"
         " S[table-format=1.4] S[table-format=1.4] S[table-format=1.2] S[table-format=2.2]"
         " S[table-format=1.3] S[table-format=1.3] S[table-format=2.1] S[table-format=3.1]@{}}\n"
         "\\toprule\n"
         " & \\multicolumn{2}{c}{MAE [\\si{\\newton}]} & \\multicolumn{2}{c}{$R^2$}"
         " & \\multicolumn{2}{c}{MAPE [\\si{\\percent}]} & \\multicolumn{2}{c}{$S$}"
         " & \\multicolumn{2}{c}{Peak MAE [\\si{\\newton}]} \\\\\n"
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}"
         "\\cmidrule(lr){10-11}\n")
    s += _row(["{$N_{\\mathrm{in}}$}"] + ["{St.~2}", "{Interp.}"] * 5)
    s += "\\midrule\n"
    for n in LAYOUTS:
        cells = [f"{n}"]
        for key, fmt in spec:
            cells += [fmt.format(grid_cell(g, n, N_OUT, key)),
                      fmt.format(grid_cell(gb, n, N_OUT, key))]
        s += _row(cells)
    s += "\\midrule\n"
    cells = ["{Pooled}"]
    for key, fmt in spec:
        cells += [fmt.format(float(p[key])), fmt.format(float(pb[key]))]
    s += _row(cells)
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


# ================================================================ Section 4.3 (noise)
# Eval-only job 236311 (CLAUDE.md 16.16/16.17): the shipped Stage-2 ep76 weights and the
# interpolated Stage-1 baseline re-scored with the sensed positions perturbed. Every
# level scores the same 50k paired test windows; the clean level must reproduce the
# shipped model (asserted below). Tags follow the job's file suffixes.
NOISE_LEVELS = (   # (file tag, error model, level label, sigma_p [m])
    ("", "none", "{--}", 0.0),
    ("_p010", "white", "\\SI{0.01}{\\metre}", 0.01),
    ("_p050", "white", "\\SI{0.05}{\\metre}", 0.05),
    ("_p100", "white", "\\SI{0.10}{\\metre}", 0.10),
    ("_p250", "white", "\\SI{0.25}{\\metre}", 0.25),
    ("_coupled050", "differenced", "\\SI{0.05}{\\metre}", 0.05),
    ("_bias050", "offset", "\\SI{0.05}{\\metre}", 0.05),
    ("_snr20", "relative", "\\SI{20}{\\deci\\bel}", None),
    ("_snr10", "relative", "\\SI{10}{\\deci\\bel}", None),
    ("_snr05", "relative", "\\SI{5}{\\deci\\bel}", None),
    ("_snr00", "relative", "\\SI{0}{\\deci\\bel}", None),
)
NOISE_ABS = ("", "_p010", "_p050", "_p100", "_p250")     # the result axis
NOISE_INSTR = "_p050"      # grafe2024virtual Table 6: 0.05 m heave, 0.05 m/s velocity


@functools.lru_cache(maxsize=None)
@derived
def fairlead_motion():
    """Median temporal std of the fairlead's x and z over every finite cached simulation,
    after the forcing ramp, and the share of stations that never move.

    The reference against which a noise level is expressed as a signal-to-noise ratio of
    the MOTION, rather than of the channel's pooled std (which is line layout).
    """
    ramp = int(round(SOLVER["ramp_s"] / SOLVER["dt"]))
    fx, fz, fvx, fvz, still, tot = [], [], [], [], 0, 0
    for d in sorted(CACHE.iterdir()):
        for c in sorted(d.iterdir()):
            x = np.asarray(npload(c / "x_abs.npy", mmap_mode="r")[ramp:], dtype=np.float64)
            z = np.asarray(npload(c / "z_abs.npy", mmap_mode="r")[ramp:], dtype=np.float64)
            if not (np.isfinite(x).all() and np.isfinite(z).all()):
                continue
            sx, sz = x.std(0), z.std(0)
            still += int(((sx == 0) & (sz == 0)).sum()); tot += x.shape[1]
            fx.append(sx[-1]); fz.append(sz[-1])
            dt2 = 2.0 * SOLVER["dt"]          # central difference, as the model's input
            fvx.append(((x[2:, -1] - x[:-2, -1]) / dt2).std())
            fvz.append(((z[2:, -1] - z[:-2, -1]) / dt2).std())
    return dict(n=len(fx), x=statistics.median(fx), z=statistics.median(fz),
                vx=statistics.median(fvx), vz=statistics.median(fvz), still=100.0 * still / tot)


@functools.lru_cache(maxsize=None)
def noise_sigma():
    """{tag: {channel: sigma}} and the channels' training std, from sensor_noise_levels.csv."""
    sig, std = collections.defaultdict(dict), {}
    for r in rows(SN / "sensor_noise_levels.csv"):
        tag = "" if r["tag"] == "clean" else r["tag"]
        if r["channel"]:
            sig[tag][r["channel"]] = float(r["sigma_physical"])
            std[r["channel"]] = float(r["channel_train_std"])
    return sig, std


def snr_motion(tag):
    """10 log10 of fairlead position variance over position-noise variance, x and z summed."""
    sig, _ = noise_sigma()
    m = fairlead_motion()
    nv = sig[tag].get("x_abs", 0.0) ** 2 + sig[tag].get("z_abs", 0.0) ** 2
    return float("inf") if nv == 0 else 10.0 * np.log10((m["x"] ** 2 + m["z"] ** 2) / nv)


@functools.lru_cache(maxsize=None)
def noise_rows():
    """Per level: Stage 2 and the interpolated baseline at N_in = 4 -> 21, pooled metrics,
    fairlead-peak parity and the snap capture from four sensors."""
    # the job's pass on the PUBLISHED draw must reproduce the shipped model (it does, string for
    # string); the noise-free level below is then its clean draw, the one Section 4 reports.
    ship, clean = kv(S2 / "test_metrics.csv"), kv(T2 / "published" / "test_metrics.csv")
    for k in ("global_R2_tension", "MAE_tension", "MAPE_tension", "temporal_diff_skill_tension",
              "peak_tension_MAE", "peak_tension_underpred_rate_p90",
              "peak_tension_underpred_rate_p99", "contact_touchdown_R2_median"):
        a, b = float(ship[f"test_{k}"]), float(clean[f"test_{k}"])
        assert abs(a - b) <= 1e-9 * max(1.0, abs(a)), (k, a, b)
    out = {}
    for tag, *_ in NOISE_LEVELS:
        d = {int(r["node_count"]): r for r in rows(SN / f"test_metrics_by_node_count{tag}.csv")}
        gb = rows(SN / f"superres_grid_baseline{tag}.csv")
        p = kv(SN / f"test_metrics{tag}.csv")
        z = npload(SN / f"test_timeseries_dump{tag}.npz")
        m, pt, pp = z["meta_all"], z["peak_true_all"], z["peak_pred_all"]
        dev = np.isin(m[:, 0], DEV_SITES)
        ts = tail_split(z)
        out[tag] = dict(
            D={n: {k: float(d[n][k]) for k in ("MAE_tension", "global_R2_tension",
                                                "temporal_diff_skill_tension")} for n in LAYOUTS},
            B={n: {k: grid_cell(gb, n, N_OUT, k) for k in ("MAE_tension", "global_R2_tension",
                                                           "temporal_diff_skill_tension")}
               for n in LAYOUTS},
            pooled={k: float(p[f"test_{k}"]) for k in ("MAE_tension", "temporal_diff_skill_tension",
                                                       "peak_tension_MAE", "MAPE_tension")},
            ratio_all=float(np.median(pp / pt)), ratio_dev=float(np.median(pp[dev] / pt[dev])),
            ratio_ten=float(np.median(pp[m[:, 0] == 10] / pt[m[:, 0] == 10])),
            tail90=ts[(90, "all")], tail99=ts[(99, "all")],
            snap4=snap_stats(snap_events(z, 4)))
    return out


@functools.lru_cache(maxsize=None)
@derived
def noise_bands(tags=("", "_p010", NOISE_INSTR, "_p100")):
    """Where the noise-induced error sits in frequency, on the stored 'typical' windows.

    The 'ty' reservoirs (pooled and per layout) are drawn independently of the model, so
    every level stores the SAME windows (asserted). Per station series: the skill score
    of eq. metrics_skill, the same after a 1 Hz low-pass of the prediction (the solver
    dissipates response above 1/(10 dt) = 1 Hz, Data 2.5), the rms of the prediction
    above 1 Hz, and the correlation of prediction and truth in the wave band
    [1/Tp_max, 1/Tp_min] from the campaign's peak periods.
    """
    from scipy.signal import butter, sosfiltfilt
    fs = 1.0 / SOLVER["dt"]
    tp = [float(npload(c / "env.npy")[2]) for d in sorted(CACHE.iterdir()) for c in sorted(d.iterdir())]
    band = (1.0 / max(tp), 1.0 / min(tp))
    lp = butter(4, 1.0, btype="low", fs=fs, output="sos")
    bp = butter(4, band, btype="band", fs=fs, output="sos")
    skill = lambda t, p: float(np.abs(np.diff(p) - np.diff(t)).mean() / np.abs(np.diff(t)).mean())  # noqa: E731
    out, ref = {}, None
    for tag in tags:
        z = npload(SN / f"test_timeseries_dump{tag}.npz")
        stems, seen = [], set()
        for k in z.files:
            mm = re.match(r"^((?:nc\d+_)?ty\d+)_meta$", k)
            key = tuple(int(v) for v in z[k]) if mm else None
            if mm and key not in seen:
                seen.add(key); stems.append((key, mm.group(1)))
        stems.sort()
        keys = [k for k, _ in stems]
        ref = keys if ref is None else ref
        assert keys == ref, "the ty windows differ between noise levels"
        S, SL, hf, cw = [], [], [], []
        for _k, s in stems:
            tr, pr = z[s + "_true"].astype(np.float64), z[s + "_pred"].astype(np.float64)
            pl = sosfiltfilt(lp, pr, axis=0)
            tb, pb = sosfiltfilt(bp, tr, axis=0), sosfiltfilt(bp, pr, axis=0)
            for i in range(tr.shape[1]):
                if np.abs(np.diff(tr[:, i])).mean() <= 0 or (tb[:, i] ** 2).sum() <= 0:
                    continue
                S.append(skill(tr[:, i], pr[:, i])); SL.append(skill(tr[:, i], pl[:, i]))
                hf.append(float(np.sqrt(((pr[:, i] - pl[:, i]) ** 2).mean())))
                cw.append(float(np.corrcoef(pb[:, i], tb[:, i])[0, 1]))
        out[tag] = dict(win=len(stems), series=len(S), S=statistics.median(S),
                        S_lp=statistics.median(SL), hf=statistics.median(hf),
                        corr=statistics.median(cw))
    return band, out


def build_noise():
    """Section 4.3 prose macros: the noise parameterisation, the four-sensor headline,
    where the advantage lasts, the frequency content of the error, peaks, and the
    differenced-velocity and fixed-offset levels."""
    m = fairlead_motion()
    sig, std = noise_sigma()
    R = noise_rows()
    band, nb = noise_bands()
    I = NOISE_INSTR
    out = [("noiseStillPct", f"{m['still']:.0f}",
            f"stations with zero position std over the record, all {m['n']} finite cached simulations [%]"),
           ("noiseFairX", f"{m['x']:.2f}", "median temporal std of the fairlead's x after the ramp [m]"),
           ("noiseFairZ", f"{m['z']:.2f}", "same, z [m]"),
           ("noiseChanStdX", f"{std['x_abs']:.1f}", "x_abs channel std over the training data, as run [m]"),
           ("noiseChanStdVX", f"{std['x_vel']:.2f}", "x_vel channel std over the training data [m/s]"),
           ("noiseFairVX", f"{m['vx']:.2f}", "median temporal std of the fairlead's x velocity [m/s]"),
           ("noiseSnrTwentyX", f"{sig['_snr20']['x_abs']:.2f}", "sigma_x at a nominal 20 dB [m]"),
           ("noiseSnrTwentyMotion", f"{snr_motion('_snr20'):.1f}",
            "20 dB level expressed against the fairlead's motion [dB]"),
           ("noiseSnrInstr", f"{snr_motion(I):.1f}", "0.05 m level against the fairlead's motion [dB]"),
           ("noiseVelFactor", f"{1.0 / (np.sqrt(2.0) * SOLVER['dt']):.1f}",
            "sigma_v/sigma_p [1/s] when velocity is centrally differenced at dt = 0.1 s")]
    # the four-sensor headline and where the advantage is lost
    d4 = lambda tag, k: R[tag]["D"][4][k]                                # noqa: E731
    b4 = lambda tag, k: R[tag]["B"][4][k]                                # noqa: E731
    wins = [s for tag, _a, _l, s in NOISE_LEVELS if tag in NOISE_ABS
            and d4(tag, "MAE_tension") < b4(tag, "MAE_tension")]
    lost = [s for tag, _a, _l, s in NOISE_LEVELS if tag in NOISE_ABS
            and d4(tag, "MAE_tension") >= b4(tag, "MAE_tension")]
    assert max(wins) < min(lost), "the four-sensor advantage is not lost at a single threshold"
    swept = [s for tag, _a, _l, s in NOISE_LEVELS if tag in NOISE_ABS and s > 0]
    out += [("noiseSigMin", f"{min(swept):.2f}", "smallest absolute white-noise sigma swept [m] (abstract)"),
            ("noiseSigMax", f"{max(swept):.2f}", "largest absolute white-noise sigma swept [m] (abstract)")]
    out += [("noiseRsqFourInstr", f"{d4(I, 'global_R2_tension'):.3f}", "S2 N_in=4 R2 at 0.05 m"),
            ("noiseMAEFourInstr", f"{d4(I, 'MAE_tension'):.1f}", "S2 N_in=4 MAE at 0.05 m [N]"),
            ("noiseMAEFourInstrB", f"{b4(I, 'MAE_tension'):.1f}", "baseline (4,21) MAE at 0.05 m [N]"),
            ("noiseKeepSigma", f"{max(wins):.2f}", "largest absolute sigma at which S2 still beats interpolation at N_in=4 [m]"),
            ("noiseLoseSigma", f"{min(lost):.2f}", "smallest absolute sigma at which it no longer does [m]"),
            ("noiseSnapFourClean", f"{R['']['snap4']['median']:.3f}",
             f"per-event snap capture at N_in=4, clean, n = {R['']['snap4']['n']}"),
            ("noiseSnapFourInstr", f"{R[I]['snap4']['median']:.3f}", "same, 0.05 m"),
            ("noiseFullD", f"{R[I]['D'][N_OUT]['MAE_tension']:.1f}", "S2 N_in=21 MAE at 0.05 m [N]"),
            ("noiseFullB", f"{R[I]['B'][N_OUT]['MAE_tension']:.1f}",
             "baseline (21,21) = the matched Stage-1 model, MAE at 0.05 m [N]"),
            ("noiseSkillFourLowD", f"{d4('_p010', 'temporal_diff_skill_tension'):.2f}", "S2 N_in=4 S at 0.01 m"),
            ("noiseSkillFourLowB", f"{b4('_p010', 'temporal_diff_skill_tension'):.2f}", "baseline (4,21) S at 0.01 m"),
            # the differenced-velocity and fixed-offset paragraph quotes Table 13 only (N_in = 4)
            ("noiseMAEFourClean", f"{d4('', 'MAE_tension'):.1f}", "S2 N_in=4 MAE, clean [N]"),
            ("noiseMAEFourDiff", f"{d4('_coupled050', 'MAE_tension'):.1f}",
             "S2 N_in=4 MAE, velocity differenced from the 0.05 m positions [N]"),
            ("noiseRsqFourDiff", f"{d4('_coupled050', 'global_R2_tension'):.3f}", "same, R2"),
            ("noiseMAEFourOff", f"{d4('_bias050', 'MAE_tension'):.1f}",
             "S2 N_in=4 MAE, fixed per-station offset 0.05 m [N]"),
            ("noiseSkillFourClean", f"{d4('', 'temporal_diff_skill_tension'):.2f}", "S2 N_in=4 S, clean"),
            ("noiseSkillFourInstr", f"{d4(I, 'temporal_diff_skill_tension'):.2f}", "S2 N_in=4 S, white noise 0.05 m"),
            ("noiseSkillFourOff", f"{d4('_bias050', 'temporal_diff_skill_tension'):.2f}",
             "S2 N_in=4 S, fixed offset 0.05 m")]
    # where the advantage lasts: baseline minimum at an intermediate layout (checked)
    for tag in ("_p050", "_p100", "_p250"):
        bmin = min(LAYOUTS, key=lambda n: R[tag]["B"][n]["MAE_tension"])
        assert 4 < bmin < N_OUT and R[tag]["B"][N_OUT]["MAE_tension"] > R[tag]["B"][bmin]["MAE_tension"]
    # the frequency content of the error, on the stored typical windows
    out += [("noiseBandWin", f"{nb['']['win']}", "distinct ty windows stored in every noise dump"),
            ("noiseBandSeries", f"{nb['']['series']}", "station series in them"),
            ("noiseBandLo", f"{band[0]:.3f}".rstrip("0"), "wave band lower edge, 1/Tp_max [Hz]"),
            ("noiseBandHi", f"{band[1]:.2f}", "wave band upper edge, 1/Tp_min [Hz]"),
            ("noiseHFClean", f"{nb['']['hf']:.1f}", "median rms of the prediction above 1 Hz, clean [N]"),
            ("noiseHFInstr", f"{nb[I]['hf']:.0f}", "same, 0.05 m [N]"),
            ("noiseSkillClean", f"{nb['']['S']:.2f}", "median per-series S, clean"),
            ("noiseSkillInstr", f"{nb[I]['S']:.2f}", "same, 0.05 m"),
            ("noiseSkillLpInstr", f"{nb[I]['S_lp']:.2f}", "same, prediction low-passed at 1 Hz"),
            ("noiseCorrClean", f"{nb['']['corr']:.3f}", "median wave-band correlation, clean"),
            ("noiseCorrInstr", f"{nb[I]['corr']:.2f}", "same, 0.05 m"),
            ("noiseCorrTen", f"{nb['_p100']['corr']:.2f}", "same, 0.10 m")]
    # peaks. At the fairlead (always sensed) noise shifts the peaks UP and the tail
    # under-mass falls; over the peaks of every station the bias does not move and the
    # under-mass grows, because the error spreads. Two populations, two answers.
    fl_thr = R[""]["tail90"]["thr"]
    assert R[I]["tail90"]["thr"] == fl_thr                            # same truth, same threshold
    z2 = npload(T2 / "test_timeseries_dump.npz")
    assert abs(float(np.quantile(z2["peak_true_all"], 0.9)) - fl_thr) < 1e-6, "noise clean dump != S2 clean dump"
    st = {tag: kv(SN / f"test_metrics{tag}.csv") for tag in ("", I)}
    st_um = {tag: undermass(float(v["test_peak_tension_MAE_p90"]), float(v["test_peak_tension_bias_p90"]))
             for tag, v in st.items()}
    st_thr = float(st[""]["test_peak_tension_thr_p90"])
    assert st_thr == float(st[I]["test_peak_tension_thr_p90"])
    out += [("noiseRatioClean", f"{R['']['ratio_all']:.3f}", "median fairlead peak pred/true, clean, all windows"),
            ("noiseRatioInstr", f"{R[I]['ratio_all']:.3f}", "same, 0.05 m"),
            ("noiseRatioQuarter", f"{R['_p250']['ratio_all']:.3f}", "same, 0.25 m"),
            ("noiseFlBiasClean", f"{R['']['tail90']['bias']:+.0f}", "fairlead p90 tail mean signed error, clean [N]"),
            ("noiseFlBiasInstr", f"{R[I]['tail90']['bias']:+.0f}", "same, 0.05 m [N]"),
            ("noiseFlUMClean", f"{R['']['tail90']['um']:.0f}", "fairlead p90 tail under-mass, clean [N] (thr = sTwoThrNinety)"),
            ("noiseFlUMInstr", f"{R[I]['tail90']['um']:.0f}", "same, 0.05 m [N]"),
            ("noiseStThrNinety", f"{st_thr:,.0f}".replace(",", "\\,"),
             "p90 of the true per-station window peak, every station, N_out=21 [N]"),
            ("noiseStBiasClean", f"{float(st['']['test_peak_tension_bias_p90']):+.0f}", "that tail's mean signed error, clean [N]"),
            ("noiseStBiasInstr", f"{float(st[I]['test_peak_tension_bias_p90']):+.0f}", "same, 0.05 m [N]"),
            ("noiseStUMClean", f"{st_um['']:.0f}", "that tail's under-mass, clean [N]"),
            ("noiseStUMInstr", f"{st_um[I]:.0f}", "same, 0.05 m [N]")]
    # differenced velocity and the fixed offset, against reported-velocity noise (pooled)
    P_, C_, B_ = R[I]["pooled"], R["_coupled050"]["pooled"], R["_bias050"]["pooled"]
    out += [("noiseDiffMAE", f"{C_['MAE_tension'] / P_['MAE_tension']:.2f}",
             "pooled MAE, differenced / reported velocity, both at 0.05 m"),
            ("noiseDiffPeak", f"{C_['peak_tension_MAE'] / P_['peak_tension_MAE']:.1f}", "same, peak MAE"),
            ("noiseDiffWins", f"{sum(R['_coupled050']['D'][n]['MAE_tension'] < R['_coupled050']['B'][n]['MAE_tension'] for n in LAYOUTS)}",
             "layouts of 10 where S2 beats interpolation with differenced velocity"),
            ("noiseOffMAE", f"{B_['MAE_tension']:.0f}", "pooled MAE, fixed per-station offset 0.05 m [N]"),
            ("noiseInstrMAE", f"{P_['MAE_tension']:.0f}", "pooled MAE, white noise 0.05 m [N]"),
            ("noiseCleanMAE", f"{R['']['pooled']['MAE_tension']:.1f}", "pooled MAE, clean [N]"),
            ("noiseOffSkill", f"{B_['temporal_diff_skill_tension']:.2f}", "pooled S, offset"),
            ("noiseInstrSkill", f"{P_['temporal_diff_skill_tension']:.2f}", "pooled S, white noise 0.05 m"),
            ("noiseCleanSkill", f"{R['']['pooled']['temporal_diff_skill_tension']:.2f}", "pooled S, clean"),
            ("noiseOffDevRatio", f"{R['_bias050']['ratio_dev']:.3f}", "offset, training-site median fairlead peak ratio"),
            ("noiseTenRatioClean", f"{R['']['ratio_ten']:.3f}", "clean, loc10 median fairlead peak ratio")]
    return out


def table_noise():
    """Table 13: Stage 2 and the interpolated baseline from four sensors, per noise level.
    The per-channel SNR levels (arm 'relative') were run but are not reported: Xie et al.'s dB
    convention has no stated reference and does not transfer to along-line positions."""
    R = noise_rows()
    names = {"none": "None", "white": "White", "differenced": "White, velocity differenced",
             "offset": "Fixed offset"}
    s = ("\\begin{tabular}{@{}l l S[table-format=2.1] S[table-format=3.1] S[table-format=3.1]"
         " S[table-format=1.2] S[table-format=1.4] S[table-format=1.4] S[table-format=2.2]"
         " S[table-format=2.2] S[table-format=1.3]@{}}\n\\toprule\n"
         " & & & \\multicolumn{3}{c}{MAE [\\si{\\newton}]} & \\multicolumn{2}{c}{$R^2$}"
         " & \\multicolumn{2}{c}{$S$} & {Snap} \\\\\n"
         "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-11}\n")
    s += _row(["{Error}", "{Level}", "{$\\mathrm{SNR}_f$ [\\si{\\deci\\bel}]}", "{St.~2}", "{Interp.}",
               "{Ratio}", "{St.~2}", "{Interp.}", "{St.~2}", "{Interp.}", "{capture}"])
    s += "\\midrule\n"
    prev = None
    for tag, arm, label, _s in NOISE_LEVELS:
        if arm == "relative":
            continue
        if prev is not None and arm != prev:
            s += "\\addlinespace\n"
        D, B = R[tag]["D"][4], R[tag]["B"][4]
        snr = "{--}" if arm in ("none", "offset") else f"{snr_motion(tag):.1f}"
        s += _row([names[arm] if arm != prev else "", label, snr,
                   f"{D['MAE_tension']:.1f}", f"{B['MAE_tension']:.1f}",
                   f"{D['MAE_tension'] / B['MAE_tension']:.2f}",
                   f"{D['global_R2_tension']:.4f}", f"{B['global_R2_tension']:.4f}",
                   f"{D['temporal_diff_skill_tension']:.2f}", f"{B['temporal_diff_skill_tension']:.2f}",
                   f"{R[tag]['snap4']['median']:.3f}"])
        prev = arm
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


# ------------------------------------------------------------------ Appendix D: the screened windows
# Data 2.5: the fairlead condition (no station other than the fairlead above 1 kN and above the
# fairlead at the same sample) is a snap condition, NOT in the training mask; after training it
# screened the published test draws (EVAL_CLEAN job, patch sr-22). This measures what that removal
# did: each shipped model on its published draw (EVAL_CLEAN/.../published/, which reproduces the
# published metrics) against the same model on the clean draw. Per-entry metrics for BOTH stages
# (per-window sufficient statistics), plus the fairlead peak from the dumps.
WINDOW = 1000


def _field(rs):
    """Pooled field metrics from per-window sufficient statistics, as evaluate() reports them."""
    n = ys = yq = ss = 0.0
    mae, mape, skill = [], [], []
    for r in rs:
        n += float(r["n_pts"]); ys += float(r["y_sum"]); yq += float(r["y_sq"])
        ss += float(r["ss_res"]); mae.append(float(r["mae"]))
        mape.append(float(r["mape"])); skill.append(float(r["td_skill"]))
    return dict(mae=float(np.mean(mae)), r2=1.0 - ss / (yq - ys * ys / n),
                mape=float(np.nanmean(mape)), skill=float(np.nanmean(skill)), n=len(mae))


def _peak(t, p, sel):
    e = p[sel] - t[sel]
    return dict(ratio=float(np.median(p[sel] / t[sel])),
                relerr=100.0 * float(np.mean(np.abs(e) / t[sel])),
                up=100.0 * float((e < 0).mean()), n=int(sel.sum()))


@functools.lru_cache(maxsize=None)
def residue_data():
    out = {}
    for tag, sub, dump_pub in (("s1", "stage1", SF), ("s2", "", S2)):
        pub = rows(EVAL_CLEAN / sub / "published" / "test_per_window_stats.csv")
        cln = rows(EVAL_CLEAN / sub / "test_per_window_stats.csv")
        hit = [eval_dropped(r["lc_id"], r["case_id"], r["start_idx"]) for r in pub]
        # the clean pass must be exactly the published draw minus the screened windows
        key = lambda r: (r["lc_id"], r["case_id"], r["n_in"], r["start_idx"])       # noqa: E731
        assert {key(r) for r, h in zip(pub, hit) if not h} == {key(r) for r in cln}, tag
        fa, fw = _field(pub), _field(cln)
        # and each pooled aggregate must reproduce the test_metrics.csv of its pass
        for d, path in ((fa, EVAL_CLEAN / sub / "published"), (fw, EVAL_CLEAN / sub)):
            ref = kv(path / "test_metrics.csv")
            for k, col in (("mae", "test_MAE_tension"), ("r2", "test_global_R2_tension"),
                           ("mape", "test_MAPE_tension"), ("skill", "test_temporal_diff_skill_tension")):
                assert abs(d[k] - float(ref[col])) <= 1e-5 * max(1.0, abs(float(ref[col]))), \
                    (tag, path.name, k, d[k], ref[col])
        ss_hit = sum(float(r["ss_res"]) for r, h in zip(pub, hit) if h)
        ss_all = sum(float(r["ss_res"]) for r in pub)
        # fairlead peak: the published dump, and the clean dump of the job
        zp = npload(dump_pub / "test_timeseries_dump.npz")
        zc = npload(EVAL_CLEAN / sub / "test_timeseries_dump.npz")
        mp, tp_, pp_ = zp["meta_all"], zp["peak_true_all"], zp["peak_pred_all"]
        mc, tc, pc = zc["meta_all"], zc["peak_true_all"], zc["peak_pred_all"]
        out[tag] = dict(
            f_all=fa, f_wo=fw, n_hit_rows=int(sum(hit)), ss_share=100.0 * ss_hit / ss_all,
            mae_med_hit=float(np.median([float(r["mae"]) for r, h in zip(pub, hit) if h])),
            mae_med_all=float(np.median([float(r["mae"]) for r in pub])),
            pk_all=_peak(tp_, pp_, np.ones(len(tp_), bool)), pk_wo=_peak(tc, pc, np.ones(len(tc), bool)),
            ten=(_peak(tp_, pp_, mp[:, 0] == 10), _peak(tc, pc, mc[:, 0] == 10)))
    ex = [r for r in rows(ROOT / "anchor_outpull_samples.csv")
          if (int(r["loc"]), int(r["case"]), int(r["t"])) == (10, 224, 7603)][0]
    extra = _eval_extra()
    out["cat_samples"] = int(sum(v.size for v in extra.values()))
    out["cat_sims"] = int(sum(1 for v in extra.values() if v.size))
    out["ex"] = ex
    return out


def build_residue():
    d = residue_data()
    a, b = d["s1"], d["s2"]
    return [
        ("resCatSamples", f"{d['cat_samples']:,}".replace(",", "\\,"),
         "samples violating the fairlead condition (k = 1.0, > 1 kN) that the training mask "
         "does not already hold: artifact_samples_cat.csv minus artifact_samples.csv"),
        ("resCatSims", f"{d['cat_sims']}", "simulations holding them"),
        ("resExAnchor", f"{float(d['ex']['anchor_N']) / 1e3:.1f}",
         "loc10/case0224 t=7603, anchor tension [kN]"),
        ("resExFair", f"{float(d['ex']['fairlead_N']) / 1e3:.1f}",
         "same sample, fairlead tension [kN]"),
        ("resSSShareOne", f"{a['ss_share']:.0f}",
         f"percent of the Stage-1 published-draw squared error in the {a['n_hit_rows']} screened "
         "windows (EVAL_CLEAN stage1/published per-window statistics)"),
        ("resSSShareTwo", f"{b['ss_share']:.0f}",
         f"percent of the Stage-2 published-draw squared error in the {b['n_hit_rows']} screened "
         "layout-window pairs"),
        ("resMedMAEHitOne", f"{a['mae_med_hit']:.0f}",
         "median window MAE [N] of the screened Stage-1 windows"),
        ("resMedMAEAllOne", f"{a['mae_med_all']:.0f}", "same over the whole published draw [N]"),
        ("resMedMAEHitTwo", f"{b['mae_med_hit']:.0f}",
         "median window MAE [N] of the screened Stage-2 layout-window pairs"),
        ("resMedMAEAllTwo", f"{b['mae_med_all']:.0f}", "same over all published pairs [N]"),
    ]


def table_residue():
    """Appendix table: each shipped model on its published test draw and on the clean draw."""
    d = residue_data()
    P = "\\mathrm{MAPE}^{\\mathrm{peak}}"
    s = "\\begin{tabular}{@{}l r r@{}}\n\\toprule\n"
    s += _row(["", "{Published draw}", "{After screening}"])
    for lab, key, unit in (("Stage~1, matched resolution", "s1", "Test windows"),
                           ("Stage~2, $N_{\\mathrm{out}} = 21$", "s2", "Layout-window pairs")):
        fa, fw = d[key]["f_all"], d[key]["f_wo"]
        a, w = d[key]["pk_all"], d[key]["pk_wo"]
        s += ("\\midrule\n" if key == "s1" else "\\addlinespace\n") + f"\\multicolumn{{3}}{{@{{}}l}}{{{lab}}} \\\\\n"
        s += _row([f"\\quad {unit}", f"{fa['n']}", f"{fw['n']}"])
        s += _row(["\\quad MAE [\\si{\\newton}]", f"{fa['mae']:.2f}", f"{fw['mae']:.2f}"])
        s += _row(["\\quad $R^{2}$", f"{fa['r2']:.5f}", f"{fw['r2']:.5f}"])
        s += _row(["\\quad MAPE [\\si{\\percent}]", f"{fa['mape']:.2f}", f"{fw['mape']:.2f}"])
        s += _row(["\\quad $S$", f"{fa['skill']:.3f}", f"{fw['skill']:.3f}"])
        s += _row(["\\quad Fairlead $\\tilde{\\rho}$", f"{a['ratio']:.3f}", f"{w['ratio']:.3f}"])
        s += _row([f"\\quad Fairlead ${P}$ [\\si{{\\percent}}]", f"{a['relerr']:.2f}",
                   f"{w['relerr']:.2f}"])
        s += _row(["\\quad Fairlead $u^{\\mathrm{peak}}$ [\\si{\\percent}]", f"{a['up']:.1f}",
                   f"{w['up']:.1f}"])
    s += "\\addlinespace\n\\multicolumn{3}{@{}l}{loc10, fairlead peak} \\\\\n"
    for lab, key in (("Stage~1", "s1"), ("Stage~2", "s2")):
        ta, tw = d[key]["ten"]
        s += _row([f"\\quad {lab}, $\\tilde{{\\rho}}$", f"{ta['ratio']:.3f}", f"{tw['ratio']:.3f}"])
        s += _row([f"\\quad {lab}, ${P}$ [\\si{{\\percent}}]", f"{ta['relerr']:.2f}",
                   f"{tw['relerr']:.2f}"])
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


def table_snapaccounting():
    """Data 2.5, Table 3: snap events reachable under each exclusion policy, by split."""
    names = (("train_val", "Training and validation"), ("test_temporal", "Test, temporal"),
             ("test_ood", "Test, withheld"))
    s = ("\\begin{tabular}{@{}l r r r r@{}}\n\\toprule\n"
         "{Split} & {Present} & {Whole-case} & {Per-window} & {Gain} \\\\\n\\midrule\n")
    for key, lab in names:
        p, b, a = _REACH[key]
        s += _row([lab, f"{p}", f"{b}", f"{a}", f"$+{a - b}$"])
    p, b, a = _REACH["TOTAL"]
    s += "\\midrule\n" + _row(["Total", f"{p}", f"{b}", f"{a}", f"$+{a - b}$"])
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


def _snapsens_rows():
    f = CAT_STUDY / "snap_sens_out.csv"
    if not f.exists():
        sys.exit(f"[numbers] cat_study/{f.name} missing -- run snap_sens_scan.py then "
                 f"snap_sens_out.py in cat_study/")
    return rows(f)


def _snapsens_get(R, code, val):
    def same(a, b):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except ValueError:
            return str(a) == str(b)
    m = [r for r in R if r["code"] == code and same(r["value"], val)]
    assert len(m) == 1, (code, val, len(m))
    return m[0]


def build_snapsens():
    """Data 2.5: what the thresholds of the snap rule rest on.

    Ordinary wave crests at the fairlead, the sensitivity of the catalogue to each condition, and
    whether the exclusion-policy finding (Table 3) survives every variant. Generators:
    cat_study/snap_sens_scan.py (one pass over the archive) and snap_sens_out.py (variants).
    """
    pk = CAT_STUDY / "snap_sens.pkl"
    if not pk.exists():
        sys.exit("[numbers] cat_study/snap_sens.pkl missing -- run snap_sens_scan.py in cat_study/")
    D = pickle.loads(pk.read_bytes())
    R = _snapsens_rows()
    base = _snapsens_get(R, "Tmin", 3)
    n0 = int(base["events"])
    # The relaxed table must reproduce the published catalogue and the accounting of Table 3.
    assert n0 == len(rows(SNAP_CAT)), (n0, len(rows(SNAP_CAT)))
    for code, val in (("iso", 3), ("coh", 3), ("slack", 0.25), ("look", 10), ("cat", 1.0)):
        assert _snapsens_get(R, code, val)["events"] == base["events"], (code, val)
    if _REACH:
        assert tuple(int(base[k]) for k in ("present", "whole", "perwin")) == tuple(_REACH["TOTAL"]), \
            (base, _REACH["TOTAL"])
    old = _snapsens_get(R, "cat", "own1.5")
    assert int(old["events"]) == len(rows(ROOT / "snap_catalogue.csv")), old

    def pct(code, val):
        return 100.0 * (int(_snapsens_get(R, code, val)["events"]) / n0 - 1.0)

    flat = ([pct("coh", v) for v in (1.5, 2, 5, 8)] + [pct("slack", v) for v in (0.1, 0.15, 0.4, 0.5, 0.75)]
            + [pct("look", v) for v in (5, 20)])
    fi = D["fair_iso"].astype(np.float64)
    whole = [100.0 * int(r["whole"]) / int(r["present"]) for r in R]
    perwin = [100.0 * int(r["perwin"]) / int(r["present"]) for r in R]
    return [
        ("snapSensCrestN", f"{len(fi) / 1e6:.1f}",
         f"fairlead crests (strict temporal local maxima) above 3 kN that no artifact rule flags, "
         f"all 3000 simulations = {len(fi)}; snap_sens.pkl"),
        ("snapSensCrestMed", f"{float(np.median(fi)):.2f}",
         f"median temporal isolation ratio of those crests = {float(np.median(fi))!r}"),
        ("snapSensCrestPct", f"{float(np.percentile(fi, 99.9)):.2f}",
         f"99.9th percentile of the same = {float(np.percentile(fi, 99.9))!r}"),
        ("snapSensCrestAbove", f"{int((fi > 3.0).sum())}",
         f"of them with iso > 3 (> 10: {int((fi > 10.0).sum())}, > 1.5: {int((fi > 1.5).sum())})"),
        ("snapSensFlatMax", f"{max(abs(x) for x in flat):.0f}",
         "largest |change| [%] of the event count over coh 1.5-8, slack fraction 0.1-0.75 and "
         f"look-back 5-20 samples = {max(abs(x) for x in flat)!r}"),
        ("snapSensFloorTwo", f"{pct('Tmin', 2):.0f}", "event count change [%] with the floor lowered to 2 kN"),
        ("snapSensFloorFour", f"{-pct('Tmin', 4):.0f}", "event count reduction [%] with the floor raised to 4 kN"),
        ("snapSensIsoTwo", f"{pct('iso', 2):.0f}", "event count change [%] with the isolation bound lowered to 2"),
        ("snapSensIsoFour", f"{-pct('iso', 4):.0f}", "event count reduction [%] with the isolation bound raised to 4"),
        ("snapSensWholeMin", f"{min(whole):.0f}",
         f"smallest share [%] of events reachable under whole-case exclusion over all {len(R)} variants"),
        ("snapSensWholeMax", f"{max(whole):.0f}", "largest share of the same"),
        ("snapSensWinMin", f"{min(perwin):.0f}",
         "smallest share [%] reachable under per-window exclusion over all variants"),
        ("snapSensWinMax", f"{max(perwin):.0f}", "largest share of the same"),
    ]


def table_snapsens():
    """Appendix table: the snap catalogue when one condition of the rule is changed."""
    R = _snapsens_rows()
    n0 = int(_snapsens_get(R, "Tmin", 3)["events"])
    fams = [("Tmin", "Peak tension floor [\\si{\\kilo\\newton}]", (2, 2.5, 3, 4, 5), 3),
            ("iso", "Isolation ratio", (2, 2.5, 3, 4, 5, 7), 3),
            ("coh", "Coherence ratio", (1.5, 2, 3, 5, 8), 3),
            ("slack", "Slack fraction", (0.1, 0.15, 0.25, 0.5, 0.75), 0.25),
            ("look", "Slack look-back [samples]", (5, 10, 20), 10),
            ("cat", "Catenary bound", (1.0, 1.1, 1.2, 1.3, 1.5), 1.0)]

    def cells(r, lab, val):
        ev = int(r["events"])
        chg = 100.0 * (ev / n0 - 1.0)
        c = "0" if abs(chg) < 0.5 else (f"$+{chg:.0f}$" if chg > 0 else f"$-{-chg:.0f}$")
        n = int(r["present"])
        return [lab, val, f"{ev}", c, f"{100.0 * int(r['whole']) / n:.1f}",
                f"{100.0 * int(r['perwin']) / n:.1f}"]

    s = "\\begin{tabular}{@{}l r r r r r@{}}\n\\toprule\n"
    s += _row(["Condition", "{Value}", "{Events}", "{Change [\\si{\\percent}]}",
               "{Whole-case [\\si{\\percent}]}", "{Per-window [\\si{\\percent}]}"])
    s += "\\midrule\n"
    for i, (code, lab, vals, ref) in enumerate(fams):
        if i:
            s += "\\addlinespace\n"
        for j, v in enumerate(vals):
            r = _snapsens_get(R, code, v)
            star = "$^{\\ast}$" if abs(float(v) - float(ref)) < 1e-9 else ""
            s += _row(cells(r, lab if j == 0 else "", f"{float(v):g}{star}"))
    r = _snapsens_get(R, "cat", "own1.5")
    s += "\\addlinespace\n" + _row(cells(r, "Earlier rule", "$\\mathrm{cat}_{i} \\le 1.5$"))
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


# ---------------------------------------------------------------------------------
# Method 3.6: wall-clock cost on the A100, against the solver cost of Data 2.1.
S2_RUN1 = ROOT / "Results" / "SR_RESULTS" / "checkpoints_P2_final_warm"   # Stage-2 run 1 (ep1-53)


def _epoch_minutes(d, lo, hi):
    """Minutes per training epoch from the spacing of two per-epoch checkpoints of one run."""
    a, b = (d / f"checkpoint_epoch{e:04d}.pt" for e in (lo, hi))
    return (b.stat().st_mtime - a.stat().st_mtime) / 60.0 / (hi - lo)


def _test_pass(sub):
    """Minutes and windows of one full test pass of the EVAL_CLEAN job: the job writes the
    published-draw metrics, then scores the screened draw, so the gap between the two
    test_metrics.csv files is one pass (loading, inference and per-window metrics)."""
    t0 = (EVAL_CLEAN / sub / "published" / "test_metrics.csv").stat().st_mtime
    t1 = (EVAL_CLEAN / sub / "test_metrics.csv").stat().st_mtime
    n = len(rows(EVAL_CLEAN / sub / "test_per_window_stats.csv"))
    return (t1 - t0) / 60.0, n


@derived
def build_compute():
    """Training and test wall-clock of both stages (Method 3.6)."""
    import json
    # Stage 1 (v5, job 10532625): no executed notebook was kept, so the epoch time is the
    # spacing of two saved checkpoints; the run stopped at the last epoch of metrics_history.
    m1 = _epoch_minutes(S1, 20, 32)
    n1 = int(rows(S1 / "metrics_history.csv")[-1]["epoch"])
    # Stage 2 (FINAL_WARM): run 1 (job 129246) trained epochs 1..start-1, timed by its checkpoint
    # spacing; run 2 (job 154471) resumed at `start` and prints its own training time.
    m2 = _epoch_minutes(S2_RUN1, 5, 50)
    nb = json.loads(next(S2.glob("*_executed_*.ipynb")).read_text(encoding="utf-8"))
    txt = "".join("".join(o.get("text", "")) for c in nb["cells"] for o in c.get("outputs", []))
    run2_s = float(re.search(r"training took ([0-9.]+) s", txt).group(1))
    start = int(re.search(r"starting at epoch (\d+)", txt).group(1))
    n2 = int(rows(S2 / "metrics_history.csv")[-1]["epoch"])
    h1 = m1 * n1 / 60.0
    h2 = (m2 * (start - 1) * 60.0 + run2_s) / 3600.0
    t1, w1 = _test_pass("stage1")
    t2, w2 = _test_pass("")
    ms1, ms2 = t1 * 60e3 / w1, t2 * 60e3 / w2
    rec = SOLVER["sim_s"] / (REPEAT_W * SOLVER["dt"])          # 100-s windows in one record
    assert 5 < m1 < 9 and 20 < m2 < 32 and 3 < t1 < 20 and 3 < t2 < 20, (m1, m2, t1, t2)
    assert rec * max(ms1, ms2) < 1000.0          # abstract: "a record in under a second"
    return [
        ("sOneTrainHours", f"{h1:.0f}",
         f"{m1!r} min/epoch (checkpoint spacing ep20-ep32) x {n1} epochs = {h1!r} h"),
        ("sTwoTrainHours", f"{h2:.0f}",
         f"run 1: {m2!r} min/epoch (ep5-ep50) x {start - 1}; run 2: {run2_s} s for epochs "
         f"{start}-{n2}; total {h2!r} h"),
        ("sOneTestMin", f"{t1:.0f}", f"Stage-1 screened test pass, {w1} windows, {t1!r} min"),
        ("sTwoTestMin", f"{t2:.0f}", f"Stage-2 screened test pass, {w2} window-layout pairs, "
                                     f"{t2!r} min"),
        ("sOneTestMs", f"{ms1:.0f}", f"{ms1!r} ms per 100-s window, incl. loading and metrics"),
        ("sTwoTestMs", f"{ms2:.0f}", f"{ms2!r} ms per 100-s window, incl. loading and metrics; "
                                     f"a {rec:.1f}-window record takes {rec * ms2 / 1e3:.2f} s"),
    ]


def table_splits():
    """Data 2.4, Table 2: windows per split; test rows as evaluated after the screen of 2.5."""
    def count(d):
        c = collections.Counter("wh" if int(r["lc_id"]) in (10, 11) else "dev"
                                for r in rows(d / "test_per_window_stats.csv"))
        return c["dev"], c["wh"]
    pub1, pub2 = count(T1 / "published"), count(T2 / "published")
    ev1, ev2 = count(T1), count(T2)
    assert sum(pub1) == 50000 and sum(pub1) - sum(ev1) == 487, (pub1, ev1)
    assert sum(pub2) == 50000 and sum(pub2) - sum(ev2) == 470, (pub2, ev2)

    def n(x):
        return f"{x:,}".replace(",", "\\,")
    s = ("\\begin{tabular}{@{}l l l r r@{}}\n\\toprule\n"
         "{Split} & {Sites} & {Time span} & {Stage~1} & {Stage~2} \\\\\n\\midrule\n")
    s += _row(["Training", "8 training", "first \\SI{70}{\\percent}, blocked",
               "\\multicolumn{2}{c}{" + n(1_000_000) + "}"])
    s += _row(["Validation", "8 training", "first \\SI{70}{\\percent}, blocked",
               "\\multicolumn{2}{c}{" + n(10_000) + "}"])
    s += _row(["Test, temporal", "8 training", "last \\SI{30}{\\percent}", n(ev1[0]), n(ev2[0])])
    s += _row(["Test, withheld", "loc10, loc11", "whole record", n(ev1[1]), n(ev2[1])])
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


KENDALL_TERMS = (   # (term name in the notebook, label, symbol of Method 3.5)
    ("mse", "Mean squared error", "L_{\\mathrm{mse}}"),
    ("l1", "Mean absolute error", "L_{\\mathrm{l1}}"),
    ("temporal_grad", "Temporal gradient", "L_{\\mathrm{grad}}"),
    ("spectral", "Spectral", "L_{\\mathrm{spec}}"),
    ("constitutive", "Constitutive", "L_{\\mathrm{const}}"),
    ("catenary", "Catenary", "L_{\\mathrm{cat}}"),
    ("momentum", "Momentum", "L_{\\mathrm{mom}}"),
    ("peak_pinball", "Pinball", "L_{\\mathrm{pin}}"),
)


@functools.lru_cache(maxsize=None)
def shipped_kendall(stage):
    """Epoch and learned log-variances s_k of the shipped model of a stage (1 or 2).

    Read from the criterion state of the shipped checkpoint, the artefact itself, and checked
    against the sigma_k = exp(s_k / 2) that the same run logged at that epoch.
    """
    import torch
    path = {1: S1, 2: S2}[stage]
    ck = torch.load(P.src(path / "best_mooring_gat_lstm_seed_42.pt"), map_location="cpu",
                    weights_only=False)
    ep = int(ck["epoch"])
    lv = [float(v) for v in ck["criterion_state_dict"]["log_vars"]]
    del ck
    names = [t[0] for t in KENDALL_TERMS]
    assert len(lv) == len(names), (len(lv), len(names))
    hist = [r for r in rows(path / "metrics_history.csv") if int(r["epoch"]) == ep]
    assert len(hist) == 1, (path.name, ep)
    for n, s in zip(names, lv):
        logged = float(hist[0]["sigma_" + n])
        assert abs(float(np.exp(0.5 * s)) - logged) <= 1e-5 * logged, (path.name, n, s, logged)
    return ep, dict(zip(names, lv))


def _kendall_order(stage):
    ep, s = shipped_kendall(stage)
    return [t[0] for t in sorted(KENDALL_TERMS, key=lambda t: s[t[0]])]


def _kendall_sigma(stage, name):
    return float(np.exp(0.5 * shipped_kendall(stage)[1][name]))


def _kendall_checks():
    """Guards for the statements Appendix B makes about the two rankings."""
    o1, o2 = _kendall_order(1), _kendall_order(2)
    diff = [i for i in range(len(o1)) if o1[i] != o2[i]]
    # the two stages order the terms alike except that L1 and pinball exchange places
    assert len(diff) == 2 and {o1[i] for i in diff} == {"l1", "peak_pinball"}, (o1, o2)
    # in Stage 2 those two agree to the four significant digits the table shows
    assert (f"{_kendall_sigma(2, 'l1'):#.4g}" == f"{_kendall_sigma(2, 'peak_pinball'):#.4g}"), \
        (_kendall_sigma(2, "l1"), _kendall_sigma(2, "peak_pinball"))
    for stage, o in ((1, o1), (2, o2)):
        # the constitutive term comes last, by a wide margin
        assert o[-1] == "constitutive", (stage, o)
        assert _kendall_sigma(stage, "constitutive") > 100 * _kendall_sigma(stage, o[-2]), stage


def table_kendall(stage):
    """Appendix B: the eight loss terms ranked by their learned scale sigma_k, one stage."""
    _kendall_checks()
    ep, s = shipped_kendall(stage)
    label = {n: (lab, sym) for n, lab, sym in KENDALL_TERMS}
    t = ("\\begin{tabular}{@{}c l l r r@{}}\n\\toprule\n"
         "{Rank} & {Loss term} & {Symbol} & {$s_k$} & {$\\sigma_k$} \\\\\n\\midrule\n")
    for rank, name in enumerate(_kendall_order(stage), start=1):
        lab, sym = label[name]
        t += _row([f"{rank}", lab, f"${sym}$", f"${s[name]:.2f}$",
                   f"{_kendall_sigma(stage, name):#.4g}"])
    t += "\\bottomrule\n\\end{tabular}\n"
    return t


# ------------------------------------------------------------------ Appendix D: linear regression
# Linear_Regression.ipynb, cluster job 885495 (CLAUDE.md 17.3): five least-squares models fitted on
# Stage 1's exact training split and scored with Stage 1's metric code on its screened test draw.
# The job's LR0-LR3 outputs are byte-identical to job 872463's (checked 2026-10-01).
LR = ROOT / "Results" / "LR_baseline"
LR_MODELS = ("LR0_static", "LR1_pointwise", "LR2_line", "LR3_lagged", "LR4_lagged40")
LR_S1 = "Stage 1 (GAT+BiLSTM)"
LR_GROUPS = (("Deeper training sites", (1, 3, 4, 5, 7, 8, 9)),
             ("loc06, shallow training site", (6,)),
             ("loc11, withheld site", (11,)),
             ("loc10, shallow withheld site", (10,)))
_LR = {}


def _lr_data():
    """Summary table, fit report, per-window MAE/S and per-group statistics, all cross-checked."""
    if _LR:
        return _LR
    import json
    summ = {r[""]: {m: float(v) for m, v in r.items() if m}
            for r in rows(LR / "lr_vs_stage1.csv")}
    fit = json.loads(P.read_text(LR / "fit_report.json"))

    # the summary's columns are each model's own test_metrics.csv (Stage 1: Table 6's source)
    cols = (("pooled R2", "test_global_R2_tension"), ("MAE [N]", "test_MAE_tension"),
            ("MAPE [%]", "test_MAPE_tension"),
            ("skill score S (lower = better)", "test_temporal_diff_skill_tension"),
            ("station-peak MAE [N]", "test_peak_tension_MAE"),
            ("station R2, grounded (median)", "test_contact_grounded_R2_median"),
            ("station R2, touchdown (median)", "test_contact_touchdown_R2_median"),
            ("station R2, suspended (median)", "test_contact_suspended_R2_median"))
    for model, d in [(LR_S1, T1)] + [(m, LR / m) for m in LR_MODELS]:
        tm = kv(d / "test_metrics.csv")
        for row, key in cols:
            a, b = summ[row][model], float(tm[key])
            assert abs(a - b) <= 1e-9 * max(1.0, abs(b)), (model, row, a, b)

    # per-window MAE and S of every model, on the identical screened test windows
    pw = {}
    for m, path in [("S1", T1 / "test_per_window_stats.csv")] + \
                   [(m, LR / m / "test_per_window_stats.csv") for m in LR_MODELS]:
        pw[m] = {(int(r["lc_id"]), int(r["case_id"]), int(r["n_in"]), int(r["start_idx"])):
                 (float(r["mae"]), float(r["td_skill"])) for r in rows(path)}
    keys = sorted(pw["S1"])
    assert len(keys) == 49513 and all(set(d) == set(keys) for d in pw.values())
    for m, model in [("S1", LR_S1)] + [(m, m) for m in LR_MODELS]:   # rows rebuild MAE and S
        assert abs(np.mean([pw[m][k][0] for k in keys]) - summ["MAE [N]"][model]) < 1e-6, m
        assert abs(np.mean([pw[m][k][1] for k in keys])
                   - summ["skill score S (lower = better)"][model]) < 1e-9, m

    # fairlead-peak MAPE per group from each model's dump (Eq. metrics_fairlead)
    # the dumps list the same windows in different orders: align each to Stage 1's by window key
    z1 = npload(T1 / "test_timeseries_dump.npz")
    meta, peak_true = z1["meta_all"], z1["peak_true_all"]
    peaks = {"S1": z1["peak_pred_all"]}
    for m in LR_MODELS:
        z = npload(LR / m / "test_timeseries_dump.npz")
        pos = {tuple(r): i for i, r in enumerate(z["meta_all"].tolist())}
        order = np.array([pos[tuple(r)] for r in meta.tolist()])
        assert len(pos) == len(meta) and np.array_equal(z["peak_true_all"][order], peak_true), m
        peaks[m] = z["peak_pred_all"][order]

    def mape_peak(m, sites):
        sel = np.isin(meta[:, 0], sites)
        t, p = peak_true[sel], peaks[m][sel]
        return 100.0 * float(np.mean(np.abs(p - t) / t))

    for m, model in [("S1", LR_S1)] + [(m, m) for m in LR_MODELS]:
        for row, sites in (("fairlead peak, training sites: mean |error| [%]", DEV_SITES),
                           ("fairlead peak, loc10: mean |error| [%]", (10,)),
                           ("fairlead peak, loc11: mean |error| [%]", (11,))):
            assert abs(mape_peak(m, sites) - summ[row][model]) < 1e-6, (m, row)

    groups = {}
    for name, sites in LR_GROUPS:
        ks = [k for k in keys if k[0] in sites]
        g = dict(n=len(ks))
        for m in pw:
            g[m] = dict(mae=float(np.mean([pw[m][k][0] for k in ks])),
                        S=float(np.mean([pw[m][k][1] for k in ks])),
                        worse=100.0 * float(np.mean([pw[m][k][1] > 1.0 for k in ks])),
                        mape_peak=mape_peak(m, sites))
        groups[name] = g
    assert sum(g["n"] for g in groups.values()) == len(keys)
    _LR.update(summ=summ, fit=fit, pw=pw, keys=keys, groups=groups)
    return _LR


def build_linear():
    """Appendix D: the linear ladder. Every number in its prose."""
    d = _lr_data()
    fit, fits, pw, keys = d["fit"], d["fit"]["fits"], d["pw"], d["keys"]
    n_in = {m: sum(1 for _ in rows(LR / m / "coefficients.csv")) - 1 for m in LR_MODELS[1:]}
    assert n_in == {"LR1_pointwise": 34, "LR2_line": 70, "LR3_lagged": 286, "LR4_lagged40": 1618}, n_in
    assert n_in["LR1_pointwise"] == 12 + 22                   # the station's own node features
    dt = SOLVER["dt"]
    lag_short = max(fit["lags"]) * dt
    lag_long = max(fit["lr4_lags"]) * dt
    lag_min = min(abs(x) for x in fit["lags"]) * dt
    assert (round(lag_short, 6), round(lag_long, 6), round(lag_min, 6)) == (5.0, 40.0, 0.1)
    # LR4 = LR3's lags plus every second out to lag_long
    assert {abs(x) for x in fit["lr4_lags"]} == {abs(x) for x in fit["lags"]} | set(range(10, 401, 10))
    sub_pct = 100.0 * fit["lr3_windows"] / fit["train_windows"]
    assert abs(sub_pct - 10.0) < 0.01 and fit["lr3_stride"] == fit["lr4_stride"] == 5
    for m in LR_MODELS[1:]:                       # no ridge penalty lowered the validation error
        assert fits[m]["ridge"] == 0.0, (m, fits[m]["ridge"])
        scan = fits[m].get("lambda_scan")
        if scan:
            mse = [v for _, v in scan]
            assert all(b > a for a, b in zip(mse, mse[1:])), m
        tr, va = fits[m]["train"]["r2"], fits[m]["val"]["r2"]
        assert abs(tr - va) < 1e-3, (m, tr, va)    # no overfitting
    curve = fits["LR4_lagged40"]["memory_curve"]
    rm = {c["span_s"]: c["val_rmse"] for c in curve}
    assert curve[0]["inputs"] == n_in["LR3_lagged"] and curve[-1]["inputs"] == n_in["LR4_lagged40"]
    gain = 100.0 * (rm[lag_short] - rm[lag_long]) / rm[lag_short]
    win_mae = 100.0 * np.mean([pw["S1"][k][0] < pw["LR4_lagged40"][k][0] for k in keys])
    win_s = sum(pw["S1"][k][1] < pw["LR4_lagged40"][k][1] for k in keys)
    assert win_s == len(keys)          # the prose says "in every window"
    for name in ("loc06, shallow training site", "loc10, shallow withheld site"):
        for m in LR_MODELS[1:]:        # the prose says "in most windows" at both shallow sites
            assert d["groups"][name][m]["worse"] > 50.0, (name, m)
    for name in ("Deeper training sites", "loc11, withheld site"):
        for m in LR_MODELS[1:]:        # ... and only there
            assert d["groups"][name][m]["worse"] < 5.0, (name, m)
    r2_const = d["summ"]["pooled R2"]["LR0_static"]
    worse_else = max(d["groups"][name][m]["worse"] for name in ("Deeper training sites",
                     "loc11, withheld site") for m in LR_MODELS[1:])
    mae4, mae1 = d["summ"]["MAE [N]"]["LR4_lagged40"], d["summ"]["MAE [N]"][LR_S1]
    s4, s1 = (d["summ"]["skill score S (lower = better)"][m] for m in ("LR4_lagged40", LR_S1))
    return [
        ("lrTrainWindows", f"{fit['train_windows']:,}".replace(",", "\\,"),
         "fit_report.json train_windows (the Stage-1 training pool)"),
        ("lrSubsetPct", f"{sub_pct:.0f}", f"LR3/LR4: {fit['lr3_windows']} of {fit['train_windows']} windows"),
        ("lrStride", f"{fit['lr3_stride']}", "LR3/LR4 fitted on one sample in lr3_stride"),
        ("lrTestWindows", f"{len(keys):,}".replace(",", "\\,"), "screened Stage-1 test draw"),
        ("lrInOne", f"{n_in['LR1_pointwise']}", "LR1 inputs (coefficients.csv)"),
        ("lrInTwo", f"{n_in['LR2_line']}", "LR2 inputs"),
        ("lrInThree", f"{n_in['LR3_lagged']}", "LR3 inputs"),
        ("lrInFour", f"{n_in['LR4_lagged40']}", "LR4 inputs"),
        ("lrLagMin", f"{lag_min:.1f}", "shortest lag [s]"),
        ("lrLagShort", f"{lag_short:.0f}", "LR3's longest lag [s]"),
        ("lrLagLong", f"{lag_long:.0f}", "LR4's longest lag [s]"),
        ("lrRsqConst", f"{r2_const:.3f}", "LR0 (static pretension held constant) pooled R2"),
        ("lrWinMAE", f"{win_mae:.1f}", f"windows where Stage 1 has a lower MAE than LR4, of {len(keys)}"),
        ("lrMemShort", f"{rm[lag_short]:.1f}", "LR4 memory curve: validation RMSE [N], lags to 5 s"),
        ("lrMemLong", f"{rm[lag_long]:.1f}", "LR4 memory curve: validation RMSE [N], lags to 40 s"),
        ("lrMemGain", f"{gain:.1f}", "relative fall of that RMSE from 5 s to 40 s [%]"),
        ("lrWorseElseMax", f"{worse_else:.1f}",
         "largest share of windows with S_w > 1 for LR1-LR4 at the deeper training sites and loc11 [%]"),
        ("lrMAEFour", f"{mae4:.1f}", f"LR4 test MAE [N] (Stage 1: {mae1:.4f})"),
        ("lrSFour", f"{s4:.2f}", "LR4 test S"),
        ("lrSOne", f"{s1:.2f}", "Stage 1 test S, same table"),
        ("lrMAERatio", f"{mae4 / mae1:.1f}", "LR4 MAE / Stage 1 MAE"),
    ]


def _lr_num(v, fmt):
    s = format(v, fmt)
    return "$-$" + s[1:] if s.startswith("-") else s


def _lr_count(n):
    return f"{n:,}".replace(",", "{,}")


def table_linear():
    """Appendix D table (trimmed 2026-10-02): Stage 1 against LR0-LR4 on the screened draw.
    The full comparison per site group is in the data repository."""
    d = _lr_data()
    summ, groups = d["summ"], d["groups"]
    models = [LR_S1] + list(LR_MODELS)
    keys = ["S1"] + list(LR_MODELS)
    s = "\\begin{tabular}{@{}l r r r r r r@{}}\n\\toprule\n"
    s += _row(["", "Stage~1", "LR0", "LR1", "LR2", "LR3", "LR4"])
    s += "\\midrule\n"
    for label, row, fmt in (("Pooled $R^{2}$", "pooled R2", ".3f"),
                            ("MAE [\\si{\\newton}]", "MAE [N]", ".1f"),
                            ("$S$", "skill score S (lower = better)", ".2f")):
        s += _row([label] + [_lr_num(summ[row][m], fmt) for m in models])
    s += "\\addlinespace\n"
    s += "\\multicolumn{7}{@{}l}{Median station $R^{2}_{i,w}$} \\\\\n"
    for label, row in (("\\quad Grounded", "station R2, grounded (median)"),
                       ("\\quad Touchdown", "station R2, touchdown (median)"),
                       ("\\quad Suspended", "station R2, suspended (median)")):
        s += _row([label] + [_lr_num(summ[row][m], ".2f") for m in models])
    s += "\\addlinespace\n"
    s += "\\multicolumn{7}{@{}l}{Windows with $S_{w} > 1$ [\\si{\\percent}]} \\\\\n"
    for label, name in (("\\quad loc06", "loc06, shallow training site"),
                        ("\\quad loc10", "loc10, shallow withheld site")):
        g = groups[name]
        # LR0 is a constant: S_w = 1 in every window by construction
        s += _row([label] + ["--" if k == "LR0_static" else _lr_num(g[k]["worse"], ".0f")
                             for k in keys])
    s += "\\bottomrule\n\\end{tabular}\n"
    return s


TABLES = [
    ("data_splits.tex", table_splits),
    ("data_snapaccounting.tex", table_snapaccounting),
    ("results_layouts.tex", table_layouts),
    ("results_sites.tex", table_sites),
    ("results_regions.tex", table_regions),
    ("results_tails.tex", table_tails),
    ("results_snaps.tex", table_snaps),
    ("results_snapstation.tex", table_snapstation),
    ("results_sweep.tex", table_sweep),
    ("results_noise.tex", table_noise),
    ("appendix_kendall_s1.tex", functools.partial(table_kendall, 1)),
    ("appendix_kendall_s2.tex", functools.partial(table_kendall, 2)),
    ("appendix_linear.tex", table_linear),
    # appendix_residue.tex and appendix_snapsens.tex are no longer written: Appendices D and G
    # were removed from the manuscript (user, 2026-09-24). table_residue/table_snapsens are kept.
]
TABLE_HEADER = "%% GENERATED by paperA_numbers.py -- do not edit by hand.\n"


MACRO_GROUPS = [
    ("Scope: sites, sensor layouts, dataset", build_scope),
    ("Simulation campaign: forcing, record geometry, solver cost", build_campaign),
    ("Axial wave speed and the frequency limits on a recorded snap peak", build_axial),
    ("Stage 1 -- matched-resolution reconstruction (v5 ep32), test set", build_stage1),
    ("Stage 1 -- contact-regime R2, uniform sample over all nodes", build_stage1_regions),
    ("Stage 1 -- peak-tension tails split by snap membership", build_stage1_calibration),
    ("Full sensing at N_out = 21 -- the Stage-1 anchor, and the withheld-site share",
     build_full_sensing_anchor),
    ("Stage 2 -- headline pooled over all N_in layouts (FINAL_WARM ep76), test set",
     build_stage2),
    ("Stage 2 -- sparse sensing at N_out = 21 (FINAL_WARM ep76), test set",
     build_stage2_sparse),
    ("Stage 2 -- contact-regime R2 at N_out = 21, pooled over N_in", build_stage2_regions),
    ("Snap loads -- reachability and per-event capture", build_snaps),
    ("Curation: artifact mask", build_curation),
    ("Curation: the catenary rule (Data 2.5)", build_catenary),
    ("Wave-train repeat, per-entry metrics of both stages, clean test draws", build_repeat),
    ("Per-site transfer -- the withheld locations", build_sites),
    ("Method: architecture, optimisation and selection", build_architecture),
    ("Method 3.6: training and test wall-clock on the A100", build_compute),
    ("Results 4.1 -- matched resolution: layouts, sites, regions, extremes, snaps",
     build_results_matched),
    ("Results 4.2 -- sparse sensing: sweep, query positions, touchdown, tail, snaps",
     build_results_sparse),
    ("Results 4.3 -- motion-measurement noise (levels of job 236311, clean draw of job 402698)", build_noise),
    ("Results 4.4 -- what the test-window screen removed (published vs screened draw)",
     build_residue),
    ("Appendix D -- linear regression models versus Stage 1 (job 885495)", build_linear),
]

HEADER = """%% =====================================================================
%%  GENERATED FILE -- do not edit by hand.
%%  Produced by paperA_numbers.py (project root) from:
%%    Results/SR_RESULTS/checkpoints_P2_eval_clean/          (EVAL_CLEAN, job 402698: every
%%        test number -- both shipped models on the published draws minus the windows
%%        the evaluation mask covers, the grids and the noise levels)
%%    Results/checkpoints_P1_matched_v5/                     (Stage 1, v5 ep32, training side)
%%    Results/SR_RESULTS/checkpoints_P2_final_warm(run2)/    (Stage 2, ep76, training side)
%%  Regenerate with:  python paperA_numbers.py
%%  Every result number in the prose goes through one of these macros.
%% =====================================================================
"""


def render(allow_stale):
    parts = [HEADER]
    for title, fn in MACRO_GROUPS:
        macros = fn(allow_stale=allow_stale) if fn is build_snaps else fn()
        parts.append("\n%% --- " + title + " " + "-" * max(0, 63 - len(title)) + "\n")
        width = max(len(n) for n, _, _ in macros)
        for name, value, src in macros:
            pad = " " * (width - len(name))
            parts.append(f"\\newcommand{{\\{name}}}{{{value}}}{pad}  % {src}\n")
    return "".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if numbers.tex differs from a fresh render")
    ap.add_argument("--allow-stale", action="store_true",
                    help="keep the recorded reachability if cache_npy is unmounted")
    ap.add_argument("--write-derived", action="store_true",
                    help="development layout: store the results that need non-deposited "
                         "inputs in data/derived_inputs.json (read back when PAPERA_DATA is set)")
    args = ap.parse_args()
    if P.ARCHIVES is not None:
        print(f"[numbers] reading the archives in {P.ARCHIVES}; results that need inputs not "
              f"in them come from {P.rel_out(P.DERIVED_JSON)}"
              + (" (recomputed: PAPERA_RECOMPUTE=1)" if P.RECOMPUTE else ""))

    text = render(args.allow_stale)
    tables = {name: TABLE_HEADER + fn() for name, fn in TABLES}
    if args.check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        stale = [name for name, t in tables.items()
                 if not (TABLES_DIR / name).exists()
                 or (TABLES_DIR / name).read_text(encoding="utf-8") != t]
        if old != text or stale:
            sys.exit("[numbers] OUT OF DATE -- run python paperA_numbers.py "
                     f"(numbers.tex {'stale' if old != text else 'ok'}; tables {stale or 'ok'})")
        print("[numbers] numbers.tex and tables are up to date")
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text, encoding="utf-8")
        TABLES_DIR.mkdir(exist_ok=True)
        for name, t in tables.items():
            (TABLES_DIR / name).write_text(t, encoding="utf-8")
        print(f"[numbers] wrote {P.rel_out(OUT)} -- "
              f"{text.count(chr(92) + 'newcommand')} macros; {len(tables)} tables in "
              f"{P.rel_out(TABLES_DIR)}")
    if args.write_derived:
        keys = P.write_derived("Results of the @derived functions of paperA_numbers.py, computed "
                               "in the development layout from the raw simulations and from "
                               "training-side files that are not in the 4TU archives. "
                               "Regenerate with: python paperA_numbers.py --write-derived")
        print(f"[numbers] stored {len(keys)} derived results in {P.rel_out(P.DERIVED_JSON)}")
