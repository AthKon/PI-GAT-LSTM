#!/bin/bash
#SBATCH --job-name=PUB_FINAL_V5
#SBATCH --partition=gpu-a100
#SBATCH --account=education-me-msc-ode
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/PUB_FINAL_V5_%j.out
#SBATCH --error=logs/PUB_FINAL_V5_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =====================================================================
# Publication Stage-1 FINAL run v5 -- THE LAST STAGE-1 RUN OF THE PAPER.
# Identical to v4 (job 10526752) in EVERY respect except the learning-rate
# schedule, and writing to a NEW checkpoint_dir so v4 is preserved intact.
#
# WHY v5 EXISTS -- the LR-scheduler sign bug (CLAUDE.md 10.9).
#   build_optimizer_and_scheduler feeds ReduceLROnPlateau(mode="min") the
#   score -R2, a NEGATIVE metric, under PyTorch default threshold_mode "rel".
#   The improvement test a < best*(1-thr) then moves the bar the WRONG WAY,
#   so almost every epoch registered as an improvement and reset patience.
#   v4 trained ALL 90 epochs at lr 1e-3 with ZERO decays -- it never had an
#   annealing phase at all. The bug is PROGRESSIVE: the closer R2 gets to 1,
#   the more permissive the bar, so it was silently worsening project-wide.
#   Patch 26 fixes it with threshold_mode="abs" plus min_lr=5e-5. The floor
#   is part of the fix, not a tweak: "abs" with no floor fires ~12 times and
#   collapses lr to 2.4e-7 (a frozen model), because late R2 gains (~5e-5)
#   sit under the 1e-4 bar forever.
#
# RATIFIED BY THE HPO BAKE-OFF (jobs 10529949 abs / 10529950 rel, 2026-07-28).
#   Arms were bit-identical through ep36 and their config.json differed in
#   EXACTLY 2 keys, so it was a clean one-variable experiment. abs won every
#   axis: feasible epochs 18 vs 7; up90 5.17 vs 8.68 %; up99 25.72 vs 35.14 %;
#   under-mass p90 68.6 vs 76.1 N and p99 582.5 vs 645.2 N; skill 0.2732 vs
#   0.2817 (LOWER is better); MAPE 1.189 vs 1.512 %; R2 .999508 vs .999470.
#   max_abs_err went DOWN (11,659 vs 12,666 N), so the over-smoothing risk
#   that motivated running a bake-off at all did NOT materialise.
#   NOTE the control fired 2 decays, not the predicted 0 -- replaying each
#   arm own R2 through the scheduler logic reproduces its actual decay epochs
#   exactly, so that is genuine "rel" behaviour on a noisier HPO trajectory,
#   not a wiring fault.
#
# PATCH 27 (new for this run): the test dump also keeps REPRESENTATIVE snap
#   windows. Measured on the v4 dump, 76 of 108 stored full-array windows
#   contained a catalogued snap -- but every one came from the hi/un tail
#   selectors and ZERO from the median-peak `ty` selector. For a paper whose
#   centrepiece is the snap population, illustrating it only with the most
#   extreme members repeats the very selection bias the exclusion-policy
#   finding is about. `sn` (pooled 8) and `nc{N}_sn` (2 per topology) are
#   reservoir samples over snap-containing windows, and every stored window
#   now carries `_evrows` (snap-event row indices) so the spike can be
#   annotated directly. Dump grows ~10 -> ~15 MB.
#
# Original v4 / GCon rationale, still in force:
# GLOBAL-CONTEXT variant (bake-off winner).
# Same as FINAL v2 in every respect EXCEPT use_global_context=True (patch-17
# line-global readout, the Finding-4 grounded-blind-spot fix) and a fresh
# checkpoint_dir so the shipped non-GC v2 results are never clobbered.
#   winner of jobs 10517068 (GCon) / 10517069 (GCoff), session 2026-07-24:
#   grounded R2 -0.026 -> +0.983, fracneg 59.1% -> 0.9%, global R2 0.995 -> 1.000,
#   MAE -65%, peak MAE -78%, with NO regression on touchdown/suspended/aggregate.
# gat 128/128, lstm 384x2 bidirectional; ALL 10 node counts; full budgets;
# held-out TEST evaluation ON; checkpoint_dir=checkpoints_P1_matched_v5.
# 100 epochs / patience 30 on the skill objective; feasible-only checkpoints.
# SNAP RUN (patch 19): solver artifacts are excluded per WINDOW instead of
#   per CASE. The old 109-case blacklist deleted 1,380 of the 1,611 genuine
#   snap-load events (86%); window-level exclusion recovers them at a cost of
#   0.62% of windows instead of 3.46%. Train pool snap share 0.44% -> 2.17%.
#   Whole-case exclusion is KEPT for the 11 non-finite cases only.
#   snap_oversample_frac=0.0 -> exactly ONE variable changed vs v3_gc.
#   NOTE the test set therefore differs from v3_gc's: it now contains the
#   recovered snap windows, so tail statistics are NOT directly comparable.
#   Run run_pub_EVAL_v3gc.slurm FIRST -- it scores the shipped v3_gc model on
#   this same corrected pipeline in ~1 h and tells you whether this 16 h
#   retrain is needed at all.
# PATCH 22 (v4, the two fixes job 10521852 needed):
#   1  the held-out-location split branch built list(range(len(ds))) and never
#      called the artifact filter, so ALL loc10/loc11 test windows entered
#      unfiltered (peaks to 943 MN -> p99 thresholds destroyed -> 0 feasible
#      epochs in 100). It now goes through the same filter as the dev path.
#   2  a 50 kN ceiling closes the shape rule's sustained-divergence blind spot
#      (it ANDs iso > 10, so a diverged plateau is invisible to it). 638
#      corrupt samples escaped it, the largest at 1.3e18 N inside a window
#      shape masking alone would have KEPT. The ceiling fires in 36 cases, ALL
#      already flagged -- it implicates ZERO new simulations.
#   Verified locally (verify_patch22.py, full cache): worst tension in a window
#   any split would build is 49,483 (train/val) / 49,185 (test-time) / 47,818
#   (held-out loc) N. Snap events reachable 231 -> 1030 (4.46x vs v3_gc).
# PATCH 23: the test dump records window start indices + per-snap-event capture
#   (line max at the event timestep), so the snap question is answerable from
#   the .npz alone.
# The notebook carries a single FIXED FINAL config (cell a04f1f5f, no _RUN
# toggle); HARD GATE 2 below refuses to run unless that config -- AND the
# global-context flag -- is in place.
# Walltime: v4 did 90 ep in 10:43 (~7.15 min/ep), so 100 ep ~ 11.9 h. The fix
# gives MORE feasible epochs (bake-off: 18/60 vs 7/60) => more checkpoints =>
# a longer per-checkpoint-dump tail (~30 x 40 s ~ 20 min), and annealing makes
# early stopping LESS likely to fire, so budget for the full 100 epochs:
# ~12.5 h expected. The 20 h cap below covers a slow node (s7 precedent x1.37).
# Historical: v2 finished all 100 ep in 14:55; global context adds ~4-6% compute
# (HPO bake-off: GCon +4.5% over GCoff) + a longer per-checkpoint-dump tail (GCon
# is more feasible -> more checkpoints) -> ~16-16.5 h expected. 20 h covers a
# slow node (s7 precedent: x1.37 per-epoch -> ~21 h worst case is the only tail
# not fully covered). AUTO-RESUME makes even a timeout cheap: the run cell scans
# checkpoint_dir on startup and continues from the newest valid checkpoint, so a
# plain resubmit of THIS SAME script picks up where it stopped (no manual edit).
# Derived from run_pub_FINAL_v4.slurm (which came from v2).
# =====================================================================

set -u

TAG=FINAL_V5
NOTEBOOK=GAT_LSTM_NN_Publication.ipynb
CKPT_DIR=checkpoints_P1_matched_v5
VENV_PY=/scratch/tkonstantaras/venvs/pyg_env/bin/python
SRC_CACHE=/scratch/tkonstantaras/mooring_data/cache_npy

die() { echo "[FATAL] $1" >&2; echo "[exit] job aborted -- resubmit (likely a different node)"; exit 1; }

echo "[job] tag=${TAG} jobid=${SLURM_JOB_ID} node=$(hostname) started $(date)"
cd "${SLURM_SUBMIT_DIR}"

# --- environment (order matters; venv relies on py-torch module) -----
module load 2024r1
module load openmpi
module load py-torch
source /scratch/tkonstantaras/venvs/pyg_env/bin/activate
echo "[env] python: $(which python)"
nvidia-smi || true

# --- thread limiting: 8 DataLoader workers on 16 cores must NOT each spawn a
# full BLAS pool. Pin every worker's math libs to 1 thread.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR=/scratch/tkonstantaras/tmp
mkdir -p "${TMPDIR}" logs

# --- HARD GATE 1: GPU + core imports MUST be OK before staging 13 GB / running.
# The job is UNATTENDED, so a CPU-only node or a broken venv has to kill it in
# the first seconds -- never silently train on CPU for hours.
"${VENV_PY}" - <<'PY' || die "env/GPU preflight failed (see [diag] lines) -- node has no usable GPU or a broken venv"
import sys
print("[diag] py:", sys.executable)
import torch
print("[diag] torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch.cuda.is_available() is False -- no GPU visible on this node"
print("[diag] gpu:", torch.cuda.get_device_name(0))
import numpy, torch_geometric, nbconvert
print("[diag] numpy:", numpy.__version__, "pyg:", torch_geometric.__version__, "nbconvert:", nbconvert.__version__)
from torch_geometric.nn import GATv2Conv
print("[diag] all imports OK -- proceeding to config gate")
PY

# --- HARD GATE 2: the notebook MUST carry the FINAL config AND global context --
# FINAL ships a single fixed config in cell a04f1f5f (no _RUN toggle). A stale
# HPO cell, or the non-GC v2 config, would silently train the wrong model into
# the wrong dir. Verify every FINAL literal + use_global_context=True is present,
# and that the dataset build-CALL is on all 10 node counts, before staging.
"${VENV_PY}" - "${NOTEBOOK}" "${CKPT_DIR}" <<'PY' || die "notebook is NOT the v5 FINAL config -- apply _nb_patch_pub_27_snapwindows.py then _nb_patch_pub_28_v5run.py before uploading"
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
ckpt = sys.argv[2]
cells = {c.get("id"): c for c in nb["cells"]}
cfg_src = "".join(cells["a04f1f5f"]["source"])
need = [
    'checkpoint_dir="./%s"' % ckpt,
    '_RESUME_SCAN_DIR = "./%s"' % ckpt,   # auto-resume MUST point at THIS run
    'use_global_context=True',
    # patch 26/28: the LR fix, explicit in the config (not just the default)
    'scheduler_threshold_mode="abs"',
    'scheduler_threshold=1e-4',
    'scheduler_min_lr=5e-5',
    'run_test_evaluation=True',
    'lstm_bidirectional=True',
    'num_lstm_layers=2',
    'lstm_hidden_dim=384',
    'gat_hidden_dim=128',
    'node_counts=(4, 5, 6, 7, 8, 10, 12, 15, 18, 21)',
    'save_feasible_checkpoints=True',
    'sel_up90_cap=10.0',
    'sel_abort_epoch=20,',   # trailing comma => the ASSIGNMENT, not the comment above it
    'save_every_n_epochs=0',
    'num_epochs=100',
    'early_stopping_patience=30',
    'seed_list=(42,)',
    '_pick_resume_checkpoint',              # auto-resume machinery present
    'resume_from_checkpoint=_resume_path',  # ...and threaded into the config
]
miss = [n for n in need if n not in cfg_src]
assert not miss, "GCon-FINAL cfg literals missing from cell a04f1f5f: %r" % miss
assert 'checkpoint_dir="./checkpoints_P1_matched_v2"' not in cfg_src, "still writing into the v2 dir -- re-apply _nb_patch_pub_final_gcon.py"
# v5 must NOT inherit v4's directory. This is not cosmetic: the run cell
# auto-resumes from the newest valid checkpoint in _RESUME_SCAN_DIR, so a v4
# path here would silently CONTINUE job 10526752 instead of starting fresh,
# and would overwrite v4's results in the process.
assert "checkpoints_P1_matched_v4" not in cfg_src, \
    "run cell still references the v4 checkpoint dir -- re-apply _nb_patch_pub_28_v5run.py"
# patch 26 must be live in the SCHEDULER, not merely present in the config
_bo = "".join(cells["7a1e407c"]["source"])
assert 'threshold_mode=getattr(cfg, "scheduler_threshold_mode"' in _bo, \
    "ReduceLROnPlateau does not read scheduler_threshold_mode -- re-apply _nb_patch_pub_26_lr_threshold.py"
assert 'min_lr=getattr(cfg, "scheduler_min_lr"' in _bo, \
    "ReduceLROnPlateau has no min_lr floor -- re-apply patch 26"
_tc = "".join(cells["c53b5783"]["source"])
assert 'scheduler_threshold_mode: str = "abs"' in _tc, \
    "TrainingConfig default threshold_mode is not abs -- re-apply patch 26"
assert "scheduler_min_lr: float = 5e-5" in _tc, \
    "TrainingConfig has no scheduler_min_lr -- re-apply patch 26"
assert "_RUN = dict(" not in cfg_src, "stale round-3 _RUN toggle still active in cell a04f1f5f"
assert "sel_bias90_cap=" not in cfg_src, "patch-09f bias box still set in a04f1f5f -- re-apply _nb_patch_pub_12_final_v2.py"
# patch 25 (selection v4): the p99 rate cap must be GONE from the config. It was
# unreachable on the corrected pipeline (p99 tail is 43-82% snap-composed, snap
# magnitude aliasing-bounded at 0.1 s) and produced 0 feasible epochs twice.
assert "sel_up99_cap" not in cfg_src, \
    "p99 rate cap STILL in a04f1f5f -- re-apply _nb_patch_pub_25_selection_v4.py"
# the global-context flag must actually be threaded through the encoder
enc = "".join(cells["bcd95251"]["source"])
assert "self.global_mlp" in enc and "use_global_context" in enc, "encoder is not patch-17 aware -- re-apply _nb_patch_pub_17_global_context.py"
bm = "".join(cells["7a1e407c"]["source"])
assert "use_global_context=cfg.use_global_context" in bm, "build_model does not thread use_global_context -- re-apply patch 17"
# the rate-only Gate B must actually be live in the train loop
tl = "".join(cells["c62bc05c"]["source"])
assert "cfg.sel_up90_cap" in tl, "train loop has no p90 rate gate -- re-apply _nb_patch_pub_10_selection_v2.py"
assert "cfg.sel_up99_cap" not in tl, \
    "train loop STILL gates on the p99 rate cap -- re-apply _nb_patch_pub_25_selection_v4.py"
# patch 25: the hard abort must be live, or an infeasible run burns the full
# walltime and then ships the safety-blind 10-R2 fallback (jobs 10466291/10521852).
assert "sel_abort_epoch" in tl and "[ABORT] no feasible epoch by epoch" in tl, \
    "hard abort missing from the train loop -- re-apply _nb_patch_pub_25_selection_v4.py"
assert "0.0 < _sb90 <= cfg.sel_bias90_cap" not in tl, "train loop STILL enforces the infeasible bias box -- re-apply _nb_patch_pub_10_selection_v2.py"
# contact-regime metric (patch 16) must be logged to metrics_history.csv
assert "val_contact_grounded_R2_median" in tl, "grounded-R2 metric not wired into the CSV -- re-apply _nb_patch_pub_16_contact_regime_r2.py"
ev = "".join(cells["8f268a8b"]["source"])
assert "compute_contact_regime_metrics" in ev, "evaluate() has no contact-regime metric -- re-apply patch 16"
# corrupt blacklist must be the 109-case union (patch 18)
bl = "".join(cells["f80f8d84"]["source"])
assert "assert len(CORRUPT_CASES_ALL) == 109" in bl, "blacklist is not the 109-case union -- re-apply _nb_patch_pub_18_blacklist_109.py"
# patch 19: artifacts excluded per WINDOW, not per case; whole-case drop kept
# only for the non-finite cases, which would otherwise poison the normalizer fit
assert "EXCLUDE_ARTIFACT_WINDOWS = True" in bl, "per-window artifact exclusion is OFF -- re-apply _nb_patch_pub_19_window_exclusion.py"
assert "assert len(CORRUPT_CASES_NONFINITE) == 11" in bl, "non-finite case set missing -- re-apply patch 19"
assert "ARTIFACT_SAMPLES" in bl and "SNAP_EVENT_TIMES" in bl, "artifact/snap catalogues missing -- re-apply patch 19"
# patch 21: FE runs that diverged and TERMINATED EARLY are shorter than one
# window, so per-window masking cannot salvage them -- they must stay whole-case
# drops even under EXCLUDE_ARTIFACT_WINDOWS. Job 10521328 died on exactly this.
assert "CORRUPT_CASES_TOO_SHORT" in bl, "too-short case set missing -- re-apply _nb_patch_pub_21_short_cases.py"
assert "(CORRUPT_CASES_NONFINITE | CORRUPT_CASES_TOO_SHORT)" in bl, \
    "too-short cases are NOT unioned into the whole-case drop set -- re-apply patch 21"
assert "assert len(CORRUPT_CASES_TOO_SHORT) == 2" in bl, "too-short set is not the measured 2 cases -- re-apply patch 21"
sp = "".join(cells["858f4194"]["source"])
assert "ARTIFACT_SAMPLES.get(key)" in sp, "window filter not wired into valid_window_starts_inside_raw_interval -- re-apply patch 19"
# patch 22 fix 1: the held-out-location branch must be FILTERED. Job 10521852
# wasted 13 h because it was not: list(range(len(ds))) bypassed exclusion for
# every loc10/loc11 window, putting 943 MN peaks in the test set.
assert "all_wins = list(range(len(ds)))" not in sp, \
    "OOD BYPASS PRESENT -- re-apply _nb_patch_pub_22_ceiling_and_ood.py"
assert sp.count("all_wins = valid_window_starts_inside_raw_interval(") == 1, \
    "held-out branch does not call the artifact filter exactly once -- re-apply patch 22"
# patch 22 fix 2: the 50 kN ceiling, and the sample lists it generates
assert "ARTIFACT_TENSION_CEILING_N = 50_000.0" in bl, \
    "50 kN ceiling missing -- re-apply _nb_patch_pub_22_ceiling_and_ood.py"
_ns = {}
_lo = bl.index("ARTIFACT_SAMPLES = {")
exec(bl[_lo:bl.index("\n}\n", _lo) + 3], _ns)
_A = _ns["ARTIFACT_SAMPLES"]
assert len(_A) == 109, "ARTIFACT_SAMPLES covers %d cases, expected 109" % len(_A)
_nsamp = sum(len(v) for v in _A.values())
assert _nsamp == 1643, \
    "ARTIFACT_SAMPLES has %d samples, expected 1643 (996 shape + 638 ceiling + 9 non-finite)" % _nsamp
# patch 23: the dump must be able to answer the snap question
_ts = "".join(cells["pubts01"]["source"])
assert "snap_ev" in _ts and "int(starts[k])))" in _ts, \
    "snap capture / window start missing -- re-apply _nb_patch_pub_23_snap_capture.py"
assert "_snap_enriched_sample(" in sp, "snap-aware train sampling missing -- re-apply patch 19"
assert "snap_oversample_frac=0.0" in cfg_src, "snap_oversample_frac is not the explicit 0.0 baseline -- re-apply _nb_patch_pub_20_snaprun.py"
# the time-series dump must be wired in
assert any(c.get("id") == "pubts01" for c in nb["cells"]), "ts-dump cell pubts01 missing -- re-apply _nb_patch_pub_11_tsdump.py"
assert "dump_test_timeseries(" in cfg_src, "run cell does not call dump_test_timeseries -- re-apply _nb_patch_pub_11_tsdump.py"
# patch 14: per-TOPOLOGY tail safety must reach both per-node-count CSVs
assert "nc_peak_signed" in ev, "per-topology tail accumulators missing -- re-apply _nb_patch_pub_14_per_nc_tailsafety.py"
assert "peak_tension_undermag_p99" in tl, "per-nc tail columns missing from metrics_by_node_count.csv -- re-apply patch 14"
tm = "".join(cells["d75ee9f2"]["source"])
assert "peak_tension_undermag_p99" in tm, "per-nc tail columns missing from test_metrics_by_node_count.csv -- re-apply patch 14"
# patch 15: per-TOPOLOGY time-series windows in the dump
ts = "".join(cells["pubts01"]["source"])
assert "n_nc_extreme" in ts, "per-topology window selection missing -- re-apply _nb_patch_pub_15_per_nc_timeseries.py"
assert "pin_sampler_epoch" in ts, "sampler pin missing -- per-checkpoint dumps would be incomparable"
# patch 27: representative snap windows + per-window snap-event rows. Without
# these the only snap TRACES in the dump come from the hi/un tail selectors.
assert '_store("sn", sorted(snap_res' in ts, \
    "representative snap-window selector missing -- re-apply _nb_patch_pub_27_snapwindows.py"
assert '_store(f"nc{_nc}_sn"' in ts, \
    "per-topology snap-window selector missing -- re-apply patch 27"
assert "n_snap: int = 8," in ts and "n_nc_snap: int = 2," in ts, \
    "snap-window budgets missing -- re-apply patch 27"
assert '_evrows"] = p.get(' in ts, \
    "snap-event rows are not stored per window -- re-apply patch 27"
# per-checkpoint dumps must stay ON, or only the selected epoch gets a .npz
# and no candidate epoch can be inspected locally after the run
assert "ts_dump_all_checkpoints:      bool = True" in _tc, \
    "per-checkpoint dumps are OFF -- every feasible epoch must get its own .npz"
assert "ts_dump_all_checkpoints" in cfg_src, \
    "run cell does not gate the per-checkpoint dump loop -- re-apply _nb_patch_pub_13_perckpt_dumps.py"
# dataset build-CALL must also be on all 10 node counts
fb = "".join(cells["fbac801f"]["source"])
i = fb.index("load_case_datasets = build_lazy_load_case_datasets(")
assert "node_counts=(4, 5, 6, 7, 8, 10, 12, 15, 18, 21)" in fb[i:i + 500], \
    "dataset build-call node_counts is NOT all-10 -- re-apply _nb_patch_pub_final.py"
# patch 21: the builder must SKIP a too-short case, not raise inside the dataset
# constructor -- an unattended 20 h job must not die on a truncated record
assert "_short_skipped" in fb, "builder has no too-short guard -- re-apply _nb_patch_pub_21_short_cases.py"
print("[sel] v5 FINAL verified: LR fix (threshold_mode=abs, min_lr=5e-5) explicit "
      "AND live in the scheduler, use_global_context=True, PC1 dims, 10 node counts, "
      "test ON, selection v4 (p90 gate only + abort@20), patch 16/17/18/21/22/23/25/27 "
      "wired, OOD path FILTERED, ceiling 50 kN / 109 cases / 1643 samples ->", ckpt)
PY

# --- HARD GATE 3: record-length preflight (patch 21) ------------------
# Job 10521328 spent 23 min staging 13 GB and then died 46 s into the notebook
# because two cached cases are shorter than one window. Reading the .npy HEADERS
# of every in-use case costs seconds and catches that class of defect BEFORE the
# stage, so a bad blacklist fails in the first minute instead of the 25th.
"${VENV_PY}" - "${NOTEBOOK}" "${SRC_CACHE}" <<'PY' || die "record-length preflight failed -- a non-excluded case is shorter than window_len (see [len] lines)"
import ast, json, pathlib, re, sys
import numpy.lib.format as nf

nb = json.load(open(sys.argv[1], encoding="utf-8"))
cache = pathlib.Path(sys.argv[2])
cells = {c.get("id"): c for c in nb["cells"]}

# evaluate the plain-literal assignments of the constants cell (frozensets,
# dicts, AVAILABLE_LOC_IDS, N_CASES); anything needing imports is skipped
ns = {}
for node in ast.parse("".join(cells["f80f8d84"]["source"])).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        try:
            ns[node.targets[0].id] = eval(
                compile(ast.Expression(node.value), "<bl>", "eval"),
                {"frozenset": frozenset}, ns)
        except Exception:
            pass
drop = ns["CORRUPT_CASES"]
locs = ns["AVAILABLE_LOC_IDS"]
n_cases = ns["N_CASES"]
W = int(re.search(r"^WINDOW_LEN\s*=\s*(\d+)", "".join(cells["fbac801f"]["source"]), re.M).group(1))
print(f"[len] window_len={W} | cases dropped whole={len(drop)} | scanning {len(locs)}x{n_cases} headers")

def steps(p):
    with open(p, "rb") as fh:
        ver = nf.read_magic(fh)
        shape, _, _ = (nf.read_array_header_1_0(fh) if ver == (1, 0)
                       else nf.read_array_header_2_0(fh))
    return shape[0]

bad, missing, seen = [], [], 0
for loc in locs:
    for case in range(1, n_cases + 1):
        if (loc, case) in drop:
            continue
        f = cache / f"loc{loc:02d}" / f"case_{case:04d}" / "x_abs.npy"
        if not f.exists():
            missing.append((loc, case)); continue
        n = steps(f); seen += 1
        if n < W:
            bad.append((loc, case, n))
assert not missing, f"[len] cache missing for {len(missing)} case(s), e.g. {missing[:5]}"
if bad:
    for loc, case, n in bad:
        print(f"[len] TOO SHORT: loc{loc:02d} case{case:04d} = {n} steps < {W}")
assert not bad, (f"{len(bad)} non-excluded case(s) are shorter than window_len={W} -- "
                 "add them to CORRUPT_CASES_TOO_SHORT (cell f80f8d84)")
print(f"[len] OK: all {seen} in-use cases have >= {W} steps")
PY

# --- stage the ~13 GB npy cache to node-local /tmp -------------------
# Removes BeeGFS from the TRAINING read path (the C1 freeze fix). No BeeGFS
# fallback: staging is guaranteed or the job dies now.
LOCAL_ROOT=/tmp/mooring_${SLURM_JOB_ID}
LOCAL_CACHE=${LOCAL_ROOT}/cache_npy
trap 'echo "[stage] cleanup ${LOCAL_ROOT}"; rm -rf "${LOCAL_ROOT}"' EXIT

# 0. capacity pre-check BEFORE the ~5 min copy: need cache size + 2 GB margin.
NEED_KB=$(( $(du -sk "${SRC_CACHE}" | cut -f1) + 2*1024*1024 ))
FREE_KB=$(df -k --output=avail /tmp | tail -1 | tr -d ' ')
echo "[stage] /tmp free ${FREE_KB} KB, need ${NEED_KB} KB"
[ "${FREE_KB}" -ge "${NEED_KB}" ] || die "/tmp too small for the cache (free ${FREE_KB} < need ${NEED_KB} KB)"

# 1. copy (rsync self-verifies), then canary-verify by file count.
echo "[stage] copying ${SRC_CACHE} -> ${LOCAL_CACHE} ($(date))"
mkdir -p "${LOCAL_CACHE}"
rsync -a "${SRC_CACHE}/" "${LOCAL_CACHE}/" || die "rsync failed"
SRC_N=$(find "${SRC_CACHE}"   -type f | wc -l)
DST_N=$(find "${LOCAL_CACHE}" -type f | wc -l)
[ "${DST_N}" -gt 0 ]           || die "canary: 0 files staged"
[ "${SRC_N}" -eq "${DST_N}" ]  || die "canary: file count ${SRC_N} != ${DST_N}"

# 2. training reads ONLY node-local from here on.
export MOORING_CACHE_DIR="${LOCAL_CACHE}"
echo "[stage] OK: ${DST_N} files staged ($(date))"
echo "[stage] MOORING_CACHE_DIR=${MOORING_CACHE_DIR}"

# --- run --------------------------------------------------------------
mkdir -p "${CKPT_DIR}"

echo "[run] executing notebook ($(date)) -- stdout goes into the executed"
echo "[run] .ipynb, NOT this .out; monitor ${CKPT_DIR}/metrics_history.csv"
srun "${VENV_PY}" -m nbconvert --to notebook --execute \
    "${NOTEBOOK}" \
    --output "GAT_LSTM_NN_Publication_${TAG}_executed_${SLURM_JOB_ID}.ipynb" \
    --ExecutePreprocessor.timeout=-1
RC=$?

echo "[job] nbconvert exit code ${RC}; finished $(date)"
exit ${RC}
