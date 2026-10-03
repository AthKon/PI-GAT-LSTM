#!/bin/bash
#SBATCH --job-name=PUB_EVAL_V5_SNAPFIX
#SBATCH --partition=gpu-a100
#SBATCH --account=education-me-msc-ode
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/PUB_EVAL_V5_SNAPFIX_%j.out
#SBATCH --error=logs/PUB_EVAL_V5_SNAPFIX_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =====================================================================
# STAGE-1 CLOSING JOB. EVAL-ONLY: re-score the SHIPPED v5 checkpoint
# (job 10532625, epoch 32) against the CORRECTED snap catalogue.
# NO TRAINING, NO CHECKPOINT WRITTEN. ~1 h, not ~12.5 h.
#
# THE DATA PIPELINE IS UNCHANGED FROM v5. ARTIFACT_SAMPLES (the per-sample
# exclusion mask), CORRUPT_CASES*, the 50 kN ceiling and the split builder are
# all byte-identical, and snap_oversample_frac stays 0.0 exactly as in v5, so
# SNAP_EVENT_TIMES never reaches window selection. The 50k test draw is
# therefore bit-identical to v5's and EVERY AGGREGATE METRIC MUST REPRODUCE --
# the run cell asserts three of them against v5's reported values and prints a
# loud DIFF if the population moved. That is this job's integrity check.
#
# WHAT ACTUALLY CHANGES: snap_ev / snap_true / snap_pred inside
# test_timeseries_dump.npz, i.e. which physical events are scored.
#
#   defect 1 -- ARTIFACT LEAK (CLAUDE.md 12.1). A whole-line one-sample
#     artifact whose profile DECREASES anchor -> fairlead is spatially smooth,
#     so the artifact rule's coh branch cannot fire, and its
#     T > 3 x fairlead_max branch is dead when the case's own fairlead_max is
#     itself contaminated. The per-node snap test then catalogued the ONE node
#     where the ramp crossed both T > 3 kN and fair <= 1.5.
#     Worked example: loc10/case0224 t=7603, a 24.2 kN artifact catalogued as a
#     4.1 kN "snap" at node 14. Guard A (the anchor may not out-pull the
#     fairlead by more than 3x) removes it. REMOVAL-only.
#
#   defect 2 -- REFRACTORY GROUPING KEPT THE PRECURSOR (CLAUDE.md 12.8). The
#     2.0 s window correctly groups a snap with its own aliased ringdown, but
#     the loop kept the FIRST passing timestep rather than the largest. In 274
#     of 1611 groups (17.0 %) a dropped timestep is >5 % larger than the kept
#     one (p50 1.56x, p90 5.49x, max 939x). loc04/case0135 kept 4.4 kN at
#     t=9773 and threw away the 16.8 kN main event at t=9783.
#     THIS IS WHY A JOB IS NEEDED: it MOVES events. 261 of the 1388 corrected
#     events sit at a timestep for which no prediction was ever stored, so
#     they cannot be recovered by re-filtering the existing dump.
#
#   defect 3 -- falls out of fixing 2. Taking a group's argmax inside a
#     diverged case selects the divergence, which would have RAISED the count
#     of catalogued events above the pipeline's own 50 kN ceiling from 27 to
#     36. Guard C applies that same ceiling (and a non-finite veto) to the
#     catalogue, so it can never name a sample the MASK excludes from every
#     built window. Not a new criterion -- the mask's own RULE M.
#
#   Net, measured over the full 41.3 M-sample cache:
#     1611 -> 1388 events | max peak 1.08 GN -> 44.1 kN | above 50 kN 27 -> 0
#     per-event peak p99 301 kN -> 29.0 kN | loc11/case0296 (the canonical
#     clean 20.4 kN snap) survives unchanged.
#
# ALSO SHIPPED: patch 29 event de-duplication in the dump (CLAUDE.md 13.5).
# Any window start in [t-999, t] covers a peak at sample t, so one FE event was
# being drawn at many offsets and topologies and the bounded heaps ranked
# WINDOWS -- the v5 dump's 8-panel highest-load grid showed 2 distinct events
# eight times. hi/un now rank DISTINCT events, keeping the window whose peak
# sits furthest from either edge (the model is bidirectional, so an event at a
# window boundary loses one side of its context).
#
# Prerequisites in the submission directory:
#   GAT_LSTM_NN_Publication_EVAL.ipynb          (built by
#                                                _nb_patch_pub_29_eval_snapfix.py)
#   checkpoints_P1_matched_v5/best_mooring_gat_lstm_seed_42.pt
#   this script
# The main notebook GAT_LSTM_NN_Publication.ipynb is NOT used and NOT touched.
#
# Walltime: staging ~23 min + build_dataloaders/normalizer fit ~15 min + a
# 50k-window forward-only test pass + ts-dump ~10 min. 3 h is generous.
# =====================================================================

set -u

TAG=EVAL_V5_SNAPFIX
NOTEBOOK=GAT_LSTM_NN_Publication_EVAL.ipynb
CKPT_DIR=eval_v5_snapfix
EVAL_CKPT=${PUB_EVAL_CKPT:-./checkpoints_P1_matched_v5/best_mooring_gat_lstm_seed_42.pt}
VENV_PY=/scratch/tkonstantaras/venvs/pyg_env/bin/python
SRC_CACHE=/scratch/tkonstantaras/mooring_data/cache_npy

# the corrected catalogue, asserted below so a stale notebook cannot be run
N_SNAP_EVENTS=1388
N_SNAP_CASES=200
# the exclusion MASK, which this job must NOT change
N_MASK_CASES=109
N_MASK_SAMPLES=1643

die() { echo "[FATAL] $1" >&2; echo "[exit] job aborted"; exit 1; }

echo "[job] tag=${TAG} jobid=${SLURM_JOB_ID} node=$(hostname) started $(date)"
cd "${SLURM_SUBMIT_DIR}"

module load 2024r1
module load openmpi
module load py-torch
source /scratch/tkonstantaras/venvs/pyg_env/bin/activate
echo "[env] python: $(which python)"
nvidia-smi || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR=/scratch/tkonstantaras/tmp
export PUB_EVAL_CKPT="${EVAL_CKPT}"
mkdir -p "${TMPDIR}" logs

# --- HARD GATE 0: the checkpoint must exist, be v5's, and carry the right
# architecture. An eval whose checkpoint is the wrong model is worse than no
# eval -- it produces plausible numbers for the wrong thing.
[ -f "${EVAL_CKPT}" ] || die "checkpoint not found: ${EVAL_CKPT} (upload checkpoints_P1_matched_v5/, or set PUB_EVAL_CKPT=/path/to/best_....pt)"
"${VENV_PY}" - "${EVAL_CKPT}" <<'PY' || die "checkpoint preflight failed"
import sys
import torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
for k in ("model_state_dict", "criterion_state_dict", "cfg", "epoch"):
    assert k in ck, f"checkpoint has no {k!r}"
c = ck["cfg"]
want = dict(gat_hidden_dim=128, gat_out_dim=128, num_heads=4,
            lstm_hidden_dim=384, num_lstm_layers=2, lstm_bidirectional=True,
            head_hidden_dim=128, use_global_context=True, window_len=1000)
bad = {k: (c.get(k, "<absent>"), v) for k, v in want.items() if c.get(k) != v}
assert not bad, f"checkpoint architecture mismatch: {bad}"
# v5 identity: epoch 32 was the single feasible epoch and IS the shipped file
ep = ck["epoch"]
assert ep == 32, (f"checkpoint is epoch {ep}, expected v5's epoch 32. "
                  "If you deliberately want another epoch, edit this gate.")
# v5 carries the patch-26 LR fix explicitly; v4 and earlier do not
assert c.get("scheduler_threshold_mode") == "abs", (
    "checkpoint has scheduler_threshold_mode=%r -- that is NOT the v5 run"
    % c.get("scheduler_threshold_mode", "<absent>"))
print(f"[ckpt] OK: v5 epoch {ep}, PC1 dims, global context ON, W=1000, LR fix present")
PY

# --- HARD GATE 1: GPU + imports before staging 13 GB.
"${VENV_PY}" - <<'PY' || die "env/GPU preflight failed -- no usable GPU or a broken venv"
import sys
print("[diag] py:", sys.executable)
import torch
print("[diag] torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print("[diag] gpu:", torch.cuda.get_device_name(0))
import numpy, torch_geometric, nbconvert
print("[diag] numpy:", numpy.__version__, "pyg:", torch_geometric.__version__)
from torch_geometric.nn import GATv2Conv
print("[diag] imports OK")
PY

# --- HARD GATE 2: the notebook must be the patch-29 EVAL notebook: it must not
# train, it must point at v5, it must carry the CORRECTED catalogue, and -- the
# assertion that makes this job interpretable -- it must leave the exclusion
# MASK exactly as v5 had it.
"${VENV_PY}" - "${NOTEBOOK}" "${CKPT_DIR}" "${N_SNAP_EVENTS}" "${N_SNAP_CASES}" \
    "${N_MASK_CASES}" "${N_MASK_SAMPLES}" <<'PY' || die "notebook is NOT the patch-29 EVAL notebook -- re-run: python _nb_patch_pub_29_eval_snapfix.py"
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
ckpt = sys.argv[2]
N_EV, N_CS, N_MC, N_MS = (int(v) for v in sys.argv[3:7])
cells = {c.get("id"): "".join(c["source"]) for c in nb["cells"]}
run = cells["a04f1f5f"]

# --- it must be an EVAL cell, not a training cell ---
assert "EVAL-ONLY" in run, "run cell is not the eval cell"
assert "train_model(" not in run, "run cell still calls train_model -- this is not an eval"
# the CALL, not the word: the cell documents in a comment why it is omitted
assert "calibrate_log_vars(" not in run, "calibrate_log_vars() would overwrite the checkpoint's Kendall log_vars"
for lit in ['checkpoint_dir="./%s"' % ckpt,
            'resume_from_checkpoint=None',
            'save_feasible_checkpoints=False',
            'run_test_evaluation=True',
            'snap_oversample_frac=0.0',
            'PUB_EVAL_CKPT',
            'strict=True',
            'model.eval()']:
    assert lit in run, "eval run cell missing: %r" % lit
# it must target v5 and nothing else
assert "checkpoints_P1_matched_v5" in run, "run cell does not default to the v5 checkpoint"
for stale in ("v3_gc", "v3gc", "checkpoints_P1_matched_v4",
              "eval_v3gc_on_fixed_pipeline"):
    assert stale not in run, "run cell still references %r" % stale
# architecture + budget literals must still match v5
for lit in ['gat_hidden_dim=128', 'gat_out_dim=128', 'lstm_hidden_dim=384',
            'num_lstm_layers=2', 'lstm_bidirectional=True',
            'use_global_context=True',
            'node_counts=(4, 5, 6, 7, 8, 10, 12, 15, 18, 21)',
            'max_train_windows=1_000_000',
            'max_val_windows=10_000',
            'max_test_windows=50_000',
            'n_fit_windows=5_000',
            'raw_block_len=2000']:
    assert lit in run, "eval cfg missing: %r" % lit
# the reproduction check must be present -- it is how we learn the population
# did not move
assert "_V5_REF" in run, "the v5 reproduction check is missing from the run cell"

# --- the OOD split path is filtered (patch 22 fix 1, the job-10521852 bug) ---
sp = cells["858f4194"]
assert "all_wins = list(range(len(ds)))" not in sp, \
    "the OOD BYPASS IS STILL PRESENT -- this is the bug that wasted job 10521852"
assert sp.count("all_wins = valid_window_starts_inside_raw_interval(") == 1, \
    "held-out branch does not call the artifact filter exactly once"
assert "ARTIFACT_SAMPLES.get(key)" in sp, "window filter not wired -- re-apply patch 19"

# --- the pipeline is UNCHANGED from v5 (patches 19/21/22) -------------------
bl = cells["f80f8d84"]
assert "ARTIFACT_TENSION_CEILING_N = 50_000.0" in bl, "50 kN ceiling missing"
assert "EXCLUDE_ARTIFACT_WINDOWS = True" in bl, "per-window exclusion is OFF"
assert "assert len(CORRUPT_CASES_ALL) == 109" in bl, "109-case union missing"
assert "assert len(CORRUPT_CASES_NONFINITE) == 11" in bl, "non-finite set missing"
assert "assert len(CORRUPT_CASES_TOO_SHORT) == 2" in bl, "too-short set missing -- re-apply patch 21"
assert "(CORRUPT_CASES_NONFINITE | CORRUPT_CASES_TOO_SHORT)" in bl, "too-short cases not unioned"

ns = {}
lo = bl.index("ARTIFACT_SAMPLES = {")
hi = bl.index("\n}\n", lo) + 3
exec(bl[lo:hi], ns)
A = ns["ARTIFACT_SAMPLES"]
assert len(A) == N_MC, f"ARTIFACT_SAMPLES covers {len(A)} cases, expected {N_MC}"
n_s = sum(len(v) for v in A.values())
assert n_s == N_MS, (
    f"ARTIFACT_SAMPLES has {n_s} samples, expected {N_MS}. THE MASK MOVED -- "
    "this eval is only interpretable if the test population is identical to "
    "v5's. Changing the mask requires re-measuring reachability and re-running "
    "FINAL (CLAUDE.md 12.6 step 5), not an eval.")

# --- patch 29: the CORRECTED snap catalogue --------------------------------
ns2 = {}
lo = bl.index("SNAP_EVENT_TIMES = {")
hi = bl.index("\n}\n", lo) + 3
exec(bl[lo:hi], ns2)
S = ns2["SNAP_EVENT_TIMES"]
n_ev = sum(len(v) for v in S.values())
assert n_ev == N_EV and len(S) == N_CS, (
    f"SNAP_EVENT_TIMES has {n_ev} events in {len(S)} cases, expected {N_EV} in "
    f"{N_CS}. This is the LEGACY catalogue (1611/201) or a stale one -- "
    "re-run: python scan_snap_events.py && python _nb_patch_pub_29_eval_snapfix.py")
assert "'snap catalogue is not the patch-29 corrected one'" in bl, \
    "the runtime catalogue assert is missing"
assert "guard A" in bl and "refractory" in bl, "catalogue provenance comment missing"
# the two worked examples, positively
assert 9783 in S.get((4, 135), ()), \
    "loc04/case0135 t=9783 (the 16.8 kN main event) is NOT catalogued -- the refractory argmax fix did not apply"
assert 9773 not in S.get((4, 135), ()), \
    "loc04/case0135 still points at its t=9773 precursor"
assert 7603 not in S.get((10, 224), ()), \
    "loc10/case0224 t=7603 (the catalogued artifact) is still present -- guard A did not apply"
assert 7583 in S.get((11, 296), ()), \
    "loc11/case0296 t=7583 (the canonical clean 20.4 kN snap) was lost -- the guards are too aggressive"

# --- patch 23 + 27 + 29: the dump ------------------------------------------
ts = cells["pubts01"]
assert "snap_ev" in ts and "snap_true" in ts, "snap-event capture missing -- re-apply patch 23"
assert "int(starts[k])))" in ts, "meta_all does not record the window start -- re-apply patch 23"
assert "pin_sampler_epoch" in ts, "sampler pin missing"
assert '_store("sn"' in ts, "patch 27 representative-snap selector missing"
assert "evrows" in ts, "patch 27 event-row annotation missing"
assert "patch 29: DE-DUPLICATE BY PHYSICAL EVENT" in ts, "patch 29 dedup missing"
assert "_push(hi_sel_d" in ts and "_push(un_sel_d" in ts, "hi/un not de-duplicated"
assert "_ranked(hi_sel_d, n_extreme)" in ts, "hi ranking not event-aware"
assert "nwin" in ts, "collapsed-window count not stored"
assert "heapq" not in ts, "the old window-ranking heap is still present"
assert "dump_test_timeseries(" in run, "run cell does not call the dump"

print(f"[sel] EVAL notebook verified: no training, v5 checkpoint -> {ckpt}")
print(f"[sel]   mask UNCHANGED: {len(A)} cases / {n_s} samples (= v5)")
print(f"[sel]   catalogue CORRECTED: {n_ev} events / {len(S)} cases (was 1611/201)")
print(f"[sel]   dump: patch 23 snap capture + 27 sn/evrows + 29 event dedup")
PY

# --- HARD GATE 3: record-length preflight (patch 21) ------------------
"${VENV_PY}" - "${NOTEBOOK}" "${SRC_CACHE}" <<'PY' || die "record-length preflight failed -- a non-excluded case is shorter than window_len"
import ast, json, pathlib, re, sys
import numpy.lib.format as nf

nb = json.load(open(sys.argv[1], encoding="utf-8"))
cache = pathlib.Path(sys.argv[2])
cells = {c.get("id"): c for c in nb["cells"]}

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
for loc, case, n in bad:
    print(f"[len] TOO SHORT: loc{loc:02d} case{case:04d} = {n} steps < {W}")
assert not bad, f"{len(bad)} non-excluded case(s) shorter than window_len={W}"
print(f"[len] OK: all {seen} in-use cases have >= {W} steps")
PY

# --- stage the ~13 GB npy cache to node-local /tmp -------------------
LOCAL_ROOT=/tmp/mooring_${SLURM_JOB_ID}
LOCAL_CACHE=${LOCAL_ROOT}/cache_npy
trap 'echo "[stage] cleanup ${LOCAL_ROOT}"; rm -rf "${LOCAL_ROOT}"' EXIT

NEED_KB=$(( $(du -sk "${SRC_CACHE}" | cut -f1) + 2*1024*1024 ))
FREE_KB=$(df -k --output=avail /tmp | tail -1 | tr -d ' ')
echo "[stage] /tmp free ${FREE_KB} KB, need ${NEED_KB} KB"
[ "${FREE_KB}" -ge "${NEED_KB}" ] || die "/tmp too small (free ${FREE_KB} < need ${NEED_KB} KB)"

echo "[stage] copying ${SRC_CACHE} -> ${LOCAL_CACHE} ($(date))"
mkdir -p "${LOCAL_CACHE}"
rsync -a "${SRC_CACHE}/" "${LOCAL_CACHE}/" || die "rsync failed"
SRC_N=$(find "${SRC_CACHE}"   -type f | wc -l)
DST_N=$(find "${LOCAL_CACHE}" -type f | wc -l)
[ "${DST_N}" -gt 0 ]          || die "canary: 0 files staged"
[ "${SRC_N}" -eq "${DST_N}" ] || die "canary: file count ${SRC_N} != ${DST_N}"

export MOORING_CACHE_DIR="${LOCAL_CACHE}"
echo "[stage] OK: ${DST_N} files staged ($(date))"

# --- run --------------------------------------------------------------
mkdir -p "${CKPT_DIR}"

echo "[run] executing ${NOTEBOOK} ($(date))"
echo "[run] stdout lands in the executed .ipynb, NOT this .out;"
echo "[run] a traceback lands in logs/PUB_EVAL_V5_SNAPFIX_${SLURM_JOB_ID}.err"
srun "${VENV_PY}" -m nbconvert --to notebook --execute \
    "${NOTEBOOK}" \
    --output "GAT_LSTM_NN_Publication_${TAG}_executed_${SLURM_JOB_ID}.ipynb" \
    --ExecutePreprocessor.timeout=-1
RC=$?

echo "[job] nbconvert exit code ${RC}; finished $(date)"
if [ ${RC} -eq 0 ]; then
    echo "[job] results in ${CKPT_DIR}/ :"
    ls -la "${CKPT_DIR}" || true
    echo "[job] bring back ${CKPT_DIR}/test_timeseries_dump.npz -- snap_ev/"
    echo "[job] snap_true/snap_pred are the deliverable; join to snap_catalogue.csv"
    echo "[job] on (loc, case, t) locally."
fi
exit ${RC}
