#!/bin/bash
#SBATCH --job-name=LR_BASELINE
#SBATCH --partition=compute-p1
#SBATCH --account=education-me-msc-ode
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=logs/LR_BASELINE_%j.out
#SBATCH --error=logs/LR_BASELINE_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =====================================================================
# Linear-regression baselines for Stage 1 -- Linear_Regression.ipynb, CPU only, no GPU.
#
# WHY THE CLUSTER. Every worker process peaks at ~1-1.5 GB, and on a 16 GB laptop ten
# workers paged the machine until it froze (2026-09-30). One compute-p1 node (48 cores,
# 187 GB) runs 47 workers.
#
# RUNTIME. Job 872463 (LR0-LR3) took 2:15, measured. This version adds LR4 (lags every 1 s out
# to +-40 s, 1,618 inputs): its fit (pass C, ~2e8 rows) is ~30-60 min of the 47 workers and its
# evaluation ~15 min. Submitted from the SAME directory as job 872463, its cached passes (A, B,
# validation: ~1.7 h) are reused and the whole job takes ~2-3 h; from a fresh directory ~4 h.
# The 8 h limit leaves room for a slow node.
#
# WHAT IT DOES
#   GATE 0    every input file is present and the code is the cluster-ready version
#   GATE 0a   the software environment imports (scikit-learn is optional)
#   GATE 0b   the npy cache on /scratch is complete
#   STEP 1    SMOKE run: six simulations through every cell and gate (~5 min). A preflight:
#             a failure here costs minutes, not the full run.
#   STEP 2    FULL run
#   STEP 3    the comparison table into this log, and one results tarball to download
#
# THE DATA are read straight from /scratch, not staged to /tmp. Each worker reads a case's
# five small .npy files once per pass (~50 GB over the job, spread over 47 readers), while
# staging 13.65 GB with rsync measured 40-60 min on the GPU jobs. Nothing is written there.
#
# UPLOAD: the CONTENTS of LR_cluster_upload/ (python _make_lr_cluster_bundle.py builds it)
# into the directory job 872463 ran in, overwriting the files there: it holds
# Results/LR_baseline/_cache, whose finished passes are then reused. Then
#     cd <that directory> && sbatch run_LR_baseline.slurm
# A fresh directory also works; it only recomputes the cached passes.
# NOTE: logs/ must exist BEFORE sbatch (Slurm does not create it; without it the job dies
#   without a trace). The bundle contains an empty logs/.
#
# FOLLOW IT: the notebook's printed progress is mirrored into this .out (and into
#   logs/LR_progress_<jobid>_<mode>.log). A notebook traceback lands in the .err.
# RESULT: logs/LR_baseline_results_<jobid>.tar.gz = Results/LR_baseline/ including both
#   executed notebooks, without the pass caches. Extract it in the project root.
# RESUME: finished training tasks are cached in Results/LR_baseline/_cache, so a
#   resubmission from the same directory reruns the smoke check and continues the full run.
# FEWER CORES: lower --cpus-per-task; the notebook sizes its worker pool from Slurm.
# =====================================================================

set -u

NOTEBOOK=Linear_Regression.ipynb
OUT=Results/LR_baseline
VENV_PY=/scratch/tkonstantaras/venvs/pyg_env_2025/bin/python
SRC_CACHE=/scratch/tkonstantaras/mooring_data/cache_npy
REQUIRED=(
  Linear_Regression.ipynb
  linear_regression_lib.py
  stage1_verbatim.py
  artifact_samples.csv
  artifact_samples_cat.csv
  snap_catalogue_cat.csv
  Results/checkpoints_P1_matched_v5/config.json
  Results/checkpoints_P1_matched_v5/test_timeseries_dump.npz
  Results/eval_v5_snapfix/test_timeseries_dump.npz
  Results/SR_RESULTS/checkpoints_P2_eval_clean/stage1/test_metrics.csv
  Results/SR_RESULTS/checkpoints_P2_eval_clean/stage1/test_metrics_by_node_count.csv
  Results/SR_RESULTS/checkpoints_P2_eval_clean/stage1/test_per_window_stats.csv
  Results/SR_RESULTS/checkpoints_P2_eval_clean/stage1/test_timeseries_dump.npz
  Results/SR_RESULTS/checkpoints_P2_eval_clean/stage1/published/test_per_window_stats.csv
  Results/LR_ref_872463/LR0_static/test_metrics.csv
  Results/LR_ref_872463/LR0_static/published/test_metrics.csv
  Results/LR_ref_872463/LR1_pointwise/test_metrics.csv
  Results/LR_ref_872463/LR1_pointwise/published/test_metrics.csv
  Results/LR_ref_872463/LR2_line/test_metrics.csv
  Results/LR_ref_872463/LR2_line/published/test_metrics.csv
  Results/LR_ref_872463/LR3_lagged/test_metrics.csv
  Results/LR_ref_872463/LR3_lagged/published/test_metrics.csv
)

die() { echo "[FATAL] $1" >&2; echo "[exit] job aborted: $1"; exit 1; }

echo "[job] LR_BASELINE jobid=${SLURM_JOB_ID} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK} started $(date)"
cd "${SLURM_SUBMIT_DIR}" || die "cannot cd to ${SLURM_SUBMIT_DIR}"
mkdir -p logs

# --- GATE 0: the inputs, and the RIGHT version of the code -----------------------------
_missing=0
for f in "${REQUIRED[@]}"; do
  [ -f "$f" ] || { echo "[files] MISSING: $f" >&2; _missing=1; }
done
[ "${_missing}" -eq 0 ] || die "input files missing (listed in the .err) -- upload the whole LR_cluster_upload/ folder"
grep -q "def check_ols_vs_reference" linear_regression_lib.py \
  && grep -q "MIN_AVAIL_GB" linear_regression_lib.py \
  && grep -q "LAGS_LONG" linear_regression_lib.py && grep -q "def restandardise" linear_regression_lib.py \
  || die "linear_regression_lib.py is an older version (no LR4) -- re-upload the bundle"
grep -q "the reference's own scaling" linear_regression_lib.py \
  || die "linear_regression_lib.py predates the GATE 3 fix of job 882173 (raw-input reference) -- re-upload the bundle"
grep -q "LR_PROGRESS_LOG" "${NOTEBOOK}" && grep -q "check_ols_vs_reference" "${NOTEBOOK}" \
  && grep -q "LR4_lagged40" "${NOTEBOOK}" && grep -q "GATE 8" "${NOTEBOOK}" \
  || die "${NOTEBOOK} is an older version (no LR4) -- re-upload the bundle"
[ -d "${OUT}/_cache" ] && echo "[files] ${OUT}/_cache exists: finished tasks with a matching key are reused"
echo "[files] OK: all ${#REQUIRED[@]} inputs present; code is the cluster-ready version"

# The pool never holds more than a few files per worker; raise the soft limit anyway.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true
echo "[fd] nofile soft $(ulimit -Sn)"

# --- environment: the 2025 stack and the existing venv (same as the Stage-2 jobs) ------
module load 2025
module load py-torch/2.5.1
module load py-pandas/2.2.3
module load py-matplotlib/3.9.2
source /scratch/tkonstantaras/venvs/pyg_env_2025/bin/activate
# One thread per process: the parallelism is the pool of worker processes.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/scratch/tkonstantaras/tmp
mkdir -p "${TMPDIR}"
# The Stage-1 reference files must be present: a skipped gate proves nothing.
export LR_REQUIRE_REFERENCES=1

# --- GATE 0a: the software environment -----------------------------------------------
"${VENV_PY}" - <<'PY' || die "software environment broken (see above) -- check that 'module load 2025 py-torch/2.5.1' resolves and that the venv exists"
import importlib, sys
print(f"[env] python {sys.version.split()[0]} ({sys.executable})")
for m in ("numpy", "torch", "torch_geometric", "pandas", "matplotlib", "nbconvert", "ipykernel", "psutil"):
    print(f"[env] {m} {getattr(importlib.import_module(m), '__version__', '?')}")
for m, why in (("sklearn", "GATE 3 then uses numpy's lstsq, the same algorithm"),
               ("threadpoolctl", "threads are pinned by OMP_NUM_THREADS")):
    try:
        print(f"[env] {m} {importlib.import_module(m).__version__} (optional)")
    except ImportError:
        print(f"[env] {m} not installed (optional: {why})")
import torch
print(f"[env] torch threads per process: {torch.get_num_threads()}")
PY

# --- GATE 0b: the cache ----------------------------------------------------------------
for f in loc01/case_0001/x_abs.npy loc05/case_0100/reference.npy loc10/case_0150/env.npy loc11/case_0300/tension.npy; do
  [ -f "${SRC_CACHE}/${f}" ] || die "cache file missing: ${SRC_CACHE}/${f}"
done
_n_cases=$(ls -d "${SRC_CACHE}"/loc*/case_* 2>/dev/null | wc -l)
echo "[cache] ${SRC_CACHE}: ${_n_cases} simulation directories"
[ "${_n_cases}" -ge 3000 ] || die "expected 3000 simulation directories in ${SRC_CACHE}, found ${_n_cases}"
export MOORING_CACHE_DIR="${SRC_CACHE}"

# --- run the notebook headless; its progress is mirrored into this .out ---------------
run_nb() {   # $1 = smoke | full, $2 = output directory, $3 = executed-notebook name
  local mode=$1 outdir=$2 name=$3
  local log="logs/LR_progress_${SLURM_JOB_ID}_${mode}.log"
  : > "${log}"
  echo "[run] ${mode}: started $(date)"
  LR_RUN_MODE="${mode}" LR_PROGRESS_LOG="${log}" \
    "${VENV_PY}" -m nbconvert --to notebook --execute "${NOTEBOOK}" \
      --output-dir "${outdir}" --output "${name}" \
      --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 &
  local pid=$!
  tail -n +1 -F --pid="${pid}" "${log}" 2>/dev/null &
  local tpid=$!
  wait "${pid}"
  local rc=$?
  wait "${tpid}" 2>/dev/null
  echo "[run] ${mode}: nbconvert exit code ${rc} at $(date)"
  return "${rc}"
}

_t0=$(date +%s)
run_nb smoke "${OUT}/_smoke" Linear_Regression_smoke_executed \
  || die "SMOKE run failed -- the traceback is in logs/LR_BASELINE_${SLURM_JOB_ID}.err; the full run was not started"
_t1=$(date +%s)
echo "[time] smoke run $(( (_t1 - _t0) / 60 )) min"

run_nb full "${OUT}" Linear_Regression_executed \
  || die "FULL run failed -- the traceback is in logs/LR_BASELINE_${SLURM_JOB_ID}.err; finished tasks are cached in ${OUT}/_cache, so resubmitting from this directory resumes"
_t2=$(date +%s)
echo "[time] full run $(( (_t2 - _t1) / 60 )) min"

# --- STEP 3: the comparison table, and one tarball to download ------------------------
"${VENV_PY}" - "${OUT}/lr_vs_stage1.csv" <<'PY' || echo "[result] could not print ${OUT}/lr_vs_stage1.csv"
import csv, sys
rows = list(csv.reader(open(sys.argv[1], newline="", encoding="utf-8")))

def fmt(c):
    try:
        return f"{float(c):.4g}"
    except ValueError:
        return c

w0 = max(len(r[0]) for r in rows)
for r in rows:
    print("[result] " + r[0].ljust(w0) + "".join(fmt(c).rjust(22) for c in r[1:]))
PY

mkdir -p "${OUT}/logs"
cp logs/LR_progress_"${SLURM_JOB_ID}"_*.log "${OUT}/logs/"
_tar="logs/LR_baseline_results_${SLURM_JOB_ID}.tar.gz"
tar -czf "${_tar}" --exclude="_cache" "${OUT}" || die "tar failed"
echo "[done] ${_tar} ($(du -h "${_tar}" | cut -f1)) -- download it and extract it in the project root"
echo "[job] finished $(date), total $(( ($(date +%s) - _t0) / 60 )) min"
