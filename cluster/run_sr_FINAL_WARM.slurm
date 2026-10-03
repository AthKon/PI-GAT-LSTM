#!/bin/bash
#SBATCH --job-name=SR_FINAL_WARM
#SBATCH --partition=gpu-a100
#SBATCH --account=education-me-msc-ode
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/SR_FINAL_WARM_%j.out
#SBATCH --error=logs/SR_FINAL_WARM_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# =====================================================================
# PUBLICATION STAGE 2 — SPATIAL SUPER-RESOLUTION (N_in -> N_out)
# mode: FINAL_WARM
#
# Upload before submitting, from the project root:
#   GAT_LSTM_NN_Publication_superresolution.ipynb              (with _MODE = ("FINAL_WARM", ...) ACTIVE)
#   this script
#   checkpoints_P1_matched_v5/best_mooring_gat_lstm_seed_42.pt   (GATE 0 refuses to start without it)
#
# Stage 1 is closed. v5 (job 10532625, ep32) reconstructs the tension field at
# MATCHED resolution because its per-node head is sized to the input node count.
# Stage 2 inserts a query-based cross-attention decoder between the GAT encoder
# and the LSTM, so the dense 21-node field is reconstructed from as few as 4
# motion sensors. N_out = 21 is the ceiling -- the native FE resolution is the
# finest ground truth that exists.
#
# The eight modes, and why they run in this order:
#   BASELINE       v5 + interpolation of its prediction, over the whole
#                  N_in x N_out grid. No training. Sets the bar the decoder
#                  must clear, for ~1 h instead of ~15.
#   HPO_WARM       bake-off arm A: warm-start the backbone from v5.
#   HPO_SCRATCH    bake-off arm B: train from scratch.
#   FINAL_*        full budgets with the winning arm; also re-runs the
#                  baseline in-job so the two surfaces are perfectly paired.
#   EVAL_NOISE     eval-only on the SHIPPED Stage-2 checkpoint: a motion-sensor
#                  noise sweep (absolute sigma, per-channel SNR, a
#                  derivative-coupled bound and a per-station bias) plus the
#                  per-window statistics the wave-train repeat split needs.
#                  No training, no grid -- test_model already reproduces the
#                  grid's N_out=21 column to 1.3e-7.
#   EVAL_SNAPROWS  eval-only on BOTH shipped checkpoints (patch sr-20): Stage 2
#                  on its paired draw, then v5 on the original unpaired Stage-1
#                  draw, each storing every station around every snap event so
#                  capture can be read at the snap's own station. Each pass must
#                  reproduce its published metrics and snap-event rows.
#   EVAL_CLEAN     eval-only on BOTH shipped checkpoints (patch sr-22): the
#                  published test draws thinned by the EVALUATION mask (the
#                  catenary rule). Each draw is rebuilt, fingerprinted and
#                  reproduced before it is thinned. Needs BOTH checkpoints.
#
# stderr goes to logs/SR_FINAL_WARM_<jobid>.err -- a notebook traceback lands
# THERE, not in the .out, and nbconvert does not write the executed .ipynb on
# failure. Healthy-start signature in the .out, within ~2 min:
#   [ckpt] OK: v5 epoch 32 ...            (skipped for HPO_SCRATCH)
#   [diag] all imports OK
#   [sel] STAGE-2 mode FINAL_WARM verified ...
#   [len] OK: all 2987 in-use cases have >= 1000 steps
#
# WALLTIME NOTE for the FINAL modes: with N_out pinned at 21 the LSTM and head
# always run at 21 stations instead of at N_in, so expect ~1.5-1.8x v5's
# 7.5 h. If training consumes the full 20 h the grid never runs -- just
# resubmit: the run cell auto-resumes from the newest checkpoint in
# _RESUME_SCAN_DIR, returns immediately once num_epochs is reached, and then
# proceeds to the grid + baseline + per-checkpoint dumps.
# =====================================================================

set -u

TAG=FINAL_WARM
NOTEBOOK=GAT_LSTM_NN_Publication_superresolution.ipynb
CKPT_DIR=checkpoints_P2_final_warm
V5_CKPT=checkpoints_P1_matched_v5/best_mooring_gat_lstm_seed_42.pt
NEEDS_V5=yes
VENV_PY=/scratch/tkonstantaras/venvs/pyg_env_2025/bin/python
FD_MIN=65536
SRC_CACHE=/scratch/tkonstantaras/mooring_data/cache_npy

die() { echo "[FATAL] $1" >&2; echo "[exit] job aborted -- resubmit (likely a different node)"; exit 1; }

echo "[job] tag=${TAG} jobid=${SLURM_JOB_ID} node=$(hostname) started $(date)"
cd "${SLURM_SUBMIT_DIR}"

# --- HARD GATE 0-fd: the OPEN-FILE limit ------------------------------------
# The lazy dataset opens 4 mmap handles per (loc, case, N) and NEVER closes
# them, so the normalizer fit alone holds ~18k descriptors and one test pass
# needs ~23k per DataLoader worker. That has always been true and always
# worked -- until DelftBlue's 2026-09 rebuild reset the soft `nofile` limit to
# the distro default. Job 100111 then died with
#     OSError: [Errno 24] Too many open files
# inside fit_normalizers_from_train_subset -- 2 min into the notebook, i.e.
# 27 min of staging AFTER the last gate had passed. Raise soft to hard, then
# refuse to start if it is still too low: 5 s of gate beats 30 min of staging.
_FD_BEFORE=$(ulimit -Sn)
# Descending ladder, not just "soft := hard": a hard limit of `unlimited` is
# NOT settable on Linux (the kernel caps nofile at fs.nr_open), and swallowing
# that failure would leave the soft limit at 1024 and abort for the wrong
# reason. First value that sticks wins.
ulimit -n "$(ulimit -Hn)" 2>/dev/null   || ulimit -n 1048576 2>/dev/null   || ulimit -n  262144 2>/dev/null   || ulimit -n  131072 2>/dev/null   || ulimit -n   65536 2>/dev/null   || true
_FD_SOFT=$(ulimit -Sn)
echo "[fd] nofile soft ${_FD_BEFORE} -> ${_FD_SOFT} (hard $(ulimit -Hn))"
if [ "${_FD_SOFT}" != "unlimited" ]; then
  [ "${_FD_SOFT}" -ge "${FD_MIN}" ] || die "open-file limit is only ${_FD_SOFT} (hard $(ulimit -Hn)); this job needs >= ${FD_MIN}. The lazy dataset holds 4 mmap handles per (loc,case,N) and never closes them: the normalizer fit needs ~18k and one test pass ~23k per worker. Either ask DHPC to raise nofile on the compute nodes, or cap the handle cache in MooringSequenceDatasetPositionTension._open_mmap"
fi

# --- HARD GATE 0: the Stage-1 checkpoint this mode depends on ---------
# Cheapest possible failure: no modules, no GPU, no staging. Job 10652594 died
# here at 7 s because only the notebook and the slurm had been uploaded.
if [ "${NEEDS_V5}" = "yes" ]; then
  [ -f "${V5_CKPT}" ] || die "checkpoint not found: ${V5_CKPT} -- upload the whole checkpoints_P1_matched_v5/ directory (or at least this file) into ${SLURM_SUBMIT_DIR}"
fi

# --- environment (order matters; venv relies on py-torch module) -----
# DelftBlue replaced its module stack on 2026-09-05: 2024r1 (py-torch 2.1.0,
# python 3.10.13) is gone, leaving 2025 (py-torch 2.5.1, python 3.11.9) and
# 2026 (py-torch 2.10.0, python 3.13). 2025 is the target: 2.10 removes
# torch.cuda.amp.GradScaler, which the train loop still uses, and 2026 also
# jumps to pandas 3.x. Versions are PINNED so a stack default cannot silently
# move torch under a run. openmpi is not loaded -- these jobs are single-node.
module load 2025
module load py-torch/2.5.1
module load py-pandas/2.2.3
module load py-matplotlib/3.9.2
source /scratch/tkonstantaras/venvs/pyg_env_2025/bin/activate
echo "[env] python: $(which python)"
nvidia-smi || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR=/scratch/tkonstantaras/tmp
mkdir -p "${TMPDIR}" logs

# --- HARD GATE 0a: the software environment itself --------------------
# Job 100052 died because `module load 2024r1` no longer resolves, so the venv
# had no torch -- but the FIRST python gate it reached was 0b, which reported
# it as "the Stage-1 checkpoint is not v5". Check the environment by name so
# the message always matches the cause, and record the versions in the log.
"${VENV_PY}" - <<'PY' || die "software environment is broken -- '${VENV_PY}' cannot import torch/pyg. DelftBlue replaced its module stack on 2026-09-05; verify 'module load 2025 py-torch/2.5.1' still resolves and that the venv exists"
import sys, torch, numpy, torch_geometric, resource
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
print(f"[env] python {sys.version.split()[0]} | torch {torch.__version__} "
      f"| numpy {numpy.__version__} | pyg {torch_geometric.__version__}")
print(f"[env] nofile as python sees it: soft {_soft} hard {_hard}")
assert _soft == -1 or _soft >= 65536, f"the raised ulimit did not reach python (soft {_soft})"
PY

# --- HARD GATE 0b: it must be the RIGHT Stage-1 checkpoint ------------
# Loading v4 (or any pre-LR-fix checkpoint) would silently change both the
# baseline and the warm start. Needs torch, so it runs after the module load
# but still before the 13 GB stage.
if [ "${NEEDS_V5}" = "yes" ]; then
"${VENV_PY}" - "${V5_CKPT}" <<'PY' || die "the Stage-1 checkpoint is not v5 (see [ckpt] lines)"
import sys, torch
p = sys.argv[1]
try:
    ck = torch.load(p, map_location="cpu", weights_only=False)
except TypeError:
    ck = torch.load(p, map_location="cpu")
ep = ck.get("epoch")
assert ep == 32, f"[ckpt] epoch is {ep}, expected 32 (v5's binding pick)"
sd = ck["model_state_dict"]
assert not any(k.startswith("query_decoder.") for k in sd), \
    "[ckpt] this is a STAGE-2 checkpoint -- Stage 2 needs the MATCHED-resolution v5 model"
# PC1 dims: gat 128 (heads 4 -> 512), lstm 384 bidirectional x 2 layers
assert tuple(sd["spatial_encoder.gat1.lin_l.weight"].shape)[0] == 512, "[ckpt] gat_hidden_dim is not 128"
assert tuple(sd["temporal_encoder.lstm.weight_ih_l0"].shape)[0] == 4 * 384, "[ckpt] lstm_hidden_dim is not 384"
assert "temporal_encoder.lstm.weight_ih_l0_reverse" in sd, "[ckpt] LSTM is not bidirectional"
assert "temporal_encoder.lstm.weight_ih_l1" in sd, "[ckpt] LSTM is not 2-layer"
assert "spatial_encoder.global_mlp.0.weight" in sd, "[ckpt] use_global_context was OFF"
# The pickled cfg needs TrainingConfig on the path to rehydrate, which this
# standalone gate does not define. Report threshold_mode only when it IS
# readable; absence here means "not readable", never "the LR fix is missing"
# (the run cell and GATE 2 assert that themselves).
cfg = ck.get("cfg")
tmode = getattr(cfg, "scheduler_threshold_mode", None) if cfg is not None else None
extra = f", threshold_mode={tmode!r}" if tmode else ""
print(f"[ckpt] OK: v5 epoch {ep}, PC1 dims, global context ON, "
      f"{len(sd)} tensors{extra}")
PY
fi

# --- HARD GATE 1: GPU + core imports MUST be OK before staging 13 GB --
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

# --- HARD GATE 1-rnn: cuDNN must accept THIS model's LSTM -------------
# Job 103567 passed every gate and the 27-minute stage, then died 3 min into
# the notebook with
#     RuntimeError: cuDNN error: CUDNN_STATUS_BAD_PARAM
# from torch._cudnn_rnn_flatten_weight, when .to(device) flattened the v5 LSTM.
# Same failure SHAPE as job 100111's EMFILE -- an environment defect that a
# five-second check catches before the expensive part -- so it is now a gate.
# The probe is ADAPTIVE: it keeps cuDNN when cuDNN works, falls back only when
# it must, and refuses to stage when neither path can build the model.
_RNN_VERDICT=".sr_cudnn_verdict.${SLURM_JOB_ID}"
rm -f "${_RNN_VERDICT}"
"${VENV_PY}" - "${_RNN_VERDICT}" <<'PY' || die "cuDNN RNN preflight failed -- neither cuDNN nor the ATen fallback can build the v5 LSTM on this node (see [rnn] lines). This is an environment defect, not a model one: the same nn.LSTM trained for 7.5 h as job 10532625"
import sys
import time

import torch
import torch.nn as nn

verdict_path = sys.argv[1]


def build_and_run():
    """The REAL v5 LSTM: gat_out_dim 128 -> 384 hidden, 2 layers, bidirectional."""
    m = nn.LSTM(input_size=128, hidden_size=384, num_layers=2,
                batch_first=True, dropout=0.1, bidirectional=True).cuda()
    x = torch.randn(8, 64, 128, device="cuda")
    y, _ = m(x)
    y.sum().backward()
    torch.cuda.synchronize()
    return m, tuple(y.shape)


def timed(m, batch=64, steps=1000):
    """ms per forward at the REAL window_len, to size HPO/FINAL walltimes."""
    x = torch.randn(batch, steps, 128, device="cuda")
    with torch.no_grad():
        m(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            m(x)
        torch.cuda.synchronize()
    return (time.time() - t0) / 3.0 * 1000.0


try:
    _ver = torch.backends.cudnn.version()
except Exception:
    _ver = None
print("[rnn] cudnn available:", torch.backends.cudnn.is_available(),
      "version:", _ver, "enabled:", torch.backends.cudnn.enabled)

try:
    model, shape = build_and_run()
    print("[rnn] OK: cuDNN accepts the v5 LSTM; fwd+bwd", shape)
    need_fallback = 0
except RuntimeError as exc:
    print("[rnn] cuDNN REJECTED the v5 LSTM:", exc)
    print("[rnn] retrying with torch.backends.cudnn.enabled = False ...")
    torch.backends.cudnn.enabled = False
    try:
        model, shape = build_and_run()
    except RuntimeError as exc2:
        print("[rnn] FATAL: the ATen fallback fails too:", exc2)
        raise SystemExit(1)
    print("[rnn] OK: ATen fallback builds and runs the v5 LSTM; fwd+bwd", shape)
    need_fallback = 1

try:
    print("[rnn] forward [64,1000,128]: %.1f ms/iter (cudnn=%s)"
          " -- use this to size the HPO/FINAL walltimes"
          % (timed(model), torch.backends.cudnn.enabled))
except Exception as exc:
    print("[rnn] timing skipped:", exc)

with open(verdict_path, "w") as fh:
    fh.write(str(need_fallback))
PY

if [ "$(cat "${_RNN_VERDICT}" 2>/dev/null || echo x)" = "1" ]; then
  export SR_DISABLE_CUDNN=1
  echo "[rnn] exporting SR_DISABLE_CUDNN=1 -- cell a04f1f5f (patch sr-08) will disable cuDNN; the LSTM runs on ATen kernels, numerics identical, RNN slower"
else
  echo "[rnn] cuDNN is healthy on this node -- leaving it enabled"
fi
rm -f "${_RNN_VERDICT}"

# --- HARD GATE 2: the notebook MUST be on THIS job's Stage-2 mode -----
# The five modes share ONE notebook and are selected by a commented toggle, so
# submitting the wrong file is a single-line mistake with a 20 h cost. Refuse
# unless exactly one _MODE line is active and it is this job's.
"${VENV_PY}" - "${NOTEBOOK}" "${TAG}" "${CKPT_DIR}" <<'PY' || die "notebook is NOT on Stage-2 mode ${TAG} -- uncomment the right _MODE line (see cell a04f1f5f) and re-upload"
import json, sys
nb = json.load(open(sys.argv[1], encoding="utf-8"))
tag, ckpt = sys.argv[2], sys.argv[3]
cells = {c.get("id"): c for c in nb["cells"]}
cfg_src = "".join(cells["a04f1f5f"]["source"])

# ---- the mode toggle: exactly ONE active line, and it is ours -------------
active = [l.rstrip() for l in cfg_src.splitlines()
          if l.lstrip().startswith("_MODE = (")]
assert len(active) == 1, (
    "expected EXACTLY ONE active _MODE line in cell a04f1f5f, found %d: %r"
    % (len(active), active))
assert '"%s"' % tag in active[0], (
    "active mode line is %r but this job is %s -- uncomment the right line"
    % (active[0], tag))
# the run cell derives the dir as "./checkpoints_P2_" + _TAG.lower()
assert ckpt == "checkpoints_P2_" + tag.lower(), (
    "slurm CKPT_DIR=%s does not match the mode-derived dir for %s" % (ckpt, tag))
assert '_CKPT_DIR = "./checkpoints_P2_" + _TAG.lower()' in cfg_src, \
    "run cell no longer derives the checkpoint dir from the mode -- re-apply _nb_patch_sr_06_runcell.py"
assert "_RESUME_SCAN_DIR = _CKPT_DIR" in cfg_src, \
    "auto-resume dir is not tied to the checkpoint dir -- a fresh run could CONTINUE another mode's checkpoints"
assert "checkpoints_P1_matched_v4" not in cfg_src, "run cell references the v4 dir"
# Stage 2 must never write into Stage 1's directory
assert 'checkpoint_dir="./checkpoints_P1' not in cfg_src, \
    "run cell would write into a Stage-1 checkpoint dir"

# ---- Stage-2 configuration ------------------------------------------------
for lit in ['superres_mode=True', 'superres_out_N=21',
            'use_query_decoder=(not _IS_BASELINE)',
            'query_decoder_residual_interp=True',
            'query_decoder_zero_init=True',
            'pair_windows_across_node_counts=True',
            'superres_grid_out_N=(4, 5, 6, 7, 8, 10, 12, 15, 18, 21)',
            'assert_windows_are_paired(split_indices, cfg.node_counts)']:
    assert lit in cfg_src, "run cell is missing %r -- re-apply _nb_patch_sr_06_runcell.py" % lit

# ---- architecture: the locked Stage-1 winner, unchanged -------------------
for lit in ['gat_hidden_dim=128', 'gat_out_dim=128', 'lstm_hidden_dim=384',
            'num_lstm_layers=2', 'lstm_bidirectional=True',
            'use_global_context=True', 'batch_size=_BATCH', 'seed_list=(42,)',
            'snap_oversample_frac=0.0']:
    assert lit in cfg_src, "run cell is missing %r" % lit

# ---- selection v4 + the LR fix, unchanged from Stage 1 -------------------
for lit in ['sel_abort_epoch=(0 if _IS_HPO else 20)',
            'sel_r2_delta=0.0010',
            'scheduler_threshold_mode="abs"', 'scheduler_min_lr=5e-5',
            'save_feasible_checkpoints=True']:
    assert lit in cfg_src, "run cell is missing %r" % lit
assert "sel_up99_cap" not in cfg_src, "the p99 rate cap is back -- it gave 0 feasible epochs twice"
assert "sel_bias90_cap=" not in cfg_src, "the patch-09f bias box is back"

trn = "".join(cells["c62bc05c"]["source"])   # patch sr-13: the train loop

# ---- patch sr-13: Gate B = up90 cap AND a positive conservative margin ----
# The HPO bake-off (job 124311) measured annealing walking bias_p90 down
# (+186.7 N at ep1 -> +75.0 at ep33 -> -24.9 at ep60), so up90 climbed through a
# 10 % cap and the WARM arm saved NO checkpoint after ep37 of 60.  FINAL raises
# the rate cap to 20 % and adds bias_p90 > 0, which on the v5 trajectory keeps
# every genuinely conservative epoch and drops the rest.
# ---- patch sr-16: Gate B OFF at FINAL (user, 2026-09-10) ------------------
# Run 1 of FINAL_WARM (job 129246) showed the rate cap and the bias floor are one
# sign statistic (corr -0.84): Gate B rejected every epoch after ep33 although
# those epochs are ~25 % more accurate and carry LOWER tail under-mass. FINAL now
# selects on Gate A + temporal skill; the HPO modes keep their 10 % cap.
assert 'sel_up90_cap=(10.0 if _IS_HPO else float("inf"))' in cfg_src, (
    "the sr-16 Gate-B setting is missing: FINAL must have NO up90 cap and HPO "
    "must keep 10 %. Re-apply _nb_patch_sr_16_gateb_off.py")
assert 'sel_bias90_min=float("-inf")' in cfg_src, (
    "the sr-16 Gate-B setting is missing: there must be NO bias floor. "
    "Re-apply _nb_patch_sr_16_gateb_off.py")
assert "_sr13_bias_ok(_sb90, cfg)" in trn, (
    "the train loop does not apply the sr-13 bias floor in Gate B")
assert "def _sr13_bias_ok" in trn, "the sr-13 Gate-B helper is missing from the train loop"

# ---- patch sr-13: the rolling checkpoint (resume granularity) -------------
# Feasible-only saving left a 23-epoch hole in job 124311. At ~30 min/epoch a
# timeout inside such a hole costs ~11 h and the run may never advance.
assert 'save_latest_checkpoint=True' in cfg_src, (
    "the rolling checkpoint is OFF -- a walltime timeout could resume tens of "
    "epochs earlier. Re-apply _nb_patch_sr_13_final_ready.py")
assert '_lt.replace(checkpoint_dir / "checkpoint_latest.pt")' in trn, (
    "the train loop does not write the rolling checkpoint atomically")
assert '_latest = _p / "checkpoint_latest.pt"' in cfg_src, (
    "the auto-resume picker does not know about checkpoint_latest.pt")
assert 'glob("checkpoint_epoch*.pt")' in cfg_src, (
    "the ts-dump glob changed -- checkpoint_latest.pt must NOT be swept into it")

# ---- patch sr-13: FINAL budget + periodic checkpoints ---------------------
# 60 epochs, not 100: Stage 1 measured FINAL converging in ~half the epochs of
# HPO (HPO picked ep60 of 60, FINAL v5 picked ep32 of 62), and past ~ep40 v5's
# bias_p90 went permanently negative, so later epochs are unselectable anyway.
assert '_EPOCHS, _PATIENCE, _TEST_EVAL = 78, 30, True' in cfg_src, (
    "the FINAL budget is not 78 epochs / patience 30 (patch sr-16: run 2 of "
    "FINAL_WARM trains ep54-78) -- re-apply _nb_patch_sr_16_gateb_off.py")
assert 'save_every_n_epochs=(0 if _IS_HPO else 5)' in cfg_src, (
    "periodic checkpoints are OFF at FINAL -- they are the fallback if the "
    "sr-13 Gate-B change turns out wrong at FINAL scale (v1 shipped its worst "
    "epoch and was unrecoverable because better epochs were never written)")

# ---- patch sr-15: a resumed run must evaluate its BEST epoch, not its LAST --
# best_state is set only by an improvement inside the current submission, so a
# resumed run whose best epoch lies in an earlier submission returned its LAST
# epoch, and the test eval + the super-resolution grid scored the wrong model
# (FINAL_WARM job 129246: best = ep33 in submission 1, epochs 34-53 infeasible).
assert "# -- patch sr-15: restore the best-so-far STATE, not just its score --" in trn, (
    "the sr-15 best-state restore is missing -- a resumed run would evaluate its "
    "LAST epoch instead of the selected one. Re-apply _nb_patch_sr_15_resume_best.py")
assert 'best_state           = _bk["model_state_dict"]' in trn, (
    "the resume block no longer restores best_state from the best checkpoint")
assert 'best_criterion_state = _bk.get("criterion_state_dict")' in trn, (
    "the resume block no longer restores the best criterion state")
_i_se15 = trn.index("start_epoch = _ckpt.get(")
_i_bk15 = trn.index('best_state           = _bk["model_state_dict"]')
_i_op15 = trn.index("_csv_file, _csv_writer, _sr14_hist = _sr14_resume_csv(")
assert _i_se15 < _i_bk15 < _i_op15, (
    "the sr-15 restore is not inside the resume block -- it must follow "
    "start_epoch and precede the metrics-CSV opens")

# ---- patch sr-14: the RESUME path must not destroy or mis-count history ----
# The 2-submission FINAL plan REQUIRES a resume, and that path had never been
# exercised (v4, v5 and both HPO arms each finished in one submission).
# Pre-sr-14: both metrics CSVs were opened "w" (submission 1's epochs wiped) and
# _n_feasible_epochs restarted at 0, so `epoch >= sel_abort_epoch and
# _n_feasible_epochs == 0` was already true on resume and the run aborted in its
# first epoch unless that epoch happened to be feasible.
assert "def _sr14_resume_csv(" in trn, (
    "the sr-14 resume-safe CSV helper is missing -- submission 2 would TRUNCATE "
    "submission 1's metrics_history.csv. Re-apply _nb_patch_sr_14_resume_safe.py")
assert "def _sr14_recover_counters(" in trn, (
    "the sr-14 counter recovery is missing -- submission 2 would abort in its "
    "first epoch. Re-apply _nb_patch_sr_14_resume_safe.py")
assert "_csv_file, _csv_writer, _sr14_hist = _sr14_resume_csv(" in trn, (
    "metrics_history.csv is not opened through the resume-safe helper")
assert "_nc_file, _nc_writer, _ = _sr14_resume_csv(" in trn, (
    "metrics_by_node_count.csv is not opened through the resume-safe helper")
assert '_csv_file    = open(history_path, "w"' not in trn, (
    'the old truncating open("w") of metrics_history.csv is back')
assert '_nc_file   = open(_nc_path, "w"' not in trn, (
    'the old truncating open("w") of metrics_by_node_count.csv is back')
assert "_sel_running_max_r2, _n_feasible_epochs, epochs_without_improvement = " in trn, (
    "the resumed run does not restore the loop counters")
# the opens must come AFTER the resume block, or start_epoch is not yet known
_i_resume = trn.index("start_epoch = _ckpt.get(")
_i_open = trn.index("_csv_file, _csv_writer, _sr14_hist = _sr14_resume_csv(")
assert _i_resume < _i_open, (
    "the metrics CSVs are opened BEFORE the resume block, so start_epoch is "
    "still 1 and submission 1's history would be truncated anyway")

# ---- patch sr-13: torch 2.10-safe GradScaler ------------------------------
assert 'torch.amp.GradScaler("cuda", enabled=' in trn, (
    "GradScaler uses the pre-2.4 spelling")
assert "torch.cuda.amp.GradScaler" not in trn, (
    "torch.cuda.amp.GradScaler is back -- it is REMOVED in torch 2.10")

# ---- Stage-2 code must be wired, not merely configured -------------------
dec = "".join(cells["srdec01"]["source"])
assert "class QueryTensionDecoder" in dec and "def interp_nodes" in dec, \
    "decoder cell srdec01 is not patch-sr-03 -- re-apply _nb_patch_sr_03_decoder.py"
assert "def model_forward(model, sample)" in dec, "model_forward helper missing"
mdl = "".join(cells["f2de07a6"]["source"])
assert "self.query_decoder = QueryTensionDecoder(" in mdl, "model does not build the decoder"
assert "_decode_queries" in mdl, "model has no query-decode step"
assert "query_graph_or_graphs" in mdl, "forward() does not accept a query graph"
bmc = "".join(cells["7a1e407c"]["source"])
assert "use_query_decoder=getattr(cfg" in bmc, "builder does not thread use_query_decoder"
assert "def load_stage1_warm_start" in bmc, "warm-start loader missing"
ds = "".join(cells["a5660cf4"]["source"])
assert "out_N: Optional[int] = None" in ds, "dataset has no out_N -- re-apply _nb_patch_sr_01_dataset.py"
sub = "".join(cells["b18c6667"]["source"])
assert 'out["query_graph"] = self.load_case_datasets[' in sub, "subset does not attach a query graph"
assert 'out["query_graphs"] = [s["query_graph"] for s in batch]' in sub, "collate drops the query graphs"
nrm = "".join(cells["8e7ef9e9"]["source"])
assert "qg.x         = self.static_node_norm.transform(qg.x)" in nrm, \
    "query graph is not standardised with the SAME normaliser as the input graph"
dl = "".join(cells["81d71e4b"]["source"])
assert "_raw.out_N = _sr_out_N" in dl, "out_N is not set on the raw subsets"
# --- patch sr-07: the normalizer must be fitted at STAGE-1 resolution.
# The v5 checkpoint stores NO normalizer, so fitting target_norm with the splits
# already switched to N_out rescales every warm-started / baseline prediction by
# ~6.3 % (measured: mixture 2779.3/2719.5 N vs N_out=21 2651.5/2547.6 N). That is
# the systematic magnitude deficit job 10658637 showed on the DIAGONAL.
# NOTE: a plain `dl.index("_raw.out_N = _sr_out_N") < _i_fit` is NOT a valid test
# here -- it matches the LEGACY guarded branch and passes either way.
assert "# --- patch sr-07:" in dl, \
    "dataloader cell is pre-sr-07 -- re-apply _nb_patch_sr_07_normfit.py"
assert dl.count("fit_normalizers_from_train_subset(train_raw") == 1, \
    "normalizers must be fitted exactly once"
assert dl.count("_raw.out_N = _sr_out_N") == 2, \
    "expected exactly 2 out_N assignments (guarded legacy + active)"
_i_fit = dl.index("fit_normalizers_from_train_subset(train_raw")
_i_active = dl.index("if _sr_on:\n        for _raw in (train_raw, val_raw, test_raw):")
_i_legacy = dl.index("if _sr_on and not _sr_fit_stage1:")
assert _i_fit < _i_active, \
    "the ACTIVE out_N assignment is BEFORE the normalizer fit -- target_norm " \
    "would be fitted at N_out and every prediction rescaled ~6.3 %"
assert _i_legacy < _i_fit, "the guarded legacy sr-06 branch moved"
assert '"superres_fit_norm_at_stage1_resolution", True' in dl, \
    "the sr-07 fit-resolution flag is not read with a True default"
cfgcls = "".join(cells["c53b5783"]["source"])
assert "superres_fit_norm_at_stage1_resolution: bool = True" in cfgcls, \
    "TrainingConfig is missing the sr-07 field or it does not default True"
assert "superres_fit_norm_at_stage1_resolution=False" not in cfg_src, \
    "the run cell disables the sr-07 normalizer fix"
sp = "".join(cells["858f4194"]["source"])
assert "def assert_windows_are_paired" in sp, "paired-window verifier missing -- re-apply _nb_patch_sr_02_paired_windows.py"
assert "if _pair_ncs else _shared_rng)" in sp, "per-(loc,case) RNG missing: node counts would draw DIFFERENT windows"
assert "if int(target_N) not in _use_ncs:" in sp, "split builder does not filter on cfg.node_counts"
assert sp.count("k[2] in _use_ncs]") == 2, \
    "BOTH budget divisors (_train_keys and _test_keys) must filter on cfg.node_counts, " \
    "or a 6-topology run over a 10-topology build silently gets half the pool"
assert "_groups[(_key[0], _key[1], _w)].append" in sp, "cap is not group-wise: pairing would break at the cap"
lss = "".join(cells["9e030261"]["source"])
assert "x_static_raw_q=None" in lss, "loss does not accept query-station statics"
assert "x_seq = interp_nodes(x_seq, n_out)" in lss, "loss does not move the physics geometry to the queries"
ev = "".join(cells["8f268a8b"]["source"])
assert "def _batch_static_raw_q(sample)" in ev, "query-static helper missing"
assert ev.count("x_static_raw_q=_batch_static_raw_q(sample)") == 2, \
    "both criterion call sites must pass the query statics (train + evaluate)"
assert "_frac = interp_nodes(_frac.view(-1, 1), _n_out_r).view(-1)" in ev, \
    "contact-regime R2 still mixes the N_in and N_out node axes"
assert "y_hat = model_forward(model, sample)" in ev, "train loop does not use model_forward"

# ---- patch sr-11: the pre-training diagnostic must move the WHOLE batch ----
# Jobs 117916 / 118966 died here: run_physics_sanity_check hand-rolled the device
# move for x_seq/y_seq/x_raw only, so _batch_static_raw_q(batch) returned the
# query statics still on the CPU and the physics terms mixed cuda+cpu.
phy = "".join(cells["pubphys01"]["source"])
assert "batch = move_sample_to_device(batch, dev)" in phy, (
    "run_physics_sanity_check does not move the whole batch -- this is the "
    "jobs 117916/118966 cuda/cpu split. Re-apply _nb_patch_sr_11_physcheck_device.py")
assert "x_raw = x_raw_q.float().to(dev)" in phy, (
    "the sr-04 query-static assignment does not stay on the criterion's device")

# ---- patch sr-12: the training batch must be the one that fits in 80 GB ----
# Jobs 121890 / 121892 died of CUDA OOM inside epoch 1 at batch 32: Stage 2 pins
# N_out=21, so the LSTM and head run at B*21 sequences on EVERY bucket, and cuDNN
# is disabled here so the ATen LSTM keeps its per-timestep activations.
assert "_BATCH, _ACCUM = (32, 1) if _IS_BASELINE else (16, 2)" in cfg_src, (
    "the sr-12 batch selector is missing or altered -- at batch 32 this run OOMs "
    "inside epoch 1 like jobs 121890/121892 did, an hour in, after staging. "
    "Re-apply _nb_patch_sr_12_batch.py")
assert "batch_size=_BATCH," in cfg_src and "accumulation_steps=_ACCUM," in cfg_src, (
    "TrainingConfig does not consume the sr-12 batch selector")
assert "batch_size=32," not in cfg_src, "a hard-coded batch_size=32 is back in the run cell"
tmc = "".join(cells["d75ee9f2"]["source"])
assert "def evaluate_superres_grid" in tmc, "grid evaluation missing -- re-apply _nb_patch_sr_04_loss_grid.py"
assert "SUPERRES_POOLED_FIELDS" in tmc and "def _grid_value" in tmc, "grid writer is stale"
bse = "".join(cells["srbase01"]["source"])
assert "class InterpolatedStage1Model" in bse, "baseline wrapper missing -- re-apply _nb_patch_sr_05_baseline.py"
assert "model.load_state_dict(sd, strict=True)" in bse, "baseline load is not strict"
assert "is a STAGE-2 checkpoint" in bse, "baseline does not reject a Stage-2 checkpoint by name"
assert "def compare_superres_grids" in bse, "delta-surface comparator missing"

# ---- the inherited v5 data pipeline, unchanged ---------------------------
bl = "".join(cells["f80f8d84"]["source"])
assert "assert len(CORRUPT_CASES_ALL) == 109" in bl, "blacklist is not the 109-case union"
assert "EXCLUDE_ARTIFACT_WINDOWS = True" in bl, "per-window artifact exclusion is OFF"
assert "assert len(CORRUPT_CASES_NONFINITE) == 11" in bl, "non-finite case set missing"
assert "assert len(CORRUPT_CASES_TOO_SHORT) == 2" in bl, "too-short case set missing"
assert "ARTIFACT_TENSION_CEILING_N = 50_000.0" in bl, "50 kN ceiling missing"
_ns = {}
_lo = bl.index("ARTIFACT_SAMPLES = {")
exec(bl[_lo:bl.index("\n}\n", _lo) + 3], _ns)
_A = _ns["ARTIFACT_SAMPLES"]
assert len(_A) == 109, "ARTIFACT_SAMPLES covers %d cases, expected 109" % len(_A)
assert sum(len(v) for v in _A.values()) == 1643, "ARTIFACT_SAMPLES is not the 1643-sample mask"
assert "all_wins = list(range(len(ds)))" not in sp, "OOD BYPASS PRESENT -- re-apply patch 22"
assert sp.count("all_wins = valid_window_starts_inside_raw_interval(") == 1, \
    "held-out branch does not call the artifact filter exactly once"
ts = "".join(cells["pubts01"]["source"])
assert "pin_sampler_epoch" in ts, "sampler pin missing -- per-checkpoint dumps would be incomparable"
assert '_store("sn", sorted(snap_res' in ts, "representative snap-window selector missing (patch 27)"
assert "snap_ev" in ts, "snap capture missing (patch 23)"
fb = "".join(cells["fbac801f"]["source"])
i = fb.index("load_case_datasets = build_lazy_load_case_datasets(")
assert "node_counts=(4, 5, 6, 7, 8, 10, 12, 15, 18, 21)" in fb[i:i + 500], \
    "the build-call must construct ALL 10 node counts -- the grid needs every N_out query graph"
assert "_short_skipped" in fb, "builder has no too-short guard"

# ---- patches sr-17/18/19: the sensor-noise sweep and the per-window stats ---
# Mode-agnostic: all six modes share one notebook, so every assertion below
# holds for every job. They are pinned here because a stale notebook uploaded
# over the fix would otherwise burn ~1 h of gates + staging before failing.
_mode_lines = [l for l in cfg_src.splitlines()
               if l.lstrip().startswith(("_MODE = (", "# _MODE = ("))]
assert len(_mode_lines) == 8, (
    "expected EIGHT _MODE lines (BASELINE, HPO_WARM, HPO_SCRATCH, FINAL_WARM, "
    "FINAL_SCRATCH and the eval-only noise, snap-row and clean modes), found %d"
    % len(_mode_lines))

ds = "".join(cells["a5660cf4"]["source"])
assert "def sensor_noise_is_off(spec)" in ds, \
    "patch sr-17 is missing -- re-apply _nb_patch_sr_17_sensor_noise.py"
assert "def sensor_noise_generator(" in ds and "def sensor_noise_apply(" in ds, \
    "the sr-17 noise helpers are incomplete"
# ** SILENT-FAILURE GUARD ** torch's CPU generator ignores seed bits above bit
# 32, and the injective mix puts the discriminating bits there. Without the
# splitmix64 avalanche, different noise LEVELS share a realisation and the whole
# sweep is void while looking perfectly healthy.
assert "0xBF58476D1CE4E5B9" in ds and "0x94D049BB133111EB" in ds, (
    "sensor_noise_generator has lost its splitmix64 avalanche -- distinct noise "
    "levels would silently share a realisation (torch ignores seed bits > 32)")
assert "torch.randn(sig.shape, generator=gen" in ds, \
    "the noise must be drawn from the LOCAL generator"
assert "torch.manual_seed" not in ds, (
    "the dataset cell must never touch the GLOBAL RNG -- that would shift the "
    "shared stream and move the window draw itself")
# the hook must sit between the resample and the feature stack, or the derived
# columns (velocity, contact, drag) and edge_seq would describe CLEAN geometry
_i_res = ds.index("tension = resample_fe_output(ten_w,")
_i_pos = ds.index("x_abs = sensor_noise_apply(")
_i_vel = ds.index("x_vel = sensor_noise_apply(")
_i_stk = ds.index("dyn = torch.stack(feats, dim=-1)")
assert _i_res < _i_pos < _i_vel < _i_stk, \
    "the sr-17 injection points are out of order in __getitem__"

sub = "".join(cells["b18c6667"]["source"])
assert "_ds.sensor_noise = self.sensor_noise" in sub and "_ds._sn_key = (lc_id, case_id)" in sub, \
    "the noise spec / seeding key is not forwarded to the leaf dataset"

assert "sensor_noise_enabled: bool = False" in cfgcls, \
    "TrainingConfig.sensor_noise_enabled must DEFAULT to False"
assert "sensor_noise_enabled=(_IS_NOISE or _IS_CLEAN)," in cfg_src, (
    "the run cell must tie sensor_noise_enabled to the mode, so no TRAINING "
    "mode can ever silently acquire noise")

assert "per_window_rows: Optional[list] = None," in ev, \
    "patch sr-18 is missing -- re-apply _nb_patch_sr_18_perwindow_repeat.py"
assert "if per_window_rows is not None:" in ev, \
    "the sr-18 per-window block is missing from evaluate()"
assert 'per_window_rows.append({' in ev and '"ss_res": m["ss_res"][_pwn],' in ev, \
    "the sr-18 row does not carry the raw sums -- a subset could not reproduce global_R2"
assert 'test_per_window_stats%s.csv' in tmc, \
    "test_model does not write the per-window statistics"
assert 'def set_sensor_noise(loader, spec):' in tmc, "set_sensor_noise is missing"
assert 'def sensor_noise_spec_absolute(' in tmc and 'def sensor_noise_spec_snr(' in tmc, \
    "the sr-17 level builders are missing"

# ---- the eval-only noise branch -------------------------------------------
assert '_IS_NOISE    = (_TAG == "EVAL_NOISE")' in cfg_src, \
    "the run cell has no _IS_NOISE flag -- re-apply _nb_patch_sr_19_eval_noise_mode.py"
assert "_IS_EVAL     = _IS_BASELINE or _IS_NOISE or _IS_SNAPROWS or _IS_CLEAN" in cfg_src, (
    "the _IS_EVAL flag is missing or does not cover all four eval-only modes")
assert "assert _IS_EVAL or _IS_HPO or _IS_FINAL" in cfg_src, \
    "the unknown-tag guard was not widened for the eval-only noise mode"
assert "_resume_hit = None if _IS_EVAL else" in cfg_src, (
    "an eval-only mode must NEVER auto-resume -- it would pick up another "
    "mode's checkpoint")
assert "elif _IS_NOISE:" in cfg_src, "the eval-only noise branch is missing"
assert "def _pin_test_sampler(_ep=0):" in cfg_src, (
    "the sweep does not pin the bucketed sampler -- it reshuffles on EVERY "
    "pass, so each noise level would be scored on a DIFFERENT draw")
# patch sr-22: scoped to the noise branch -- EVAL_CLEAN repeats these literals,
# so a whole-cell `in` test would keep passing after a regression in this branch
_br19 = cfg_src[cfg_src.index("elif _IS_NOISE:"):cfg_src.index("elif _IS_SNAPROWS:")]
assert "_SN_SIGMAS        = (0.01, 0.05, 0.10, 0.25)" in _br19, \
    "the absolute-sigma level table has moved"
assert "_SN_SNRS_DB       = (20.0, 10.0, 5.0, 0.0)" in _br19, \
    "the SNR level table has moved"
assert "_SR19_REF = {" in cfg_src and '"test_global_R2_tension"' in cfg_src, \
    "the clean-level reproduction reference is missing"
assert 'out_N_values=(21,), tag="_baseline" + _tg,' in _br19, (
    "the in-sweep interpolation baseline must be per-level tagged and "
    "restricted to N_out=21")
# the noise branch must not train
_i_n = cfg_src.index("elif _IS_NOISE:")
_i_t = cfg_src.index("else:\n    model = build_model_from_config(")
assert _i_n < _i_t, "the eval-only branch must precede the training branch"
for _bad in ("train_model(", "calibrate_log_vars(", "optimizer"):
    assert _bad not in cfg_src[_i_n:_i_t], \
        "the eval-only noise branch reaches %r -- it must never train" % _bad

# ---- patch sr-20: EVAL_SNAPROWS -- snap capture at the snap's OWN station ----
# The FINAL dumps store only the LINE MAXIMUM at each snap event. This mode
# re-scores both shipped models on their published draws and stores every
# station over +-2 samples. Mode-agnostic like the sr-17/18/19 block above.
assert '"snap_nbhd_true":    snap_nbhd_true,' in ts and "_SR20_OFF = (-2, -1, 0, 1, 2)" in ts, (
    "patch sr-20 is missing from the dump cell -- re-apply _nb_patch_sr_20_snaprows.py")
assert "_tt20[_j20, :_nst20] = true_w[_r + _o20]" in ts, "the sr-20 dump does not store the true row"
assert "_pp20[_j20, :_nst20] = pred_w[_r + _o20]" in ts, "the sr-20 dump does not store the predicted row"
assert '_IS_SNAPROWS = (_TAG == "EVAL_SNAPROWS")' in cfg_src, "the run cell has no _IS_SNAPROWS flag"
assert "elif _IS_SNAPROWS:" in cfg_src, "the eval-only snap-row branch is missing"
# PASS B must rebuild the ORIGINAL Stage-1 draw, or it scores other events
for lit in ("superres_mode=False, pair_windows_across_node_counts=False,",
            "use_query_decoder=False,",
            "set_seed(cfg_s1.seed_list[0])",
            "_SR20_REF_S1 = {", "_SR20_REF_S2 = {",
            "_SR20_EV_S1 = (1161, 272, '24d319ff6803c4e0')",
            "_SR20_EV_S2 = (1130, 89, '93e52a7979859fda')",
            "_SR20_V5_NORM = (2791.087646484375, 2727.0)",
            "_v520 = torch.load(cfg.superres_baseline_checkpoint,",
            'assert int(_v520.get("epoch", -1)) == 32'):
    assert lit in cfg_src, "the sr-20 branch is missing %r -- re-apply _nb_patch_sr_20_snaprows.py" % lit
_i_s20 = cfg_src.index("elif _IS_SNAPROWS:")
_i_t20 = cfg_src.index("else:\n    model = build_model_from_config(")
assert _i_s20 < _i_t20, "the snap-row branch must precede the training branch"
_br20 = cfg_src[_i_s20:_i_t20]
for _bad in ("train_model(", "calibrate_log_vars(", "optimizer"):
    assert _bad not in _br20, "the snap-row branch reaches %r -- it must never train" % _bad
assert _br20.count("_sr20_pin(test_loader)") == 2 and _br20.count("_sr20_pin(test_loader_s1)") == 2, (
    "each pass must pin the sampler before test_model AND before the dump")

# ---- patch sr-21: site-targeted window reservoirs ---------------------------
# The pooled and per-node-count selectors are blind to SITE, so at four sensors
# loc10 is represented by four windows from two simulations, every one of them
# carrying the out-pulling residue of Appendix D, and loc11 by a single
# snap-bearing window. sr-21 adds a reservoir per (site, node count, class) that
# rejects any window holding a masked artifact sample or an out-pulling sample.
assert "patch sr-21" in ts, (
    "patch sr-21 is missing from the dump cell -- re-apply _nb_patch_sr_21_sitewindows.py")
for lit in ("site_windows=((10, 11), (4,)),",
            "n_site_clean: int = 24,",
            "n_site_snap: int = 12,",
            "site_res, site_seen = {}, {}",
            'f"site{_a21:02d}_n{_b21:02d}_{_c21}"'):
    assert lit in ts, "the sr-21 dump is missing %r" % lit
# both rejection tests must be present, or a window fails only one of them
assert "ARTIFACT_SAMPLES.get((_lc21, _cs21), ())" in ts, \
    "the sr-21 filter does not reject masked artifact samples"
assert "OUTPULL_SAMPLES.get((_lc21, _cs21), ())" in ts, \
    "the sr-21 filter does not reject out-pulling samples"
assert '_cls21 = "sn" if _rows else "cl"' in ts, \
    "the sr-21 filter does not split snap-bearing from snap-free windows"
# the catalogue itself must still be the Appendix-D scan
assert "OUTPULL_SAMPLES = {" in bl, "OUTPULL_SAMPLES is missing from the data cell"
for lit in ("assert len(OUTPULL_SAMPLES) == 55, len(OUTPULL_SAMPLES)",
            "assert sum(len(v) for v in OUTPULL_SAMPLES.values()) == 1047,"):
    assert lit in bl, "the sr-21 out-pulling catalogue is not the Appendix-D one (%r)" % lit
# ...and it must NOT reach the pipeline: reading it anywhere else would move the
# test population while every published metric silently stayed in the CSV
for _cid21, _src21 in (("858f4194", sp), ("a5660cf4", ds),
                       ("fbac801f", fb), ("a04f1f5f", cfg_src)):
    assert "OUTPULL_SAMPLES" not in _src21, (
        "OUTPULL_SAMPLES is read in cell %s -- it must only ever filter the dump, "
        "never the split or the dataset" % _cid21)

# ---- patch sr-22: EVAL_CLEAN -- the EVALUATION mask (the catenary rule) -----
# The shipped models were trained on ARTIFACT_SAMPLES (1643 samples). The
# evaluation mask adds RULE C: a station above 1 kN out-pulling the fairlead at
# the same instant (CLAUDE.md 12.14). It may only ever THIN the TEST draw after
# the draw is built; reaching the split, the dataset, the dump or the loader body
# would move the population the published metrics describe.
assert "CATENARY_SAMPLES = {" in bl and "SNAP_EVENT_TIMES_CAT = {" in bl, (
    "patch sr-22 tables are missing from the data cell -- re-apply _nb_patch_sr_22_evalclean.py")
for lit in ("assert len(CATENARY_SAMPLES) == 80, len(CATENARY_SAMPLES)",
            "assert sum(len(v) for v in CATENARY_SAMPLES.values()) == 7772,",
            "assert sum(len(v) for v in SNAP_EVENT_TIMES_CAT.values()) == 902,"):
    assert lit in bl, "the sr-22 table is not the 2026-09-23 scan (%r)" % lit
for _cid22, _src22 in (("858f4194", sp), ("a5660cf4", ds), ("fbac801f", fb),
                       ("81d71e4b", dl), ("pubts01", ts)):
    for _nm22 in ("CATENARY_SAMPLES", "SNAP_EVENT_TIMES_CAT"):
        assert _nm22 not in _src22, (
            "%s is read in cell %s -- the evaluation mask may only thin the TEST draw, "
            "through build_dataloaders(test_window_drop=...)" % (_nm22, _cid22))
assert "def _sr22_filter_windows(index_pairs, drop, window_len):" in dl, \
    "the sr-22 window filter is missing from the dataloader cell"
assert "    test_window_drop=None,   # patch sr-22" in dl, \
    "build_dataloaders must take test_window_drop with a None default"
assert dl.count("_sr22_filter_windows(split_indices[") == 1, \
    "the sr-22 filter must be applied to exactly one split"
assert 'split_indices["test"] = _sr22_filter_windows(split_indices["test"],' in dl, \
    "the sr-22 filter thins a split other than TEST -- the training pool would move"
_i_cap22 = dl.index("split_indices = _cap_split_windows(split_indices, cfg)")
_i_flt22 = dl.index('split_indices["test"] = _sr22_filter_windows(')
_i_raw22 = dl.index("test_raw  = MultiLoadCaseWindowSubset(")
assert _i_cap22 < _i_flt22 < _i_raw22, (
    "the sr-22 filter must sit AFTER the cap (the published draw is reproduced "
    "first) and BEFORE the test subset is built")
assert "if j < len(b) and b[j] < int(w) + int(window_len):" in dl, \
    "the sr-22 coverage test changed -- a window spans [start, start + window_len)"
assert '_IS_CLEAN    = (_TAG == "EVAL_CLEAN")' in cfg_src, "the run cell has no _IS_CLEAN flag"
assert "elif _IS_CLEAN:" in cfg_src, "the EVAL_CLEAN branch is missing"
_i_c22 = cfg_src.index("elif _IS_CLEAN:")
_i_t22 = cfg_src.index("else:\n    model = build_model_from_config(")
assert _i_c22 < _i_t22, "the EVAL_CLEAN branch must precede the training branch"
_br22 = cfg_src[_i_c22:_i_t22]
for _bad in ("train_model(", "calibrate_log_vars(", "optimizer"):
    assert _bad not in _br22, "the EVAL_CLEAN branch reaches %r -- it must never train" % _bad
assert _br22.count("test_window_drop=CATENARY_SAMPLES,") == 2, (
    "both clean builds (Stage 2 and Stage 1) must drop the evaluation-mask windows")
assert _br22.count("SNAP_EVENT_TIMES = SNAP_EVENT_TIMES_CAT") == 3, (
    "all three clean dumps must join on the evaluation catalogue")
assert _br22.count("SNAP_EVENT_TIMES = _SR22_SNAP_PUB") == 3, (
    "the published catalogue must be restored after every clean dump")
for lit in ("pair_windows_across_node_counts=False, sensor_noise_enabled=False,",
            "_SR22_REF_S1 = {", "_SR22_REF_S2 = {",
            "_SR22_V5_NORM = (2791.087646484375, 2727.0)",
            "_SR22_DRAW_S2 = (50000, 'fb681c574549d8cb')",
            "_SR22_DRAW_S1 = (50000, 'cf2d297458b7d78b')",
            "_SR22_DROP_S2 = 470", "_SR22_DROP_S1 = 487",
            "_SR22_EV_S2 = (500, 45)", "_SR22_EV_S1 = (523, 154)",
            'assert int(_v522.get("epoch", -1)) == 32',
            "grid_rows = evaluate_superres_grid(",
            "baseline_rows = run_interpolation_baseline(",
            'out_N_values=(21,), tag="_baseline" + _tg,',
            "_SN_SIGMAS        = (0.01, 0.05, 0.10, 0.25)",
            "_SN_SNRS_DB       = (20.0, 10.0, 5.0, 0.0)"):
    assert lit in _br22, "the sr-22 branch is missing %r -- re-apply _nb_patch_sr_22_evalclean.py" % lit
assert _br22.count("_sr22_pin(test_loader_c)") >= 6, \
    "every clean Stage-2 pass must pin the sampler"

# ---- the ACTIVE mode's checkpoint must be here BEFORE the 47-minute stage ---
import ast as _ast_ck
import os as _os_ck
for _nd in _ast_ck.walk(_ast_ck.parse(cfg_src)):
    if (isinstance(_nd, _ast_ck.Assign)
            and getattr(_nd.targets[0], "id", "") == "_MODE"):
        _mv = _ast_ck.literal_eval(_nd.value)
        if _mv[1]:
            _mp = str(_mv[1]).lstrip("./")
            if _os_ck.path.exists(_mp):
                print("[sel]   active-mode checkpoint present: %s" % _mp)
            elif _os_ck.environ.get("SR_GATE_LOCAL", "") == "1":
                # Local dry-run escape hatch, same idea as SR_OPS_ALLOW_CPU=1:
                # a submit directory holds the checkpoints, a laptop does not.
                # NEVER set this on the cluster.
                print("[sel]   WARNING: %s absent (SR_GATE_LOCAL=1 dry-run)" % _mp)
            else:
                raise AssertionError(
                    "the active mode needs %s, which is not in this directory. "
                    "Upload it -- GATE 0 only guards the v5 checkpoint." % _mp)

print("[sel] STAGE-2 mode %s verified: query decoder wired, paired windows, "
      "physics at the query stations, grid + interpolation baseline present, "
      "PC1 dims, selection v4, LR fix, v5 pipeline (109/1643, OOD filtered, "
      "50 kN ceiling) -> %s" % (tag, ckpt))
PY

# --- HARD GATE 3: record-length preflight (patch 21) ------------------
"${VENV_PY}" - "${NOTEBOOK}" "${SRC_CACHE}" <<'PY' || die "record-length preflight failed -- a non-excluded case is shorter than window_len (see [len] lines)"
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
if bad:
    for loc, case, n in bad:
        print(f"[len] TOO SHORT: loc{loc:02d} case{case:04d} = {n} steps < {W}")
assert not bad, (f"{len(bad)} non-excluded case(s) are shorter than window_len={W} -- "
                 "add them to CORRUPT_CASES_TOO_SHORT (cell f80f8d84)")
print(f"[len] OK: all {seen} in-use cases have >= {W} steps")
PY

# --- HARD GATE 4-ops: does every kernel this pipeline needs RUN here? -
# The last cheap check before the 13 GB stage, and the one that closes the
# class of failure that has cost four submissions:
#   100111  nofile 1024        | 103567  cuDNN rejects the LSTM
#   103651  nvrtc cannot compile abs() on a complex tensor
# All three died INSIDE the notebook, after every gate AND after the stage.
# This gate execs the notebook's 22 pure-definition cells (no data is touched),
# lifts the REAL architecture out of the run cell, builds the real model and the
# real Kendall loss, and runs forward + backward + a zero-bin gradient probe on
# synthetic tensors on THIS GPU. Seconds, versus the ~30 min it protects.
# Body generated from sr_ops_preflight.py -- edit that file, not this block.
"${VENV_PY}" - "${NOTEBOOK}" <<'PY' || die "numerical preflight failed (see [ops] lines) -- a torch/PyG kernel this pipeline needs does not run on this node. This is an ENVIRONMENT defect, not a Stage-2 code regression: the same model and loss trained for 7.5 h as job 10532625 under the old module stack. Do NOT resubmit unchanged"
"""HARD GATE 4-ops -- generated from sr_ops_preflight.py. Edit that file."""

import ast
import json
import os
import pathlib
import sys
import time
import traceback

# ---------------------------------------------------------------- cell policy
# The 22 pure-definition cells, in notebook order. Verified: zero top-level
# calls, zero top-level control flow, zero I/O in top-level assignments.
DEF_CELLS = [
    "6a5be23c", "66cf027f", "4d3988c6", "a5660cf4", "bcd95251", "28a9e4ff",
    "srdec01", "f2de07a6", "c53b5783", "858f4194", "b18c6667", "8e7ef9e9",
    "81d71e4b", "9e030261", "pubphys01", "7a1e407c", "8f268a8b", "c62bc05c",
    "d75ee9f2", "srbase01", "pubts01", "71b74ab8",
]
# Never executed: side effects, or a data-cache dependency.
FORBIDDEN = {"f80f8d84", "fbac801f", "a04f1f5f", "07257ee8", "19c37b28"}


class _SkipProbe(Exception):
    """CPU dry-run marker -- not an error."""


def fail(msg):
    print(f"[ops] FAIL: {msg}")
    raise SystemExit(1)


def main():
    if len(sys.argv) < 2:
        fail("usage: sr_ops_preflight.py <notebook.ipynb>")
    nb_path = pathlib.Path(sys.argv[1])
    if not nb_path.exists():
        fail(f"notebook not found: {nb_path}")

    import torch

    # Mirror the run cell's device setup EXACTLY, so the gate tests the same
    # configuration the notebook will use -- including GATE 1-rnn's verdict.
    disable_cudnn = os.environ.get("SR_DISABLE_CUDNN", "") == "1"
    if disable_cudnn:
        torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # SR_OPS_ALLOW_CPU=1 is for developing this gate on a laptop. On the cluster
    # it is never set, so a node without a usable GPU is a hard failure.
    allow_cpu = os.environ.get("SR_OPS_ALLOW_CPU", "") == "1"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)
    elif allow_cpu:
        device = torch.device("cpu")
        dev_name = "CPU (SR_OPS_ALLOW_CPU=1 -- local dry-run, NOT a cluster check)"
    else:
        fail("no CUDA device -- this gate must run on the GPU the job will use")
    print(f"[ops] device {dev_name} | torch {torch.__version__} "
          f"| cudnn.enabled={torch.backends.cudnn.enabled} "
          f"(SR_DISABLE_CUDNN={os.environ.get('SR_DISABLE_CUDNN', '')!r})")

    # ------------------------------------------------------ exec the def cells
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = {c.get("id"): "".join(c["source"])
             for c in nb["cells"] if c["cell_type"] == "code"}

    missing = [c for c in DEF_CELLS if c not in cells]
    if missing:
        fail(f"definition cells missing from the notebook: {missing} -- the "
             "notebook was restructured and this gate needs updating")
    overlap = set(DEF_CELLS) & FORBIDDEN
    if overlap:
        fail(f"refusing to run: {overlap} is both a definition and a forbidden cell")

    t0 = time.time()
    ns = {"__name__": "__sr_ops_preflight__"}
    for cid in DEF_CELLS:
        try:
            exec(compile(cells[cid], f"cell_{cid}", "exec"), ns)
        except Exception:
            traceback.print_exc()
            fail(f"definition cell {cid} does not even execute -- the environment "
                 "cannot import something the notebook needs")
    print(f"[ops] {len(DEF_CELLS)} definition cells executed in {time.time()-t0:.1f} s "
          f"(no data touched; {sorted(FORBIDDEN)} skipped)")

    # ------------------------------------------------- config + feature widths
    # The gate must test the architecture the job will ACTUALLY run, not
    # TrainingConfig's defaults. Job 103567's cuDNN rejection was specific to the
    # 2-layer bidirectional 384 LSTM -- a default 1-layer 128 unidirectional one
    # may well be accepted, so testing defaults would have missed it entirely.
    # The run cell (a04f1f5f) is NOT executed here; its TrainingConfig(...) call
    # is parsed and only LITERAL keyword values are lifted, so the gate follows
    # the notebook automatically if the dims ever change.
    cfg = ns["TrainingConfig"]()
    ARCH_KEYS = (
        "gat_hidden_dim", "gat_out_dim", "num_heads", "gat_dropout",
        "lstm_hidden_dim", "num_lstm_layers", "lstm_dropout", "lstm_bidirectional",
        "head_hidden_dim", "head_dropout", "use_global_context",
        "query_decoder_heads", "query_decoder_ffn_dim", "query_decoder_fourier_bands",
        "query_decoder_dropout", "query_decoder_residual_interp",
        "query_decoder_zero_init", "superres_out_N", "superres_grid_out_N",
        "use_amp", "peak_pinball_tau", "peak_pinball_thr_n",
        "constitutive_incremental", "batch_size",
    )
    lifted = {}
    for node in ast.walk(ast.parse(cells["a04f1f5f"])):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "TrainingConfig":
            for kw in node.keywords:
                if kw.arg in ARCH_KEYS:
                    try:
                        lifted[kw.arg] = ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        pass          # a Name/expression -- keep the cfg default
            break
    if len(lifted) < 15:
        fail(f"only {len(lifted)} architecture literals lifted from the run cell "
             "-- TrainingConfig(...) has been restructured; update this gate "
             "rather than silently preflighting the wrong model")
    for k, v in lifted.items():
        setattr(cfg, k, v)
    cfg.use_query_decoder = True          # Stage 2: exercise the decoder path

    # --- patch sr-12: the REAL training shape, for the VRAM probe below.
    # batch_size and window_len are Names in the run cell (_BATCH / WINDOW_LEN),
    # so literal_eval cannot reach them and the loop above kept the defaults.
    # Resolve them the way the job will: take the _BATCH branch that matches the
    # ACTIVE _MODE, and read WINDOW_LEN from the (never executed) build cell.
    _run_ast = ast.parse(cells["a04f1f5f"])
    _tag_active = None
    for node in ast.walk(_run_ast):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "_MODE" for t in node.targets)):
            try:
                _tag_active = ast.literal_eval(node.value)[0]
            except (ValueError, SyntaxError, IndexError):
                pass
    if _tag_active is None:
        fail("no active _MODE assignment found in a04f1f5f -- GATE 2 should have "
             "rejected this notebook before the gate ever ran")
    _is_baseline = (_tag_active == "BASELINE")
    # patch sr-19: EVAL_NOISE is the SECOND eval-only mode. It takes the (16, 2)
    # branch of _BATCH/_ACCUM like the training modes -- so _is_baseline above
    # must stay exact for the lift -- but it runs under no_grad and allocates no
    # backward graph, so the VRAM probe below must skip it too.
    _is_eval_only = _tag_active in ("BASELINE", "EVAL_NOISE", "EVAL_SNAPROWS", "EVAL_CLEAN")
    _batch = _accum = None
    for node in ast.walk(_run_ast):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Tuple)
                and [getattr(e, "id", "") for e in node.targets[0].elts]
                    == ["_BATCH", "_ACCUM"]):
            v = node.value
            if not isinstance(v, ast.IfExp):
                fail("_BATCH/_ACCUM is no longer a mode conditional -- update this gate")
            try:
                _batch, _accum = ast.literal_eval(v.body if _is_baseline else v.orelse)
            except (ValueError, SyntaxError):
                fail("_BATCH/_ACCUM branches are not literal tuples -- update this gate")
    if _batch is None:
        fail("patch sr-12's _BATCH/_ACCUM selector is missing from a04f1f5f -- "
             "without it the run trains at batch 32 and OOMs the way jobs "
             "121890/121892 did, but only after the ~55 min stage")
    _win = None
    for node in ast.walk(ast.parse(cells["fbac801f"])):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "WINDOW_LEN" for t in node.targets)):
            _win = ast.literal_eval(node.value)
    if _win is None:
        fail("WINDOW_LEN literal not found in fbac801f -- update this gate")
    print(f"[ops] production training shape: mode {_tag_active} | batch {_batch} "
          f"x accum {_accum} (effective {_batch * _accum}) | window_len {_win}")

    # Derive the dynamic-feature width from the notebook's OWN constants rather
    # than hard-coding 22, and assert the formula we mirror is still the one in
    # the dataset, so this can never drift silently.
    ds_src = cells["a5660cf4"]
    if "n_base = (5 + (2 if use_velocity_features else 0)" not in ds_src:
        fail("the dynamic-feature formula in a5660cf4 has moved -- update this gate")
    n_dyn = (5
             + (2 if cfg.use_velocity_features else 0)
             + (2 if cfg.use_drag_features else 0)
             + (len(ns["ENV_COL_NAMES"]) if cfg.use_env_features else 0))
    n_edge_dyn = 4
    N_OUT = int(getattr(cfg, "superres_out_N", 21))
    N_IN_MIN = int(min(getattr(cfg, "superres_grid_out_N", (4,))))
    print(f"[ops] lifted {len(lifted)} arch literals from a04f1f5f")
    print(f"[ops] cfg: gat {cfg.gat_hidden_dim}/{cfg.gat_out_dim} heads {cfg.num_heads} | "
          f"lstm {cfg.lstm_hidden_dim}x{cfg.num_lstm_layers} bidir={cfg.lstm_bidirectional} | "
          f"global_ctx={cfg.use_global_context} | dyn feats {n_dyn}, edge dyn {n_edge_dyn} | "
          f"N_in {N_IN_MIN}..{N_OUT} -> N_out {N_OUT} | use_amp={cfg.use_amp}")

    # ------------------------------------------------------- synthetic geometry
    def make_graph(n):
        """A physically sane catenary-ish line at n stations -- no data needed."""
        s = torch.linspace(0.0, 1.0, n)
        x0 = 95.0 * s                                    # [m] horizontal
        z0 = -50.0 + 50.0 * s ** 2                       # [m] anchor on the bed
        t0 = 2000.0 + 6000.0 * s                         # [N] rising to fairlead
        s0 = 105.0 * s                                   # [m] arc length, L0=105
        g = ns["build_mooring_graph"](x0, z0, t0, s0, water_depth=50.0)
        g.x_raw = g.x.clone()                            # NormalizedSubset does this
        g.edge_attr_raw = g.edge_attr.clone()
        return g

    try:
        g_in4 = make_graph(N_IN_MIN)
        g_in21 = make_graph(N_OUT)
        g_q21 = make_graph(N_OUT)
    except Exception:
        traceback.print_exc()
        fail("build_mooring_graph failed on synthetic input")
    n_static = g_in21.x.shape[1]
    n_static_edge = g_in21.edge_attr.shape[1]
    print(f"[ops] synthetic graphs built: static node feats {n_static}, "
          f"static edge feats {n_static_edge}")

    # ---------------------------------------------------------------- the model
    def build(n_static_, n_static_edge_):
        return ns["MooringGATLSTM"](
            n_static_node_features=n_static_,
            n_dynamic_node_features=n_dyn,
            n_static_edge_features=n_static_edge_,
            n_dynamic_edge_features=n_edge_dyn,
            output_dim=1,
            gat_hidden_dim=cfg.gat_hidden_dim, gat_out_dim=cfg.gat_out_dim,
            num_heads=cfg.num_heads, gat_dropout=cfg.gat_dropout,
            gat_use_layernorm=True, add_residual_projection=True,
            use_global_context=cfg.use_global_context,
            lstm_hidden_dim=cfg.lstm_hidden_dim,
            num_lstm_layers=cfg.num_lstm_layers,
            lstm_dropout=cfg.lstm_dropout,
            bidirectional=cfg.lstm_bidirectional,
            lstm_use_layernorm=True,
            head_hidden_dim=cfg.head_hidden_dim, head_dropout=cfg.head_dropout,
            use_query_decoder=True,
            query_decoder_heads=getattr(cfg, "query_decoder_heads", 4),
            query_decoder_ffn_dim=getattr(cfg, "query_decoder_ffn_dim", 256),
            query_decoder_fourier_bands=getattr(cfg, "query_decoder_fourier_bands", 8),
            query_decoder_dropout=getattr(cfg, "query_decoder_dropout", 0.1),
            query_decoder_residual_interp=True,
            query_decoder_zero_init=True,
        )

    # .to(device) is where job 103567 died -- cuDNN flattens the LSTM weights here.
    try:
        model = build(n_static, n_static_edge).to(device)
    except Exception:
        traceback.print_exc()
        fail("the model cannot be moved to the GPU (this is where job 103567 died: "
             "cuDNN rejecting the LSTM in flatten_parameters)")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[ops] OK: model built and moved to {device} -- {n_par:,} parameters")

    # -------------------------------------------------------- synthetic batch
    B, W = 2, 64
    def make_batch(g_in, n_in, n_out):
        x_seq = torch.randn(B, W, n_in, n_dyn, device=device) * 0.5
        # columns 0/1 are x/z position; keep them physical so the physics terms
        # (catenary, momentum, constitutive) see a sane geometry rather than noise
        x_seq[..., 0] = g_in.x_raw[:, 0].to(device).view(1, 1, n_in) * 95.0
        x_seq[..., 1] = torch.linspace(-50.0, 0.0, n_in, device=device).view(1, 1, n_in)
        x_seq[..., 4] = 0.0                          # contact_flag: nothing grounded
        e_in = g_in.edge_index.shape[1]
        edge_seq = torch.randn(B, W, e_in, n_edge_dyn, device=device) * 0.1
        y_true = torch.randn(B, W, n_out, 1, device=device) * 0.3
        return x_seq, edge_seq, y_true

    def run_forward(g_in, n_in, g_q, n_out, label):
        x_seq, edge_seq, y_true = make_batch(g_in, n_in, n_out)
        gi = [g_in.to(device) for _ in range(B)]
        gq = [g_q.to(device) for _ in range(B)]
        y = model(gi, x_seq, edge_seq, gq)
        if tuple(y.shape) != (B, W, n_out, 1):
            fail(f"{label}: forward returned {tuple(y.shape)}, expected {(B, W, n_out, 1)}")
        if not bool(torch.isfinite(y).all()):
            fail(f"{label}: forward produced non-finite values")
        return x_seq, y, y_true

    for label, (g_in, n_in, g_q, n_out) in {
        f"diagonal     N_in={N_OUT} -> N_out={N_OUT}": (g_in21, N_OUT, g_q21, N_OUT),
        f"off-diagonal N_in={N_IN_MIN} -> N_out={N_OUT}": (g_in4, N_IN_MIN, g_q21, N_OUT),
    }.items():
        try:
            x_seq, y_pred, y_true = run_forward(g_in, n_in, g_q, n_out, label)
        except Exception:
            traceback.print_exc()
            fail(f"model forward failed on the {label} case")
        print(f"[ops] OK: forward {label} -> {tuple(y_pred.shape)}")

    # ------------------------------------------------------------- the LOSS
    # This is where job 103651 died: term 3 takes abs() of an rfft, and abs() on
    # a COMPLEX tensor is jiterated, i.e. compiled at run time by nvrtc.
    Std = ns["FeatureStandardizer"]
    node_norm = Std(feature_idx=tuple(range(n_dyn)))
    node_norm.mean = torch.zeros(n_dyn)
    node_norm.std = torch.ones(n_dyn)
    target_norm = Std(feature_idx=(0,))
    target_norm.mean = torch.tensor([2791.0876])   # v5's real constants, so the
    target_norm.std = torch.tensor([2727.0000])    # physics terms see real scale

    try:
        criterion = ns["ReconstructionKendallLoss"](
            node_norm=node_norm, target_norm=target_norm, cfg=cfg).to(device)
    except Exception:
        traceback.print_exc()
        fail("ReconstructionKendallLoss could not be constructed")
    print(f"[ops] loss built: {criterion.n_terms} terms {list(criterion.TERM_NAMES)}")

    x_seq, edge_seq, y_true = make_batch(g_in4, N_IN_MIN, N_OUT)
    gi = [g_in4.to(device) for _ in range(B)]
    gq = [g_q21.to(device) for _ in range(B)]
    y_pred = model(gi, x_seq, edge_seq, gq)

    y_true[0] = 0.25          # a constant TARGET: exercises the st branch's value path

    x_raw = g_in4.x_raw.to(device).unsqueeze(0).expand(B, -1, -1).contiguous()
    x_raw_q = g_q21.x_raw.to(device).unsqueeze(0).expand(B, -1, -1).contiguous()

    try:
        loss, term_log = criterion(y_pred.float(), y_true.float(), x_seq.float(),
                                   x_raw.float(), x_static_raw_q=x_raw_q.float())
    except Exception:
        traceback.print_exc()
        fail("the Kendall loss forward failed -- this is where job 103651 died "
             "(term 3, torch.abs on the complex rfft output, needs nvrtc)")
    if not bool(torch.isfinite(loss)):
        fail(f"loss forward produced a non-finite value: {loss}")
    nonfinite = [k for k, v in term_log.items()
                 if isinstance(v, float) and v != v]
    if nonfinite:
        fail(f"non-finite loss terms: {nonfinite}")
    print(f"[ops] OK: loss forward = {float(loss.detach()):.6f} "
          f"(all {criterion.n_terms} terms finite, incl. the spectral rfft term)")

    # ------------------------------------------------------------- BACKWARD
    # The HPO and FINAL arms differentiate this loss. A NaN here would poison a
    # 20-hour run, and BASELINE runs under no_grad so it would not surface.
    try:
        loss.backward()
    except Exception:
        traceback.print_exc()
        fail("loss.backward() failed -- a backward kernel is missing")
    bad = [n for n, p in model.named_parameters()
           if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
    if bad:
        fail(f"non-finite GRADIENTS in {len(bad)} tensors, first: {bad[:3]} -- "
             "the constant-tension sample produced exactly-zero rfft bins and the "
             "complex-magnitude formulation is not differentiable there")
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"[ops] OK: backward ran; {n_grad} parameter grads all finite")

    # ---------------------------------------- zero-bin gradient probe (term 3)
    # A CONSTANT prediction makes every non-DC rfft bin exactly 0+0j, where the
    # complex magnitude is not differentiable. torch.abs and linalg.vector_norm
    # return a finite zero subgradient there; hypot(re,im) and sqrt(re*re+im*im)
    # -- the two obvious nvrtc-free rewrites -- return NaN and would silently
    # poison a 20-hour training run.
    # The constant MUST be on the prediction: y_true is a leaf, so a constant
    # target exercises the value path but never the gradient path.
    if "spectral" not in list(criterion.TERM_NAMES):
        fail("the loss no longer has a 'spectral' term -- update this gate")
    # Use the full production call path (not the private _compute_terms, which
    # skips the query-station interpolation and mismatches shapes off-diagonal).
    yp_const = torch.full_like(y_pred.detach(), 0.3).requires_grad_(True)
    try:
        loss_c, _ = criterion(yp_const, y_true.float(), x_seq.float(),
                              x_raw.float(), x_static_raw_q=x_raw_q.float())
        loss_c.backward()
    except Exception:
        traceback.print_exc()
        fail("the loss is not differentiable on a constant prediction")
    if yp_const.grad is None or not bool(torch.isfinite(yp_const.grad).all()):
        n_nan = 0 if yp_const.grad is None else int(torch.isnan(yp_const.grad).sum())
        fail(f"NON-FINITE gradient ({n_nan} NaN) on a "
             "CONSTANT prediction, i.e. exactly-zero rfft bins. The complex "
             "magnitude must be computed with something finite at the origin: "
             "torch.abs (needs nvrtc) or torch.linalg.vector_norm(view_as_real). "
             "hypot(re,im) and sqrt(re*re+im*im) are NOT safe here")
    print(f"[ops] OK: spectral term differentiable at exactly-zero rfft bins "
          f"(constant prediction, grad finite) -- the sr-09 formulation is sound")

    # ------------------------------------------------------------- METRICS
    # evaluate() runs immediately after the loss, so this is the next thing in
    # execution order that could hit a missing kernel (quantiles, masked means,
    # the per-node R2 reduction). Synthesising it costs milliseconds.
    try:
        mets = ns["compute_physical_metrics_per_target"](
            y_pred.detach().float(), y_true.float(), target_norm, ["tension"])
    except Exception:
        traceback.print_exc()
        fail("compute_physical_metrics_per_target failed -- the metric path "
             "runs right after the loss in evaluate()")

    def _flat(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from _flat(v, f"{pre}{k}.")
            else:
                yield f"{pre}{k}", v
    flat = dict(_flat(mets))
    nf = [k for k, v in flat.items()
          if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf")))]
    if nf:
        fail(f"non-finite metrics: {nf[:6]}")

    # The tail statistics live in a nested helper inside evaluate(); exercise the
    # kernels they use (quantile + masked reductions) directly instead.
    try:
        pk_t = torch.rand(4096, device=device) * 20000.0
        se = torch.randn(4096, device=device) * 500.0
        for q in (0.90, 0.99):
            thr = torch.quantile(pk_t, q)
            m = pk_t >= thr
            _ = float(se[m].mean()), float(se[m].abs().mean()), float((se[m] < 0).float().mean())
    except Exception:
        traceback.print_exc()
        fail("the tail-statistic kernels (quantile + masked reductions) fail on "
             "this node -- up90/up99/undermag could not be computed")
    print(f"[ops] OK: metrics path -- {len(flat)} values all finite; "
          "quantile + masked tail reductions at p90/p99")

    # ------------------------------------- the pre-training DATA-PATH functions
    # Added after jobs 117916 / 118966 (2026-09-07). Both died ~4 min into the
    # notebook -- after every gate AND after the ~55 min stage -- with
    #     RuntimeError: Expected all tensors to be on the same device,
    #                   but found at least two devices, cuda:0 and cpu!
    # in run_physics_sanity_check: it was the only loader loop that did not call
    # move_sample_to_device(), so patch sr-04's `_batch_static_raw_q(batch)` read
    # the query-station statics off the UN-MOVED (CPU) batch while x_seq was on
    # the GPU. Everything above this line uses tensors the gate itself allocated
    # on `device`, so it could never have seen it; the CPU smoke test could not
    # either, because on CPU there is no second device to mix in.
    #
    # So: feed the real functions a real loader-shaped batch built on the CPU,
    # exactly as a DataLoader would, and let them do their own device handling.
    # Off-diagonal AND diagonal, because the sr-04 branch only fires when
    # N_in != N_out. Cost: ~1 s. It would have caught 117916/118966 outright.
    import io
    import contextlib

    def make_cpu_batch(g_in, n_in, g_q, n_out):
        """A bucketed batch as collate() hands it over: every tensor on the CPU."""
        xs = torch.randn(B, W, n_in, n_dyn) * 0.5
        xs[..., 0] = torch.linspace(0.0, 95.0, n_in).view(1, 1, n_in)
        xs[..., 1] = torch.linspace(-50.0, 0.0, n_in).view(1, 1, n_in)
        xs[..., 4] = 0.0                     # contact_flag: nothing grounded
        b = {
            "x_seq": xs,
            "edge_seq": torch.randn(B, W, g_in.edge_index.shape[1], n_edge_dyn) * 0.1,
            "y_seq": torch.randn(B, W, n_out, 1) * 0.3,
            "start_idx": [0] * B, "lc_id": [1] * B, "case_id": [1] * B,
            "node_count": n_in, "out_N": n_out,
            "graphs": [g_in.cpu() for _ in range(B)],
            # build_dataloaders sets out_N on ALL splits in Stage-2 mode, and
            # MooringSequenceDataset attaches the query graph whenever out_N is
            # not None -- so a DIAGONAL sample carries one too. Mirror that; a
            # diagonal batch without it is not a shape production ever produces.
            "query_graphs": [g_q.cpu() for _ in range(B)],
        }
        return b

    for _label, (_gi, _ni, _gq, _no) in {
        f"off-diagonal N_in={N_IN_MIN} -> N_out={N_OUT}": (g_in4, N_IN_MIN, g_q21, N_OUT),
        f"diagonal     N_in={N_OUT} -> N_out={N_OUT}": (g_in21, N_OUT, g_q21, N_OUT),
    }.items():
        _loader = [make_cpu_batch(_gi, _ni, _gq, _no)]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                # its printed diagnostic is meaningless on synthetic data; the
                # question is only whether it RUNS with a CPU batch + GPU model.
                ns["run_physics_sanity_check"](_loader, criterion,
                                               target_norm, max_batches=1)
        except Exception:
            traceback.print_exc()
            fail(f"run_physics_sanity_check failed on the {_label} case -- this is "
                 "where jobs 117916/118966 died (a loader batch stays on the CPU "
                 "while the criterion's buffers are on the GPU). Every function "
                 "that consumes a loader batch must call move_sample_to_device().")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                criterion.calibrate_log_vars(model, _loader, device, max_batches=1)
        except Exception:
            traceback.print_exc()
            fail(f"calibrate_log_vars failed on the {_label} case -- it is the very "
                 "next call after run_physics_sanity_check in the run cell")
        print(f"[ops] OK: pre-training data path ({_label}) -- "
              "run_physics_sanity_check + calibrate_log_vars on a CPU loader batch")

    # ------------------------------------------------- VRAM at the REAL shape
    # --- patch sr-12. Jobs 121890/121892 died of CUDA OOM inside EPOCH 1, after
    # every gate AND after the ~55 min stage. Everything above this point runs at
    # B=2, W=64 -- under 1/250th of the production activation footprint -- so no
    # check in this gate could possibly have seen it.
    #
    # Stage 2 pins N_out=21, so the LSTM and the head run at B*21 sequences on
    # EVERY bucket (Stage 1 ran them at B*N_in, so only the N=21 bucket was ever
    # this big), and GATE 1-rnn has to disable cuDNN here -- its fused RNN kept a
    # compact reserve where the ATen fallback keeps per-timestep activations.
    #
    # So: one forward + loss + backward at the ACTUAL batch, the ACTUAL window
    # length and the PEAK bucket (largest N_in), reporting peak VRAM. A couple of
    # seconds against the ~55 min stage it protects.
    if device.type != "cuda":
        print(f"[ops] SKIP: VRAM probe (CPU dry-run -- a batch {_batch}, W={_win} "
              "forward on the CPU is a cluster-scale workload and must never run "
              "on a laptop)")
    elif _is_eval_only:
        print(f"[ops] SKIP: VRAM probe ({_tag_active} trains nothing -- it runs "
              "under no_grad and never allocates a backward graph)")
    else:
        _n_peak = N_OUT                    # the largest N_in the job will bucket
        try:
            g_peak = make_graph(_n_peak)
        except Exception:
            traceback.print_exc()
            fail("could not build the peak-bucket graph for the VRAM probe")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _total = torch.cuda.get_device_properties(device).total_memory
        try:
            _xs = torch.randn(_batch, _win, _n_peak, n_dyn, device=device) * 0.5
            _xs[..., 0] = torch.linspace(0.0, 95.0, _n_peak, device=device).view(1, 1, _n_peak)
            _xs[..., 1] = torch.linspace(-50.0, 0.0, _n_peak, device=device).view(1, 1, _n_peak)
            _xs[..., 4] = 0.0
            _es = torch.randn(_batch, _win, g_peak.edge_index.shape[1],
                              n_edge_dyn, device=device) * 0.1
            _yt = torch.randn(_batch, _win, N_OUT, 1, device=device) * 0.3
            _gi = [g_peak.to(device) for _ in range(_batch)]
            _gq = [g_q21.to(device) for _ in range(_batch)]
            _xr = torch.stack([g.x_raw for g in _gi]).float()
            _xrq = torch.stack([g.x_raw for g in _gq]).float()
            model.zero_grad(set_to_none=True)
            _yh = model(_gi, _xs, _es, _gq)
            _l, _ = criterion(_yh.float(), _yt.float(), _xs.float(), _xr.float(),
                              x_static_raw_q=_xrq.float())
            _l.backward()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            _pk = torch.cuda.max_memory_allocated()
            fail(f"CUDA OUT OF MEMORY at the production training shape "
                 f"(batch {_batch}, window_len {_win}, N_in={_n_peak} -> "
                 f"N_out={N_OUT}): peaked at {_pk / 2**30:.1f} GiB of "
                 f"{_total / 2**30:.1f} GiB. This is exactly how jobs "
                 "121890/121892 died, except they did it an hour in, after "
                 "staging. Lower _BATCH in cell a04f1f5f (patch sr-12) and raise "
                 "_ACCUM to keep the effective batch -- memory is linear in B.")
        except Exception:
            traceback.print_exc()
            fail("the production-shape forward/backward failed for a reason other "
                 "than memory")
        _peak = torch.cuda.max_memory_allocated()
        _frac = _peak / _total
        model.zero_grad(set_to_none=True)
        del _xs, _es, _yt, _gi, _gq, _xr, _xrq, _yh, _l
        torch.cuda.empty_cache()
        print(f"[ops] OK: fwd+loss+bwd at the PRODUCTION shape (batch {_batch}, "
              f"W={_win}, N_in={_n_peak} -> N_out={N_OUT}) -- peak VRAM "
              f"{_peak / 2**30:.1f} GiB of {_total / 2**30:.1f} GiB "
              f"({_frac * 100:.0f} %)")
        if _frac > 0.85:
            fail(f"peak VRAM is {_frac * 100:.0f} % of the card at the production "
                 "shape. It fits in this probe, but the real run adds the optimizer "
                 "state, allocator fragmentation and cached blocks from other "
                 "buckets, so it WILL OOM mid-epoch. Lower _BATCH / raise _ACCUM "
                 "in cell a04f1f5f.")

    # ------------------------------------------- AMP, as the train loop uses it
    # The run cell sets use_amp=False, so autocast is NOT on the production path.
    # Probing it anyway would risk dying for a reason the job never hits, so it
    # runs only when the notebook actually asks for it.
    if device.type != "cuda":
        print("[ops] SKIP: autocast + GradScaler (CPU dry-run)")
    elif not getattr(cfg, "use_amp", False):
        print("[ops] SKIP: autocast + GradScaler (the run cell sets use_amp=False, "
              "so mixed precision is not on this job's path)")
    else:
        try:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            with torch.cuda.amp.autocast(enabled=True):
                y2 = model(gi, x_seq, edge_seq, gq)
                l2, _ = criterion(y2.float(), y_true.float(), x_seq.float(),
                                  x_raw.float(), x_static_raw_q=x_raw_q.float())
            scaler.scale(l2).backward()
            print(f"[ops] OK: autocast + GradScaler forward/backward "
                  f"(loss {float(l2):.6f}, scale {scaler.get_scale():.0f})")
        except Exception:
            traceback.print_exc()
            fail("mixed-precision path failed -- the train loop uses autocast "
                 "+ GradScaler")

    # --------------------------------------- nvrtc: diagnostic, NOT a hard gate
    # Patch sr-09 removed the only two jiterated ops, so nvrtc should no longer
    # be needed. Report its state anyway: if it is broken AND a future edit
    # reintroduces a complex-dtype op, the checks above fail and this line says why.
    try:
        if device.type != "cuda":
            print("[ops] nvrtc: not probed (CPU dry-run)")
            raise _SkipProbe()
        _z = torch.randn(8, device=device, dtype=torch.complex64)
        _ = torch.abs(_z)
        torch.cuda.synchronize()
        print("[ops] nvrtc: WORKING (jiterated complex ops available on this node)")
    except _SkipProbe:
        pass
    except Exception as exc:
        first = str(exc).strip().splitlines()[-1][:120] if str(exc).strip() else type(exc).__name__
        print(f"[ops] nvrtc: BROKEN -- {first}")
        print("[ops]   harmless: patch sr-09 removed the pipeline's only two "
              "complex-dtype ops. Do NOT reintroduce torch.abs/angle/sgn on a "
              "complex tensor without re-running this gate.")

    print("[ops] PASS -- every kernel the Stage-2 numerical core needs runs on this node")


if __name__ == "__main__":
    main()
PY


# --- stage the ~13 GB npy cache to node-local /tmp -------------------
LOCAL_ROOT=/tmp/mooring_${SLURM_JOB_ID}
LOCAL_CACHE=${LOCAL_ROOT}/cache_npy
trap 'echo "[stage] cleanup ${LOCAL_ROOT}"; rm -rf "${LOCAL_ROOT}"' EXIT

NEED_KB=$(( $(du -sk "${SRC_CACHE}" | cut -f1) + 2*1024*1024 ))
FREE_KB=$(df -k --output=avail /tmp | tail -1 | tr -d ' ')
echo "[stage] /tmp free ${FREE_KB} KB, need ${NEED_KB} KB"
[ "${FREE_KB}" -ge "${NEED_KB}" ] || die "/tmp too small for the cache (free ${FREE_KB} < need ${NEED_KB} KB)"

echo "[stage] copying ${SRC_CACHE} -> ${LOCAL_CACHE} ($(date))"
mkdir -p "${LOCAL_CACHE}"
rsync -a "${SRC_CACHE}/" "${LOCAL_CACHE}/" || die "rsync failed"
SRC_N=$(find "${SRC_CACHE}"   -type f | wc -l)
DST_N=$(find "${LOCAL_CACHE}" -type f | wc -l)
[ "${DST_N}" -gt 0 ]           || die "canary: 0 files staged"
[ "${SRC_N}" -eq "${DST_N}" ]  || die "canary: file count ${SRC_N} != ${DST_N}"

export MOORING_CACHE_DIR="${LOCAL_CACHE}"
echo "[stage] OK: ${DST_N} files staged ($(date))"
echo "[stage] MOORING_CACHE_DIR=${MOORING_CACHE_DIR}"

# --- run --------------------------------------------------------------
mkdir -p "${CKPT_DIR}"

echo "[run] executing notebook ($(date)) -- stdout goes into the executed"
echo "[run] .ipynb, NOT this .out; monitor ${CKPT_DIR}/metrics_history.csv"
srun "${VENV_PY}" -m nbconvert --to notebook --execute \
    "${NOTEBOOK}" \
    --output "GAT_LSTM_NN_Publication_SR_${TAG}_executed_${SLURM_JOB_ID}.ipynb" \
    --ExecutePreprocessor.timeout=-1
RC=$?

echo "[job] nbconvert exit code ${RC}; finished $(date)"
exit ${RC}
