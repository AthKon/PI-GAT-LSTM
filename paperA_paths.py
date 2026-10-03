"""Where paperA_numbers.py and paperA_figures.py read their inputs and write their outputs.

Two layouts are supported, and the scripts read exactly the same bytes in both.

Development layout (default, PAPERA_DATA unset)
    The result folders under ``Results/`` as the cluster jobs wrote them, the derived
    tables (snap catalogue, artifact masks, ...) next to the scripts, and the manuscript
    under ``paper_A_manuscript/``, where numbers.tex, tables/ and figures/ are written.

Archive layout (PAPERA_DATA set)
    PAPERA_DATA is the folder holding the three unpacked 4TU.ResearchData archives,
    ``stage1_matched/``, ``stage2_superres/`` and ``linear_baseline/``
    (https://doi.org/10.4121/e3f167ac-2925-4815-956e-5ef3d883f1a5). Every development
    path is translated to the archive file holding the same content (the RULES table
    below; each pair was checked to be byte-identical by SHA-256), the derived tables are
    read from ``data/``, and the outputs go to PAPERA_OUT (default ``./paperA_output``).

    A few results need inputs that are not in the archives: the raw simulations (site
    geometry, sea-state ranges, fairlead motion, the pinball-threshold exceedance, snap
    reachability, the catenary study), training-side files that were not deposited (the
    global-context bake-off logs, two Stage-2 per-checkpoint test dumps) and the file
    times behind the wall-clock figures. Functions that compute them are marked with
    ``@derived``. In the development layout they run normally; ``paperA_numbers.py
    --write-derived`` stores their results in ``data/derived_inputs.json`` (arrays in
    ``data/derived_inputs_arrays.npz``). In the archive layout they are read back from
    there, unless PAPERA_RECOMPUTE=1 (which needs the raw data, see MOORING_CACHE_DIR).

MOORING_CACHE_DIR
    The preprocessed simulation cache (``cache_npy/locNN/case_CCCC/*.npy``), in both
    layouts. It is not deposited; see the dataset README, section 12.
"""
import functools
import json
import os
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
_ARCH = os.environ.get("PAPERA_DATA")
ARCHIVES = pathlib.Path(_ARCH).expanduser().resolve() if _ARCH else None
RECOMPUTE = os.environ.get("PAPERA_RECOMPUTE", "") == "1"
CACHE = pathlib.Path(os.environ.get("MOORING_CACHE_DIR",
                                    r"C:\Users\thano\Desktop\data\cache_npy"))
DATA_DIR = ROOT / "data"
DERIVED_JSON = DATA_DIR / "derived_inputs.json"
DERIVED_NPZ = DATA_DIR / "derived_inputs_arrays.npz"

if ARCHIVES is None:
    OUT_DIR = ROOT / "paper_A_manuscript"
else:
    OUT_DIR = pathlib.Path(os.environ.get("PAPERA_OUT", ROOT / "paperA_output")).resolve()
    missing = [a for a in ("stage1_matched", "stage2_superres", "linear_baseline")
               if not (ARCHIVES / a).is_dir()]
    if missing:
        raise SystemExit(f"[paths] PAPERA_DATA={ARCHIVES} does not hold the unpacked "
                         f"archive folder(s) {missing}")


class NotInArchives(FileNotFoundError):
    """A development path that has no counterpart in the deposited archives."""


# --------------------------------------------------------------------- path translation
_EC = "Results/SR_RESULTS/checkpoints_P2_eval_clean/"
_S1 = "Results/checkpoints_P1_matched_v5/"
_S2 = "Results/SR_RESULTS/checkpoints_P2_final_warm(run2)/"
_SR = "Results/SR_RESULTS/checkpoints_P2_eval_snaprows/"
_LR = "Results/LR_baseline/"
_SCR, _UNS = "evaluation_screened_(published)/", "evaluation_unscreened/"
_S2_UNSCREENED = r"(test_metrics(_by_node_count)?\.csv|superres_grid[A-Za-z0-9_]*\.csv|test_timeseries_dump\.npz)"

# (regex on the path relative to the repository root, replacement relative to PAPERA_DATA).
# First match wins. Built from a SHA-256 comparison of every deposited file with the
# development tree (2026-10-03): each rule maps a file onto a byte-identical copy.
RULES = [
    # EVAL_CLEAN job 402698: screened draw at the top, the unscreened draw in published/
    (re.escape(_EC) + r"stage1/published/(.+)", "stage1_matched/" + _UNS + r"\1"),
    (re.escape(_EC) + r"stage1/(.+)", "stage1_matched/" + _SCR + r"\1"),
    (re.escape(_EC) + r"published/(.+)", "stage2_superres/" + _UNS + r"\1"),
    (re.escape(_EC) + r"(?!stage1/|published/)([^/]+)", "stage2_superres/" + _SCR + r"\1"),
    # EVAL_SNAPROWS job 287893: its per-window files equal the unscreened ones
    (re.escape(_SR) + r"stage1/test_per_window_stats\.csv",
     "stage1_matched/" + _UNS + "test_per_window_stats.csv"),
    (re.escape(_SR) + r"test_per_window_stats\.csv",
     "stage2_superres/" + _UNS + "test_per_window_stats.csv"),
    # withheld-site windows (EVAL_SNAPROWS with the per-site reservoirs)
    (r"Results/SR_RESULTS/checkpoints_P2_eval_withheld/test_timeseries_dump\.npz",
     "stage2_superres/" + _UNS + "test_timeseries_dump_withheld_sites.npz"),
    # Stage 1 (v5, epoch 32): checkpoint folder, and the corrected-catalogue evaluation
    (r"Results/eval_v5_snapfix/test_timeseries_dump\.npz",
     "stage1_matched/" + _UNS + "test_timeseries_dump.npz"),
    (re.escape(_S1) + r"test_timeseries_dump\.npz",
     "stage1_matched/" + _UNS + "test_timeseries_dump_training_job.npz"),
    (re.escape(_S1) + r"(metrics_history\.csv|metrics_by_node_count\.csv)",
     r"stage1_matched/training/\1"),
    (re.escape(_S1) + r"(config\.json|best_mooring_gat_lstm_seed_42\.pt|input_normalisers\.(npz|json))",
     r"stage1_matched/\1"),
    # Stage 2 (FINAL_WARM, epoch 76)
    (re.escape(_S2) + r"(metrics_history\.csv|metrics_by_node_count\.csv)",
     r"stage2_superres/training/\1"),
    (re.escape(_S2) + r"(config\.json|best_mooring_gat_lstm_seed_42\.pt|input_normalisers\.(npz|json))",
     r"stage2_superres/\1"),
    (re.escape(_S2) + _S2_UNSCREENED, "stage2_superres/" + _UNS + r"\1"),
    (re.escape(_S2) + r"(GAT_LSTM_NN_Publication_SR_FINAL_WARM_executed_154471\.ipynb)",
     r"stage2_superres/provenance/\1"),
    # linear regression baselines (job 885495)
    (re.escape(_LR) + r"(LR\d_\w+)/published/(.+)", "linear_baseline/" + r"\1/" + _UNS + r"\2"),
    (re.escape(_LR) + r"(LR\d_\w+)/coefficients\.csv", r"linear_baseline/\1/coefficients.csv"),
    (re.escape(_LR) + r"(LR\d_\w+)/(.+)", "linear_baseline/" + r"\1/" + _SCR + r"\2"),
    (re.escape(_LR) + r"(.+)", r"linear_baseline/\1"),
]
_RULES = [(re.compile(p + r"\Z"), r) for p, r in RULES]


def src(path):
    """The file to read for a development-layout ``path`` (unchanged in that layout)."""
    p = pathlib.Path(path)
    if ARCHIVES is None:
        return p
    try:
        rel = pathlib.Path(os.path.abspath(p)).relative_to(ROOT).as_posix()
    except ValueError:
        return p                                   # outside the repository (e.g. the cache)
    if "/" not in rel:                             # derived tables at the repository root
        return DATA_DIR / rel
    for pat, rep in _RULES:
        if pat.match(rel):
            return ARCHIVES / pat.sub(rep, rel)
    if rel.startswith("Results/"):
        raise NotInArchives(f"{rel} is not part of the deposited archives")
    return p


def exists(path):
    try:
        return src(path).exists()
    except NotInArchives:
        return False


def read_text(path, encoding="utf-8"):
    return src(path).read_text(encoding=encoding)


def npload(path, *args, **kwargs):
    return np.load(src(path), *args, **kwargs)


def open_(path, *args, **kwargs):
    return open(src(path), *args, **kwargs)


def rel_out(path):
    """``path`` for printing: relative to the repository root when it lies inside it."""
    try:
        return pathlib.Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------- derived inputs
# A tagged JSON encoding that round-trips the values exactly: tuples, dicts with
# non-string keys, NumPy scalars (whose repr differs from Python floats) and arrays
# (stored in the companion .npz).
def _enc(v, arrays):
    if isinstance(v, np.ndarray):
        key = f"a{len(arrays)}"
        arrays[key] = v
        return {"__nd__": key}
    if isinstance(v, np.generic):
        return {"__np__": v.dtype.str, "v": v.item()}
    if isinstance(v, tuple):
        return {"__tuple__": [_enc(x, arrays) for x in v]}
    if isinstance(v, list):
        return [_enc(x, arrays) for x in v]
    if isinstance(v, dict):
        return {"__dict__": [[_enc(k, arrays), _enc(x, arrays)] for k, x in v.items()]}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise TypeError(f"cannot store a {type(v).__name__} in {DERIVED_JSON.name}")


def _dec(v, arrays):
    if isinstance(v, list):
        return [_dec(x, arrays) for x in v]
    if isinstance(v, dict):
        if "__nd__" in v:
            return arrays[v["__nd__"]]
        if "__np__" in v:
            return np.array(v["v"], dtype=np.dtype(v["__np__"]))[()]
        if "__tuple__" in v:
            return tuple(_dec(x, arrays) for x in v["__tuple__"])
        if "__dict__" in v:
            return {_dec(k, arrays): _dec(x, arrays) for k, x in v["__dict__"]}
    return v


_RECORD, _ARRAYS = {}, {}


@functools.lru_cache(maxsize=None)
def _store():
    if not DERIVED_JSON.exists():
        raise SystemExit(f"[paths] {rel_out(DERIVED_JSON)} is missing; it ships with the "
                         "code repository")
    arrays = dict(np.load(DERIVED_NPZ)) if DERIVED_NPZ.exists() else {}
    return json.loads(DERIVED_JSON.read_text(encoding="utf-8"))["values"], arrays


def derived(fn):
    """Mark a function whose inputs are not in the archives (see the module docstring)."""
    name = fn.__name__

    @functools.wraps(fn)
    def wrap(*args):
        key = name + json.dumps(list(args))
        if ARCHIVES is not None and not RECOMPUTE:
            values, arrays = _store()
            if key not in values:
                raise SystemExit(f"[paths] {key} is missing from {rel_out(DERIVED_JSON)}")
            return _dec(values[key], arrays)
        value = fn(*args)
        _RECORD[key] = _enc(value, _ARRAYS)
        return value
    return wrap


def write_derived(note):
    """Store every @derived result computed in this run (development layout only)."""
    if ARCHIVES is not None:
        raise SystemExit("[paths] --write-derived needs the development layout (unset PAPERA_DATA)")
    DATA_DIR.mkdir(exist_ok=True)
    DERIVED_JSON.write_text(json.dumps({"note": note, "values": dict(sorted(_RECORD.items()))},
                                       indent=1) + "\n", encoding="utf-8")
    if _ARRAYS:
        np.savez_compressed(DERIVED_NPZ, **_ARRAYS)
    return sorted(_RECORD)
