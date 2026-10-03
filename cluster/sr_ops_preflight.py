"""HARD GATE 4-ops -- run the REAL numerical core on the REAL GPU, before staging.

Why this gate exists
--------------------
Four Stage-2 baseline attempts, four different causes, and the last three all
died in the SAME place: inside the notebook, after every gate had passed and
after the 27-35 minute stage.

    100052   67 s   module load 2024r1 gone          (caught by a gate, cheap)
    100111   28 min nofile reset to 1024             (after the stage)
    103567   29 min cuDNN rejects the v5 LSTM        (after the stage)
    103651   47 min nvrtc cannot compile abs(complex)(after the stage)

The gates so far test what we thought to test. The notebook keeps finding a NEW
torch operation that the rebuilt stack cannot execute, and each discovery costs
30-47 minutes of allocation.

This gate closes the CLASS instead of the instance. It execs the notebook's
definition cells -- which are pure `def`/`class` with no data access, verified
below -- builds the real MooringGATLSTM and the real ReconstructionKendallLoss
at the production architecture, and runs a forward AND a backward on synthetic
tensors on the real GPU. Every torch/PyG op the numerical core uses is therefore
exercised BY CONSTRUCTION, not from a hand-written list I might have gaps in.

It would have caught 103567 (the LSTM) and 103651 (abs on complex) in seconds.

What it deliberately does NOT do
--------------------------------
It touches no data. Cells f80f8d84 / fbac801f / a04f1f5f / 07257ee8 / 19c37b28
are NEVER executed -- they are the blacklist, the dataset builder, the run cell
and the smoke cells, i.e. everything with import-time side effects or a cache
dependency. A hard assertion enforces that. So this gate cannot itself become
the expensive step it is meant to protect.

It is a preflight, not a correctness test. Numerical CORRECTNESS is the smoke
test's job (_run_superres_smoke.py, 11 checks, on real data). This asks one
question only: does every kernel this pipeline needs actually RUN here?

Usage:  python sr_ops_preflight.py <notebook.ipynb>
        exit 0 = every op runs;  exit 1 = die before staging.
"""

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
