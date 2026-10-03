"""Export the five input normalisers of a shipped checkpoint (they are not stored in it).

    python export_input_normalisers.py stage1 --out <dir> [--zip stage1_matched.zip]
    python export_input_normalisers.py stage2 --out <dir> [--zip stage2_superres.zip]

--zip is the deposited archive of that stage (default: its place in the development tree); it
supplies the checkpoint and the unscreened test draw the rebuilt split is checked against.

The notebook refits the standardisers from the training windows every time it builds its data
loaders (build_dataloaders -> fit_normalizers_from_train_subset), and the checkpoint keeps only the
tension and position statistics. This script reproduces that fit exactly:

  * the training split is rebuilt with the verbatim split utilities (linear_regression_lib, which
    reproduces Stage 1's draw exactly), with pair_windows_across_node_counts = True for Stage 2;
  * the rebuilt TEST draw must equal the stage's deposited unscreened per-window file, key for key;
  * the fit itself is the notebook's own fit_normalizers_from_train_subset, executed from the
    notebook's definition cells, on lazily built datasets (constructed as in cell fbac801f);
  * the fitted tension mean/std and x/z position mean/std must equal the values the checkpoint
    stores (criterion_state_dict t_mean, t_std, pos_mean, pos_std).

Writes input_normalisers.npz and input_normalisers.json. Needs the preprocessed cache
(MOORING_CACHE_DIR) -- it reads 5,000 training windows, nothing is trained.
"""
import argparse, csv, io, json, os, sys, time, zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import linear_regression_lib as L                              # noqa: E402

NOTEBOOK = ROOT / "GAT_LSTM_NN_Publication_superresolution.ipynb"
DEF_CELLS = ["6a5be23c", "66cf027f", "4d3988c6", "a5660cf4", "bcd95251", "28a9e4ff", "srdec01",
             "f2de07a6", "c53b5783", "858f4194", "b18c6667", "8e7ef9e9", "81d71e4b", "9e030261",
             "pubphys01", "7a1e407c", "8f268a8b", "c62bc05c", "d75ee9f2", "srbase01", "pubts01",
             "71b74ab8"]
STAGES = {
    "stage1": dict(zip=ROOT / "Results" / "stage1_matched.zip", root="stage1_matched", paired=False),
    "stage2": dict(zip=ROOT / "Results" / "SR_RESULTS" / "stage2_superres.zip", root="stage2_superres",
                   paired=True),
}


def notebook_namespace():
    ns = {"__name__": "mooring_definitions"}
    for cell in json.load(open(NOTEBOOK, encoding="utf-8"))["cells"]:
        if cell.get("id") in DEF_CELLS:
            exec(compile("".join(cell["source"]), f"cell_{cell['id']}", "exec"), ns)
    return ns


class LazyDatasets:
    """(loc, case, N) -> MooringSequenceDatasetPositionTension, built on access exactly as
    build_lazy_load_case_datasets (cell fbac801f) builds it, and not kept (bounded memory)."""

    def __init__(self, ns, cache, cfg):
        self.ns, self.cache, self.cfg = ns, Path(cache), cfg
        self.built = 0

    def __getitem__(self, key):
        lc, case, n = key
        ns, d = self.ns, self.cache / f"loc{lc:02d}" / f"case_{case:04d}"
        row0 = lambda f: torch.tensor(np.load(d / f, mmap_mode="r")[0], dtype=torch.float32)  # noqa: E731
        x0, z0, t0, s0 = (row0(f) for f in ("x_abs.npy", "z_abs.npy", "tension.npy", "reference.npy"))
        env = np.load(d / "env.npy")
        rs = lambda v: ns["resample_fe_output"](v.unsqueeze(0), n).squeeze(0)               # noqa: E731
        graph = ns["build_mooring_graph"](
            x0_nodes=rs(x0), z0_nodes=rs(z0), tension0_nodes=rs(t0), s0_nodes=rs(s0),
            water_depth=float(env[0]), depths_current=[float(env[i]) for i in (3, 5, 7, 9, 11)])
        graph.batch_location_id, graph.case_id = lc, case
        self.built += 1
        return ns["MooringSequenceDatasetPositionTension"](
            graph_data=graph, case_dir=d, target_N=n, window_len=self.cfg.window_len,
            contact_tol=1e-6, max_time_steps=None,
            use_env_features=self.cfg.use_env_features,
            use_velocity_features=self.cfg.use_velocity_features,
            use_drag_features=self.cfg.use_drag_features)


def deposited_test_keys(spec):
    z = zipfile.ZipFile(spec["zip"])
    name = f"{spec['root']}/evaluation_unscreened/test_per_window_stats.csv"
    rows = csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8"))
    return sorted((int(r["lc_id"]), int(r["case_id"]), int(r["n_in"]), int(r["start_idx"])) for r in rows)


def checkpoint(spec):
    z = zipfile.ZipFile(spec["zip"])
    return torch.load(io.BytesIO(z.read(f"{spec['root']}/best_mooring_gat_lstm_seed_42.pt")),
                      map_location="cpu", weights_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", default=None, help="the deposited archive of this stage")
    ap.add_argument("--cache", default=os.environ.get("MOORING_CACHE_DIR", "C:/Users/thano/Desktop/data/cache_npy"))
    ap.add_argument("--n-fit", type=int, default=None, help="debug only: fewer fit windows (no validation)")
    a = ap.parse_args()
    spec, t0 = dict(STAGES[a.stage]), time.time()
    if a.zip:
        spec["zip"] = Path(a.zip)

    ck = checkpoint(spec)
    cfg = SimpleNamespace(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["cfg"].items()})
    if a.n_fit:
        cfg.n_fit_windows = a.n_fit
    print(f"[{a.stage}] checkpoint epoch {ck['epoch']}; n_fit_windows {cfg.n_fit_windows}; split seed {cfg.split_seed}")

    # ---- 1. the exact training split (and a proof that it is the deposited draw)
    T = L.load_tables(ROOT if (ROOT / "artifact_samples.csv").exists() else ROOT / "data")
    L.set_tables(T["artifact"])
    scfg = L.load_split_config_from_dict(ck["cfg"]) if hasattr(L, "load_split_config_from_dict") else None
    if scfg is None:
        scfg = SimpleNamespace(**{k: ck["cfg"][k] for k in L.SPLIT_FIELDS})
        scfg.tension_mape_min = float(ck["cfg"].get("tension_mape_min", 1000.0))
        scfg.n_fit_windows = int(ck["cfg"]["n_fit_windows"])
    scfg.pair_windows_across_node_counts = spec["paired"]
    sp = L.regenerate_split(a.cache, scfg, T["catenary"])
    test = sorted(sp["test"])
    dep = deposited_test_keys(spec)
    assert test == dep, f"rebuilt test draw differs from the deposited one ({len(test)} vs {len(dep)})"
    print(f"[{a.stage}] split rebuilt: train {len(sp['train']):,}, val {len(sp['val']):,}, "
          f"test {len(test):,} == deposited unscreened draw (key for key)  [{time.time()-t0:.0f} s]")

    # ---- 2. the notebook's own fit
    ns = notebook_namespace()
    pairs = [((lc, case, n), w) for lc, case, n, w in sp["train"]]
    lazy = LazyDatasets(ns, a.cache, cfg)
    subset = ns["MultiLoadCaseWindowSubset"](lazy, pairs)
    node, edge, target, snode, sedge = ns["fit_normalizers_from_train_subset"](subset, cfg)
    print(f"[{a.stage}] fit done: {lazy.built:,} datasets built on demand  [{time.time()-t0:.0f} s]")

    # ---- 3. validation against what the checkpoint stores
    cs = ck["criterion_state_dict"]
    got = {"t_mean": float(target.mean[0]), "t_std": float(target.std[0]),
           "pos_mean": [float(v) for v in node.mean[:2]], "pos_std": [float(v) for v in node.std[:2]]}
    want = {"t_mean": float(cs["t_mean"]), "t_std": float(cs["t_std"]),
            "pos_mean": [float(v) for v in cs["pos_mean"]], "pos_std": [float(v) for v in cs["pos_std"]]}
    assert list(node.feature_idx[:2]) == [0, 1], node.feature_idx
    print(f"[{a.stage}] fitted vs checkpoint:")
    exact = True
    for k in got:
        g, w = np.atleast_1d(got[k]), np.atleast_1d(want[k])
        rel = float(np.max(np.abs(g - w) / np.abs(w)))
        exact &= bool(np.all(g.astype(np.float32) == w.astype(np.float32)))
        print(f"   {k:9s} fitted {np.round(g, 6)}  stored {np.round(w, 6)}  max rel diff {rel:.2e}")
    if a.n_fit is None:
        assert exact, "fitted statistics do not reproduce the checkpoint's stored values exactly"
        print(f"[{a.stage}] EXACT: every stored statistic is reproduced bit for bit (float32)")

    # ---- 4. write
    ds = lazy[pairs[0][0]]
    names = {"node": list(getattr(ds, "dynamic_feature_names", [])),
             "edge": list(getattr(ds, "dynamic_edge_feature_names", [])),
             "target": list(getattr(ds, "target_feature_names", ["tension"]))}
    groups = {"dynamic_node": node, "dynamic_edge": edge, "target": target,
              "static_node": snode, "static_edge": sedge}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arrays, doc = {}, {}
    for g, n in groups.items():
        arrays[f"{g}_feature_idx"] = np.asarray(n.feature_idx, np.int64)
        arrays[f"{g}_mean"] = n.mean.numpy().astype(np.float32)
        arrays[f"{g}_std"] = n.std.numpy().astype(np.float32)
        doc[g] = {"feature_idx": list(map(int, n.feature_idx)),
                  "mean": [float(v) for v in n.mean], "std": [float(v) for v in n.std]}
    doc["_about"] = {
        "stage": a.stage, "checkpoint_epoch": int(ck["epoch"]),
        "fit": f"fit_normalizers_from_train_subset, {cfg.n_fit_windows} random training windows "
               f"(random.Random({cfg.split_seed})); static groups over the unique training graphs",
        "apply": "x[..., feature_idx] = (x[..., feature_idx] - mean) / std; channels not in "
                 "feature_idx (binary flags) are left unchanged. Output tension: y * std + mean "
                 "of the 'target' group.",
        "dynamic_node_channel_names": names["node"], "dynamic_edge_channel_names": names["edge"],
        "validated_against_checkpoint": want if a.n_fit is None else "not validated (debug run)",
    }
    np.savez(out / "input_normalisers.npz", **arrays)
    (out / "input_normalisers.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"[{a.stage}] wrote {out / 'input_normalisers.npz'} and .json  [{time.time()-t0:.0f} s]")


if __name__ == "__main__":
    main()
