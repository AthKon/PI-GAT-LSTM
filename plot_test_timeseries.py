"""
plot_test_timeseries.py -- re-plot the predicted-vs-true tension time series
LOCALLY from the small .npz produced on DelftBlue by dump_test_timeseries()
(notebook cell pubts01, patch 11).

This is deliberately cheap: it loads a few MB of arrays and draws. It does NOT
build datasets, load the model, or touch the .npy cache -- nothing here can
bog down a laptop. Runtime ~2 s per dump.

The cluster writes one dump per FEASIBLE checkpoint (patch 13), so any candidate
epoch can be inspected here, not just the selected one:

    test_timeseries_dump.npz              <- the SELECTED model, full test set
    test_timeseries_dump_epoch0031.npz    <- feasible epoch 31, bounded subset
    test_timeseries_dump_epoch0043.npz    <- feasible epoch 43, same subset
    ...

Per-epoch dumps all use the IDENTICAL test windows (the sampler epoch is pinned
on the cluster), so they are directly comparable epoch-to-epoch.

Every dump also carries per-TOPOLOGY windows (patch 15), so the time series can
be inspected node count by node count.

Usage:
    python plot_test_timeseries.py                      # default dir, selected model
    python plot_test_timeseries.py <dir> --list         # what dumps exist
    python plot_test_timeseries.py <dir> --epoch 43     # one candidate epoch
    python plot_test_timeseries.py <dir> --all          # every dump found
    python plot_test_timeseries.py <path.npz>           # an explicit file
    python plot_test_timeseries.py <dir> --epoch 43 --out figs/

  per-topology:
    python plot_test_timeseries.py <dir> -N 21          # one topology
    python plot_test_timeseries.py <dir> -N 4,12,21     # several
    python plot_test_timeseries.py <dir> -N all         # every topology
    python plot_test_timeseries.py <dir> --topology-overview
    python plot_test_timeseries.py <dir> --epoch 43 -N all --topology-overview

NOTE the per-topology figures here are FAIRLEAD-peak only, for visualisation.
The authoritative per-topology safety numbers are the tail-conditioned p90/p99
columns (thr / underpred_rate / bias / MAE / undermag, over ALL nodes) in
test_metrics_by_node_count.csv and metrics_by_node_count.csv -- patch 14.

Default input dir: Results/checkpoints_P1_matched_v2
Default output:    alongside each .npz (per-epoch figures keep the epoch suffix)
"""
import argparse
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_TRUE, C_PRED, C_GRAY = "#1F2A37", "#4884D6", "#6B7280"
C_SNAP = "#E8B44A"          # catalogued snap-load event marker (patch 27)


def grid(d, tag, title, fname, out_dir, dt, quiet=False):
    sel = []
    i = 0
    while f"{tag}{i}_true" in d:
        # patch 27: _evrows = snap-event rows inside this window (may be absent
        # in pre-patch-27 dumps, and empty for a window with no snap)
        ev = d[f"{tag}{i}_evrows"] if f"{tag}{i}_evrows" in d else np.zeros(0, int)
        sel.append((d[f"{tag}{i}_true"], d[f"{tag}{i}_pred"],
                    d[f"{tag}{i}_meta"], ev))
        i += 1
    if not sel:
        if not quiet:
            print(f"  (no '{tag}' windows in dump -- skipped)")
        return
    n = min(len(sel), 8)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.6 * rows), squeeze=False)
    for k in range(rows * 2):
        ax = axes[k // 2][k % 2]
        if k >= n:
            ax.axis("off")
            continue
        true_w, pred_w, meta, evrows = sel[k]
        tr, pr = true_w[:, -1], pred_w[:, -1]      # fairlead = last node
        t = np.arange(len(tr)) * dt
        ax.plot(t, tr / 1000.0, color=C_TRUE, lw=1.2, label="FE truth")
        ax.plot(t, pr / 1000.0, color=C_PRED, lw=1.1, ls="--", label="prediction")
        ax.axhline(tr.max() / 1000.0, color=C_GRAY, lw=0.7, ls=":")
        # patch 27: mark catalogued snap-load events. The fairlead trace is
        # plotted but a snap often peaks at an INTERIOR node, so also report
        # the line max at the event -- that is what snap_true/snap_pred score.
        for r in np.atleast_1d(evrows).astype(int):
            if 0 <= r < len(tr):
                ax.axvline(r * dt, color=C_SNAP, lw=1.0, ls="-", alpha=0.75,
                           zorder=0)
        lc, case, nc = int(meta[0]), int(meta[1]), int(meta[2])
        _sn = ""
        if np.atleast_1d(evrows).size:
            _r = int(np.atleast_1d(evrows)[0])
            _sn = (f" | SNAP x{np.atleast_1d(evrows).size} @ {_r*dt:.1f} s, "
                   f"line max {true_w[_r].max()/1000:.1f} kN "
                   f"(pred {pred_w[_r].max()/1000:.1f})")
        ax.set_title(f"loc{lc:02d} case{case:04d} N={nc} | true peak "
                     f"{tr.max()/1000:.1f} kN, pred {pr.max()/1000:.1f} kN{_sn}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        if k == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.supxlabel("time within window [s]", fontsize=9)
    fig.supylabel("fairlead tension [kN]", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=140)
    plt.close(fig)
    print(f"  wrote {fname}")


def epoch_of(p):
    """Epoch number encoded in a dump filename, or None for the selected-model dump."""
    m = re.search(r"_epoch(\d+)\.npz$", p.name)
    return int(m.group(1)) if m else None


def find_dumps(d):
    return sorted(d.glob("test_timeseries_dump*.npz"),
                  key=lambda p: (epoch_of(p) is None, epoch_of(p) or -1))


def render_topology(d, nc, out_dir, dt, sfx, who):
    """Per-topology grids + that topology's fairlead-peak parity."""
    meta = d["meta_all"]
    m = meta[:, 2] == nc
    pt, pp = d["peak_true_all"][m], d["peak_pred_all"][m]
    if pt.size == 0:
        print(f"  (no windows for N={nc} -- skipped)")
        return
    under = 100.0 * float((pp < pt).mean())
    miss = pt - pp
    neg = miss[miss > 0]
    print(f"  N={nc:>2}: {pt.size} windows | fairlead-peak underpred {under:5.1f}% "
          f"| median true peak {np.median(pt)/1000:6.2f} kN"
          + (f" | miss when low: mean {neg.mean()/1000:.2f} kN, "
             f"worst {neg.max()/1000:.2f} kN" if neg.size else ""))

    grid(d, f"nc{nc}_hi", f"N={nc} -- highest-load test windows ({who})",
         f"ts_N{nc:02d}_highest_peaks{sfx}.png", out_dir, dt, quiet=True)
    grid(d, f"nc{nc}_un", f"N={nc} -- worst fairlead-peak UNDERPREDICTIONS ({who})",
         f"ts_N{nc:02d}_worst_underpred{sfx}.png", out_dir, dt, quiet=True)
    grid(d, f"nc{nc}_ty", f"N={nc} -- typical test windows ({who})",
         f"ts_N{nc:02d}_typical{sfx}.png", out_dir, dt, quiet=True)
    # patch 27: representative SNAP-LOAD windows for this topology. hi/un are
    # pure tail selectors, so without these the only snap traces available are
    # the most extreme ones -- no typical snap example exists.
    grid(d, f"nc{nc}_sn", f"N={nc} -- SNAP-LOAD windows, representative ({who})",
         f"ts_N{nc:02d}_snap{sfx}.png", out_dir, dt, quiet=True)


def topology_overview(d, out_dir, sfx, who):
    """Fairlead-peak underprediction rate + miss magnitude vs node count.

    NOTE: this is FAIRLEAD-peak only, for visualisation. The authoritative
    per-topology safety numbers are the tail-conditioned p90/p99 columns in
    test_metrics_by_node_count.csv (all nodes, patch 14).
    """
    meta = d["meta_all"]
    ncs = sorted(set(meta[:, 2].tolist()))
    rates, means, worsts = [], [], []
    for nc in ncs:
        m = meta[:, 2] == nc
        pt, pp = d["peak_true_all"][m], d["peak_pred_all"][m]
        rates.append(100.0 * float((pp < pt).mean()))
        miss = pt - pp
        neg = miss[miss > 0]
        means.append(float(neg.mean()) / 1000 if neg.size else 0.0)
        worsts.append(float(neg.max()) / 1000 if neg.size else 0.0)

    x = np.arange(len(ncs))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
    a1.bar(x, rates, color=C_PRED, width=0.62)
    a1.set_ylabel("fairlead-peak\nunderprediction [%]")
    a1.grid(alpha=0.2, axis="y")
    a1.set_title(f"Peak behaviour by topology ({who}) -- fairlead peak\n"
                 f"authoritative tail safety: test_metrics_by_node_count.csv",
                 fontsize=10)
    a2.bar(x - 0.18, means, width=0.36, color=C_PRED, label="mean miss (when low)")
    a2.bar(x + 0.18, worsts, width=0.36, color=C_TRUE, label="worst miss")
    a2.set_ylabel("underprediction\nmagnitude [kN]")
    a2.set_xlabel("node count $N$")
    a2.set_xticks(x)
    a2.set_xticklabels([str(n) for n in ncs])
    a2.legend(fontsize=8, frameon=False)
    a2.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / f"peak_safety_by_topology{sfx}.png", dpi=140)
    plt.close(fig)
    print(f"  wrote peak_safety_by_topology{sfx}.png")


def render(npz_path, out_dir, topologies=None, overview=False):
    d = np.load(npz_path)
    dt = float(d["dt"]) if "dt" in d else 0.1
    ep = epoch_of(npz_path)
    sfx = f"_epoch{ep:04d}" if ep is not None else ""
    who = f"epoch {ep}" if ep is not None else "selected model"

    if topologies is not None or overview:
        avail = sorted(d["node_counts"].tolist()) if "node_counts" in d \
            else sorted(set(d["meta_all"][:, 2].tolist()))
        print(f"\n{npz_path.name}  [{who}]  topologies available: {avail}")
        if overview:
            topology_overview(d, out_dir, sfx, who)
        for nc in (avail if topologies == "all" else topologies or []):
            if nc not in avail:
                print(f"  (N={nc} not in this dump; available {avail})")
                continue
            render_topology(d, nc, out_dir, dt, sfx, who)
        return

    pt, pp = d["peak_true_all"], d["peak_pred_all"]
    under = 100.0 * float((pp < pt).mean())
    miss = pt - pp
    neg = miss[miss > 0]
    print(f"\n{npz_path.name}  [{who}]")
    print(f"  {len(pt)} windows | fairlead-peak underprediction {under:.1f}% | "
          f"median true peak {np.median(pt)/1000:.2f} kN")
    if neg.size:
        print(f"  when it misses low: mean {neg.mean()/1000:.2f} kN, "
              f"worst {neg.max()/1000:.2f} kN")

    grid(d, "hi", f"Highest-load test windows -- fairlead tension ({who})",
         f"ts_highest_peaks{sfx}.png", out_dir, dt)
    grid(d, "un", f"Worst fairlead-peak UNDERPREDICTIONS ({who})",
         f"ts_worst_underpred{sfx}.png", out_dir, dt)
    grid(d, "ty", f"Typical (median-peak) test windows ({who})",
         f"ts_typical{sfx}.png", out_dir, dt)
    # patch 27: an UNBIASED draw from the snap population (hi/un are its tail)
    grid(d, "sn", f"SNAP-LOAD windows -- representative sample ({who})",
         f"ts_snap{sfx}.png", out_dir, dt)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    lim = [0, max(pt.max(), pp.max()) / 1000 * 1.05]
    ax.plot(lim, lim, color=C_GRAY, ls="--", lw=1, label="perfect (y = x)")
    ax.scatter(pt / 1000, pp / 1000, s=8, color=C_PRED, alpha=0.35,
               edgecolor="none", label=f"test windows (n={len(pt)})")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("true fairlead peak tension [kN]")
    ax.set_ylabel("predicted fairlead peak tension [kN]")
    ax.set_title(f"Fairlead peak parity -- held-out test ({who})\n"
                 f"{under:.1f}% of windows underpredicted", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"peak_parity_fairlead{sfx}.png", dpi=140)
    plt.close(fig)
    print(f"  wrote peak_parity_fairlead{sfx}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="Results/checkpoints_P1_matched_v2",
                    help="a dump .npz, or a directory containing dumps")
    ap.add_argument("--epoch", type=int, default=None,
                    help="plot the dump for this feasible epoch")
    ap.add_argument("--all", action="store_true", help="plot every dump found")
    ap.add_argument("--list", action="store_true", help="list available dumps and exit")
    ap.add_argument("--node-count", "-N", default=None,
                    help="per-topology time series: a node count, a comma list "
                         "(4,12,21), or 'all'")
    ap.add_argument("--topology-overview", action="store_true",
                    help="bar chart of fairlead-peak safety vs node count")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    topologies = None
    if args.node_count is not None:
        topologies = "all" if args.node_count.strip().lower() == "all" else \
            [int(v) for v in args.node_count.replace(" ", "").split(",") if v]

    p = pathlib.Path(args.path)
    if p.is_dir():
        dumps = find_dumps(p)
        if not dumps:
            raise SystemExit(
                f"[FATAL] no test_timeseries_dump*.npz in {p}\n"
                "Copy them down from the cluster run directory first, e.g.:\n"
                '  scp "tkonstantaras@login.delftblue.tudelft.nl:/scratch/tkonstantaras/'
                'mooring_runs/<rundir>/checkpoints_P1_matched_v2/test_timeseries_dump*.npz" \\\n'
                f'      "{p}/"'
            )
        if args.list:
            print(f"{len(dumps)} dump(s) in {p}:")
            for q in dumps:
                ep = epoch_of(q)
                print(f"  {q.name:44s} {'selected model' if ep is None else f'epoch {ep}':16s}"
                      f" {q.stat().st_size/1e6:5.1f} MB")
            return
        if args.epoch is not None:
            sel = [q for q in dumps if epoch_of(q) == args.epoch]
            if not sel:
                have = sorted(e for e in (epoch_of(q) for q in dumps) if e is not None)
                raise SystemExit(f"[FATAL] no dump for epoch {args.epoch}. Available: {have}")
        elif args.all:
            sel = dumps
        else:
            sel = [q for q in dumps if epoch_of(q) is None] or dumps[:1]
    else:
        if not p.exists():
            raise SystemExit(f"[FATAL] {p} not found.")
        sel = [p]

    out_dir = pathlib.Path(args.out) if args.out else sel[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for q in sel:
        render(q, out_dir, topologies=topologies, overview=args.topology_overview)
    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
