"""
plot_node_timeseries.py -- tension time series AWAY from the fairlead, and in
particular at the TOUCHDOWN nodes that alternate between resting on the seabed
and hanging in the catenary.

Why this exists
---------------
plot_test_timeseries.py plots `true_w[:, -1]` -- the fairlead -- everywhere. But
the fairlead is the one node conventional single-point surrogates already
predict well; the claim that distinguishes this model is that it reconstructs
the tension at EVERY node, including the touchdown region where fatigue
concentrates and where nobody installs instrumentation. Those figures did not
exist, so this script makes them.

Seabed contact
--------------
The dump's per-window meta is [loc, case, node_count, start_index], so every
window can be tied back to the .npy cache and its z_abs read directly. z_abs is
measured from the seabed (the anchor sits at z = 0), so a node is in contact
when z < CONTACT_M. Nodes are then classified over the window as

    seabed        in contact essentially always
    TOUCHDOWN     in contact for part of the window  <- the interesting ones
    suspended     never in contact

For node counts below 21 the cached 21-node z is resampled with the same
normalised-index linear interpolation the pipeline uses for its targets
(notebook cell 4d3988c6), so the contact state matches the model's own
discretisation rather than the FE grid.

Figures written per window
--------------------------
    nodes_<win>.png      tension vs time at nodes spanning anchor -> fairlead,
                         true vs predicted, contact intervals shaded
    touchdown_<win>.png  the touchdown nodes alone, with the contact state and
                         liftoff/touchdown instants marked
    heatmap_<win>.png    [time x node] truth, prediction and signed error, with
                         the touchdown point's trajectory drawn on top

and once per run

    per_node_error.png   MAE, R^2 and peak underprediction rate against
                         normalised position along the line, pooled over every
                         window in the dump, split by contact regime

Cost: reads a few MB of .npz plus one z_abs per window (memory-mapped, one at a
time). It does NOT build datasets or load the model.

Usage
-----
    python plot_node_timeseries.py                       # default dir, auto windows
    python plot_node_timeseries.py <dir> --out figs/
    python plot_node_timeseries.py <dir> --window nc21_hi0
    python plot_node_timeseries.py <dir> -N 21           # only this topology
    python plot_node_timeseries.py <dir> --max-windows 6
"""
import argparse
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CACHE_DIR = pathlib.Path(r"C:\Users\thano\Desktop\data\cache_npy")
CONTACT_M = 0.01          # z below this = resting on the seabed
TD_LO, TD_HI = 0.02, 0.98  # contact fraction bounds for "intermittent"

C_TRUE, C_PRED, C_GRAY = "#1F2A37", "#4884D6", "#6B7280"
C_BED = "#E8B44A"
C_ERR = "#C0392B"

# Cases the spatial-coherence rescan flags as FE solver blowups
# (scan_corrupt_cases.py). Windows drawn from these are still plotted if asked
# for explicitly, but are excluded from the pooled per-node statistics.
BLOWUP_CASES = frozenset({
    (1, 11), (1, 116), (1, 131), (1, 142), (1, 185), (1, 243), (1, 253),
    (1, 298), (3, 129), (3, 164), (4, 2), (4, 29), (4, 72), (4, 79), (4, 99),
    (4, 114), (4, 186), (4, 239), (4, 296), (5, 19), (5, 22), (5, 133),
    (5, 224), (5, 232), (6, 65), (6, 120), (6, 126), (6, 222), (7, 2), (7, 31),
    (7, 62), (7, 80), (7, 85), (7, 92), (7, 117), (7, 169), (7, 203), (7, 244),
    (8, 77), (8, 83), (8, 133), (9, 34), (9, 65), (9, 150), (9, 156), (9, 176),
    (9, 218), (9, 238), (10, 81), (10, 100), (10, 173), (10, 205), (10, 286),
    (11, 11), (11, 43), (11, 240),
})


def resample(arr, target_N):
    """Normalised-index linear interpolation, matching resample_fe_output."""
    _, n0 = arr.shape
    if target_N == n0:
        return arr
    src = np.linspace(0.0, 1.0, n0)
    tgt = np.linspace(0.0, 1.0, target_N)
    idx = np.clip(np.searchsorted(src, tgt, side="right") - 1, 0, n0 - 2)
    w = np.clip((tgt - src[idx]) / (src[idx + 1] - src[idx]), 0.0, 1.0)
    return arr[:, idx] + w * (arr[:, idx + 1] - arr[:, idx])


def contact_state(loc, case, start, n_steps, target_N):
    """Boolean [steps, N] seabed-contact mask, or None if z is unavailable."""
    f = CACHE_DIR / f"loc{loc:02d}" / f"case_{case:04d}" / "z_abs.npy"
    if not f.exists():
        return None
    z = np.asarray(np.load(f, mmap_mode="r")[start:start + n_steps], dtype=np.float64)
    if z.shape[0] < n_steps:
        return None
    return resample(z, target_N) < CONTACT_M


def res_label(nc, n_out):
    """'N=21' for a matched dump, 'N_in=4 -> N_out=21' for a Stage-2 super-resolution one.

    A Stage-2 dump stores every window at the OUTPUT resolution (the array width)
    while meta[2] is the INPUT sensor count; in a Stage-1 dump the two are equal.
    """
    return f"N={nc}" if n_out == nc else f"N_in={nc} -> N_out={n_out}"


def sensed_stations(nc, n_out):
    """Output stations that coincide with an input sensor in normalised arc length."""
    return {j for j in range(n_out)
            if any(abs(j / (n_out - 1) - k / (nc - 1)) < 1e-9 for k in range(nc))}


def classify(inc):
    """Split node indices into seabed / touchdown / suspended by contact fraction."""
    frac = inc.mean(0)
    seabed = np.where(frac >= TD_HI)[0]
    touch = np.where((frac > TD_LO) & (frac < TD_HI))[0]
    susp = np.where(frac <= TD_LO)[0]
    return seabed, touch, susp, frac


def shade_contact(ax, inc_node, t):
    """Shade the time intervals during which this node rests on the seabed."""
    if inc_node is None or not inc_node.any():
        return
    d = np.diff(inc_node.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if inc_node[0]:
        starts = [0] + starts
    if inc_node[-1]:
        ends = ends + [len(inc_node)]
    for a, b in zip(starts, ends):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=C_BED, alpha=0.18, lw=0)


def panel(ax, t, tr, pr, inc_node, title):
    shade_contact(ax, inc_node, t)
    ax.plot(t, tr / 1000.0, color=C_TRUE, lw=1.2)
    ax.plot(t, pr / 1000.0, color=C_PRED, lw=1.1, ls="--")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)


def fig_nodes(d, key, out_dir, dt):
    loc, case, nc, st = (int(v) for v in d[key + "_meta"])
    tr_w, pr_w = d[key + "_true"], d[key + "_pred"]
    S, n_out = tr_w.shape
    t = np.arange(S) * dt
    inc = contact_state(loc, case, st, S, n_out)
    if inc is None:
        return None
    seabed, touch, susp, frac = classify(inc)
    sensed = sensed_stations(nc, n_out)

    # span the line: anchor, a seabed node, every touchdown node, a mid-span
    # suspended node, and the fairlead
    picks = [0]
    if len(seabed) > 1:
        picks.append(int(seabed[len(seabed) // 2]))
    picks += [int(i) for i in touch]
    if len(susp):
        mid = susp[(susp > (touch.max() if len(touch) else 0))]
        if len(mid):
            picks.append(int(mid[len(mid) // 2]))
    picks.append(n_out - 1)
    picks = sorted(dict.fromkeys(p for p in picks if 0 <= p < n_out))

    n = len(picks)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.3 * rows), squeeze=False)
    for k in range(rows * 2):
        ax = axes[k // 2][k % 2]
        if k >= n:
            ax.axis("off")
            continue
        i = picks[k]
        if i == 0:
            role = "anchor"
        elif i == n_out - 1:
            role = "FAIRLEAD"
        elif i in touch:
            role = f"TOUCHDOWN ({100*frac[i]:.0f}% on seabed)"
        elif i in seabed:
            role = "on seabed"
        else:
            role = "suspended"
        if n_out != nc:
            role += " [sensed]" if i in sensed else " [NOT sensed]"
        panel(ax, t, tr_w[:, i], pr_w[:, i], inc[:, i],
              f"node {i}/{n_out-1} -- {role} | true peak {tr_w[:, i].max()/1000:.2f} kN, "
              f"pred {pr_w[:, i].max()/1000:.2f} kN")
    handles = [Line2D([], [], color=C_TRUE, lw=1.4, label="FE truth"),
               Line2D([], [], color=C_PRED, lw=1.3, ls="--", label="prediction"),
               plt.Rectangle((0, 0), 1, 1, color=C_BED, alpha=0.3,
                             label="node resting on seabed")]
    fig.supxlabel("time within window [s]", fontsize=9, y=0.045)
    fig.supylabel("tension [kN]", fontsize=9)
    flag = "  [BLOWUP CASE]" if (loc, case) in BLOWUP_CASES else ""
    fig.suptitle(f"Tension along the line -- loc{loc:02d} case{case:04d} "
                 f"{res_label(nc, n_out)}{flag}", fontsize=11)
    fig.tight_layout(rect=(0, 0.058, 1, 1))
    fig.legend(handles=handles, fontsize=8, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.004))
    fn = f"nodes_{key}.png"
    fig.savefig(out_dir / fn, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return fn


def fig_touchdown(d, key, out_dir, dt):
    loc, case, nc, st = (int(v) for v in d[key + "_meta"])
    tr_w, pr_w = d[key + "_true"], d[key + "_pred"]
    S, n_out = tr_w.shape
    t = np.arange(S) * dt
    inc = contact_state(loc, case, st, S, n_out)
    if inc is None:
        return None
    _, touch, _, frac = classify(inc)
    if not len(touch):
        return None

    n = len(touch)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.7 * n), squeeze=False, sharex=True)
    for k, i in enumerate(touch):
        ax = axes[k][0]
        tr, pr, ic = tr_w[:, i], pr_w[:, i], inc[:, i]
        shade_contact(ax, ic, t)
        ax.plot(t, tr / 1000.0, color=C_TRUE, lw=1.3, label="FE truth")
        ax.plot(t, pr / 1000.0, color=C_PRED, lw=1.2, ls="--", label="prediction")
        # liftoff / touchdown instants
        dd = np.diff(ic.astype(np.int8))
        for j in np.where(dd == -1)[0]:
            ax.axvline(t[j], color=C_GRAY, lw=0.6, ls=":")
        for j in np.where(dd == 1)[0]:
            ax.axvline(t[j], color=C_GRAY, lw=0.6, ls=":")
        nev = int((dd != 0).sum())
        err = pr - tr
        ax.set_title(f"node {i}/{n_out-1} -- on seabed {100*frac[i]:.0f}% of the window, "
                     f"{nev} contact transitions | MAE {np.abs(err).mean():.0f} N, "
                     f"peak err {pr.max()-tr.max():+.0f} N", fontsize=9)
        ax.set_ylabel("tension [kN]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        if k == 0:
            h, l = ax.get_legend_handles_labels()
            h.append(plt.Rectangle((0, 0), 1, 1, color=C_BED, alpha=0.3))
            l.append("resting on seabed")
            ax.legend(h, l, fontsize=8, frameon=False, loc="upper right")
    axes[-1][0].set_xlabel("time within window [s]", fontsize=9)
    flag = "  [BLOWUP CASE]" if (loc, case) in BLOWUP_CASES else ""
    fig.suptitle(f"Touchdown-region tension -- loc{loc:02d} case{case:04d} "
                 f"{res_label(nc, n_out)}{flag}\nthe line lifts off and resettles within the window",
                 fontsize=11)
    fig.tight_layout()
    fn = f"touchdown_{key}.png"
    fig.savefig(out_dir / fn, dpi=140)
    plt.close(fig)
    return fn


def fig_heatmap(d, key, out_dir, dt):
    loc, case, nc, st = (int(v) for v in d[key + "_meta"])
    tr_w, pr_w = d[key + "_true"], d[key + "_pred"]
    S, n_out = tr_w.shape
    t = np.arange(S) * dt
    inc = contact_state(loc, case, st, S, n_out)
    err = pr_w - tr_w
    vmax = max(tr_w.max(), pr_w.max()) / 1000.0
    emax = float(np.abs(err).max()) / 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True)
    # anchor at the bottom, fairlead at the top -- matches the physical layout
    ext = [t[0], t[-1], -0.5, n_out - 0.5]
    for ax, arr, ttl, cm, kw in (
        (axes[0], tr_w / 1000.0, "FE truth", "viridis", dict(vmin=0, vmax=vmax)),
        (axes[1], pr_w / 1000.0, "prediction", "viridis", dict(vmin=0, vmax=vmax)),
        (axes[2], err / 1000.0, "signed error (pred - truth)", "RdBu_r",
         dict(vmin=-emax, vmax=emax)),
    ):
        im = ax.imshow(arr.T, aspect="auto", origin="lower", extent=ext,
                       cmap=cm, interpolation="nearest", **kw)
        ax.set_ylabel("node index\n(0 = anchor)", fontsize=9)
        ax.set_title(ttl, fontsize=9)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, pad=0.01).set_label("kN", fontsize=8)
        if inc is not None:
            # touchdown point = last node still in contact at each instant
            any_c = inc.any(1)
            td = np.where(any_c, inc.shape[1] - 1 - np.argmax(inc[:, ::-1], axis=1),
                          np.nan)
            ax.plot(t, td, color="w", lw=1.0, alpha=0.85)
            ax.plot(t, td, color="k", lw=0.5, alpha=0.7)
    axes[-1].set_xlabel("time within window [s]", fontsize=9)
    flag = "  [BLOWUP CASE]" if (loc, case) in BLOWUP_CASES else ""
    fig.suptitle(f"Full-line tension field -- loc{loc:02d} case{case:04d} "
                 f"{res_label(nc, n_out)}{flag}\n"
                 f"black/white line = touchdown point (last node on the seabed)",
                 fontsize=11)
    fig.tight_layout()
    fn = f"heatmap_{key}.png"
    fig.savefig(out_dir / fn, dpi=140)
    plt.close(fig)
    return fn


def fig_per_node(d, keys, out_dir):
    """Error against normalised position along the line, pooled over windows."""
    pos, mae, rel, r2, under, regime = [], [], [], [], [], []
    used = skipped = 0
    for key in keys:
        loc, case, nc, st = (int(v) for v in d[key + "_meta"])
        if (loc, case) in BLOWUP_CASES:
            skipped += 1
            continue
        used += 1
        tr_w, pr_w = d[key + "_true"], d[key + "_pred"]
        n_out = tr_w.shape[1]
        inc = contact_state(loc, case, st, tr_w.shape[0], n_out)
        frac = inc.mean(0) if inc is not None else np.zeros(n_out)
        for i in range(n_out):
            tr, pr = tr_w[:, i].astype(np.float64), pr_w[:, i].astype(np.float64)
            ss = ((tr - tr.mean()) ** 2).sum()
            m = np.abs(pr - tr).mean()
            pos.append(i / (n_out - 1))
            mae.append(m)
            rel.append(100.0 * m / max(tr.mean(), 1.0))
            r2.append(1.0 - ((pr - tr) ** 2).sum() / ss if ss > 0 else np.nan)
            under.append(1.0 if pr.max() < tr.max() else 0.0)
            regime.append(1 if frac[i] >= TD_HI else (2 if frac[i] > TD_LO else 0))
    pos = np.array(pos); mae = np.array(mae); rel = np.array(rel)
    r2 = np.array(r2); under = np.array(under); regime = np.array(regime)

    bins = np.linspace(0, 1, 11)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    which = np.clip(np.digitize(pos, bins) - 1, 0, len(ctr) - 1)

    def binned(v, fn):
        return [fn(v[which == k]) if (which == k).any() else np.nan
                for k in range(len(ctr))]

    cols = {0: C_PRED, 1: C_BED, 2: C_ERR}
    names = {0: "suspended", 1: "on seabed", 2: "touchdown"}
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    series = (
        (axes[0], mae, binned(mae, np.nanmean), "MAE [N]", True),
        (axes[1], rel, binned(rel, np.nanmean), "MAE / mean tension [%]", True),
        (axes[2], r2, binned(r2, np.nanmedian), "per-node $R^2$", True),
        (axes[3], under, binned(under, lambda a: 100 * np.nanmean(a)),
         "peak underprediction [%]", False),
    )
    for ax, val, bval, lbl, scat in series:
        if scat:
            for g in (0, 1, 2):
                m = regime == g
                if m.any():
                    ax.scatter(pos[m], val[m], s=11, color=cols[g], alpha=0.45,
                               edgecolor="none", label=names[g])
        ax.plot(ctr, bval, color=C_TRUE, lw=1.8, marker="o", ms=4)
        ax.set_ylabel(lbl, fontsize=9)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
    axes[2].axhline(0.0, color=C_ERR, lw=1.0, ls="--")
    axes[2].set_ylim(-1.0, 1.05)
    axes[0].legend(fontsize=8, frameon=False, ncol=3)
    axes[3].set_xlabel("normalised position along line   (0 = anchor, 1 = fairlead)",
                       fontsize=9)

    lines = []
    for g, nm in names.items():
        v = r2[regime == g]
        v = v[~np.isnan(v)]
        if v.size:
            lines.append(f"{nm}: median $R^2$ {np.median(v):.3f}, "
                         f"{100*(v<0).mean():.0f}% below 0  (n={v.size})")
    axes[2].text(0.015, 0.04, "\n".join(lines), transform=axes[2].transAxes,
                 fontsize=8, va="bottom", ha="left",
                 bbox=dict(fc="white", ec=C_GRAY, lw=0.6, alpha=0.9, pad=4))
    # the dump over-samples worst-underprediction windows by construction, so
    # the rate below is an upper bound, not a test-set average
    axes[3].text(0.015, 0.06, "selected windows over-represent worst-case peaks;\n"
                              "read as relative shape along the line, not an "
                              "absolute rate", transform=axes[3].transAxes,
                 fontsize=7.5, va="bottom", ha="left", color=C_GRAY)
    fig.suptitle("Reconstruction quality along the line\n"
                 f"pooled over {used} test windows, all topologies "
                 f"({skipped} blowup-case windows excluded)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "per_node_error.png", dpi=140)
    plt.close(fig)
    print(f"  wrote per_node_error.png  ({used} windows pooled, {skipped} excluded)")
    for g, nm in names.items():
        v = r2[regime == g]
        v = v[~np.isnan(v)]
        if v.size:
            print(f"      {nm:<10} median R2 {np.median(v):7.3f}  "
                  f"{100*(v<0).mean():5.1f}% below 0   n={v.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="Results/checkpoints_P1_matched_v2")
    ap.add_argument("--window", default=None, help="a specific dump key, e.g. nc21_hi0")
    ap.add_argument("--node-count", "-N", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = pathlib.Path(args.path)
    npz = p if p.suffix == ".npz" else p / "test_timeseries_dump.npz"
    if not npz.exists():
        raise SystemExit(f"[FATAL] {npz} not found")
    d = np.load(npz)
    dt = float(d["dt"]) if "dt" in d else 0.1
    out_dir = pathlib.Path(args.out) if args.out else npz.parent / "figs_nodes"
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = [k[:-5] for k in d.keys() if k.endswith("_meta")]
    if args.node_count is not None:
        keys = [k for k in keys if int(d[k + "_meta"][2]) == args.node_count]

    if args.window:
        chosen = [args.window]
    else:
        # prefer clean cases with the most touchdown nodes
        scored = []
        for k in keys:
            loc, case, nc, st = (int(v) for v in d[k + "_meta"])
            inc = contact_state(loc, case, st, *d[k + "_true"].shape)
            if inc is None:
                continue
            _, touch, _, _ = classify(inc)
            scored.append((len(touch), (loc, case) not in BLOWUP_CASES, nc,
                           (loc, case), k))
        scored.sort(key=lambda s: (-s[1], -s[0], -s[2]))
        # one window per (loc, case) first, so the figures span distinct
        # simulations instead of eight views of whichever case happens to
        # carry the most touchdown nodes
        chosen, seen = [], set()
        for _, _, _, cs, k in scored:
            if cs in seen:
                continue
            seen.add(cs)
            chosen.append(k)
            if len(chosen) >= args.max_windows:
                break
        if len(chosen) < args.max_windows:
            chosen += [s[4] for s in scored if s[4] not in chosen][
                :args.max_windows - len(chosen)]

    print(f"{npz.name}: {len(keys)} candidate windows -> plotting {len(chosen)}")
    for k in chosen:
        loc, case, nc, st = (int(v) for v in d[k + "_meta"])
        tag = f"loc{loc:02d} case{case:04d} {res_label(nc, d[k + '_true'].shape[1])}"
        made = [f for f in (fig_nodes(d, k, out_dir, dt),
                            fig_touchdown(d, k, out_dir, dt),
                            fig_heatmap(d, k, out_dir, dt)) if f]
        print(f"  {k:<12} {tag:<26} -> {len(made)} figure(s)")

    fig_per_node(d, keys, out_dir)
    print(f"\ndone -> {out_dir}")


if __name__ == "__main__":
    main()
