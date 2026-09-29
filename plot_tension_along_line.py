"""
plot_tension_along_line.py -- one figure per node count showing the tension
time series at a GROUNDED, a TRANSITION (touchdown) and a FULLY SUBMERGED node,
for several sea states.

Why this exists
---------------
plot_test_timeseries.py plots the fairlead only; plot_node_timeseries.py plots
one window at a time. Neither answers the question a reader of Finding 4 asks
first: for a given discretisation, how well is each CONTACT REGIME reconstructed,
and does that hold across sea states? This script lays that out directly --
rows are sea states, columns are the three regimes.

Regime classification
---------------------
The dump's per-window meta is [loc, case, node_count, start_index], so z_abs can
be read straight from the .npy cache and resampled to the window's own node count
with the same normalised-index interpolation the pipeline uses (notebook cell
4d3988c6). z_abs is measured from the seabed, so a node is in contact when
z < CONTACT_M. Over the window each node then gets a contact fraction:

    grounded         frac >= 0.98   resting on the seabed essentially always
    transition       0.02 < frac < 0.98   alternates seabed <-> catenary
    fully submerged  frac <= 0.02   hanging in the water column throughout

Coarse discretisations often have NO intermittent node -- the touchdown point
falls between two nodes. There the transition column falls back to the
touchdown-adjacent node (the highest-index grounded node) and says so.

Sea-state labels come from env.npy, layout [h0, Hs, Tp, d1, v1, ...].

Window selection (added 2026-08-02)
-----------------------------------
The dump stores both per-topology (``nc{N}_*``) and pooled (``hi``/``un``/``ty``/
``sn``) windows; both are read, deduplicated on (loc, case, start).

Two filters keep the figures representative:

* **Corrupt cases are excluded by default.** 85.7 % of the dump's stored windows
  come from the 109 flagged simulations (the ``hi``/``un`` selectors preferentially
  sample them, because solver blowups ARE the highest peaks). Plotting one shows
  the model failing to reproduce a numerical artifact. ``--allow-flagged`` restores
  the old behaviour.
* **Selector preference**, default ``ty,sn,hi,un`` -- ``ty`` is the median-peak
  reservoir sample, i.e. a genuinely typical sea state, whereas ``hi``/``un`` are
  tail selections by construction.
* **A minimum sea-state excitation**, ``--min-amp`` (default 0.15 kN), applied to
  the standard deviation of the TRUE tension at the submerged node.

The last one needs justifying, because filtering windows can look like cherry
picking. It is a criterion on the ground truth alone -- the prediction is never
consulted -- and it exists because $R^2$ stops being informative on a nearly
static signal. Measured over the clean candidates in the v5 dump, the model's
MAE is essentially constant at 85-130 N regardless of sea state, while $R^2$
tracks the denominator almost perfectly:

    true sigma  0.09 kN -> R2 0.11 |  0.15 -> 0.65 |  0.20 -> 0.76
                0.45    -> R2 0.95 |  1.0  -> 0.98 |  1.9  -> 0.99

v5's test MAE is 45 N, so the 0.15 kN default asks the sea state to excite the
line to roughly 3x the model's own mean absolute error. Below that a panel
reports how calm the sea was, not how good the model is. Pass ``--min-amp 0``
to disable. The same reasoning (at 0.3 kN) already gates the temporal
predictability study documented in CLAUDE.md.

In the v5 dump this excludes only loc10 windows -- h0 = 15 m, the shallowest
location, where the mean fairlead tension is ~3.2 kN against ~6-10 kN elsewhere.
Their MAE is unremarkable (85-100 N, i.e. normal); it is purely the variance in
the denominator that collapses. Raise the threshold to see only well-excited
seas; set it to 0 to include the calm shallow-water cases.

Usage
-----
    python plot_tension_along_line.py                          # v3 dir, all N
    python plot_tension_along_line.py <dir> --out figs/
    python plot_tension_along_line.py <dir> -N 21
    python plot_tension_along_line.py <dir> -N 4,8,15,21 --rows 3
"""
import argparse
import csv
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CACHE_DIR = pathlib.Path(r"C:\Users\thano\Desktop\data\cache_npy")
ROOT = pathlib.Path(__file__).resolve().parent
SCAN_CSV = ROOT / "corrupt_scan.csv"
DEFAULT_DIR = pathlib.Path(
    r"C:\Users\thano\Desktop\Literature Master Thesis"
    r"\Results\checkpoints_P1_matched_v3_gc"
)

CONTACT_M = 0.01            # z below this = resting on the seabed
TD_LO, TD_HI = 0.02, 0.98   # contact-fraction bounds for "intermittent"

C_TRUE, C_PRED, C_GRAY = "#1F2A37", "#4884D6", "#6B7280"
# Pre-blended #E8B44A at alpha 0.20 over white. EPS/PostScript has no
# transparency, so alpha would render as solid amber and swamp the traces --
# baking the blend in keeps PNG and EPS identical.
C_BED = "#FAF0DB"

REGIMES = ("grounded", "transition", "submerged")
REGIME_TITLE = {
    "grounded":   "Grounded (on seabed)",
    "transition": "Transition (touchdown)",
    "submerged":  "Fully submerged",
}


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #
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


def contact_fraction(loc, case, start, n_steps, target_N):
    """(contact mask [steps, N], per-node contact fraction [N]) or (None, None)."""
    f = CACHE_DIR / f"loc{loc:02d}" / f"case_{case:04d}" / "z_abs.npy"
    if not f.exists():
        return None, None
    z = np.asarray(np.load(f, mmap_mode="r")[start:start + n_steps], dtype=np.float64)
    if z.shape[0] < n_steps:
        return None, None
    inc = resample(z, target_N) < CONTACT_M
    return inc, inc.mean(0)


def sea_state(loc, case):
    """(h0, Hs, Tp) from env.npy -- layout [h0, Hs, Tp, d1, v1, ...]."""
    f = CACHE_DIR / f"loc{loc:02d}" / f"case_{case:04d}" / "env.npy"
    if not f.exists():
        return None
    e = np.load(f)
    return float(e[0]), float(e[1]), float(e[2])


def flagged_cases():
    """The 109-case union: shape-blowup OR non-finite, from corrupt_scan.csv.

    ``n_blowup > 0 OR n_nonfinite > 0`` -- both are required; the shape rule
    alone misses the 8 loc07 non-finite cases (CLAUDE.md, rescan section). NOT
    the CSV's ``blacklisted`` column, which is only the legacy 53.
    """
    if not SCAN_CSV.exists():
        print(f"[warn] {SCAN_CSV.name} not found -- corrupt cases NOT excluded")
        return set()
    out = set()
    with open(SCAN_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["n_blowup"]) > 0 or int(r["n_nonfinite"]) > 0:
                out.add((int(r["loc"]), int(r["case"])))
    return out


def collect_windows(dump):
    """Every stored window, per-topology AND pooled, keyed by node count."""
    per = {}
    for key in dump.files:
        m = re.match(r"^(?:nc\d+_)?(hi|un|ty|sn)\d+_meta$", key)
        if not m:
            continue
        stem = key[:-len("_meta")]
        loc, case, nc, start = (int(v) for v in dump[key])
        per.setdefault(nc, []).append(
            dict(stem=stem, sel=m.group(1), loc=loc, case=case, nc=nc, start=start)
        )
    return per


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def pick_nodes(frac):
    """One node index per regime; None when the regime is absent."""
    grounded = np.where(frac >= TD_HI)[0]
    trans    = np.where((frac > TD_LO) & (frac < TD_HI))[0]
    subm     = np.where(frac <= TD_LO)[0]

    out = {}
    if trans.size:
        # the most genuinely intermittent node
        out["transition"] = int(trans[np.argmin(np.abs(frac[trans] - 0.5))])
        out["transition_fallback"] = False
    elif grounded.size:
        # no node straddles the touchdown point -- use the last grounded node
        out["transition"] = int(grounded.max())
        out["transition_fallback"] = True
    else:
        out["transition"] = None
        out["transition_fallback"] = False

    # The fallback transition node is itself grounded, so exclude it here -- at
    # N=4 the grounded set is two nodes and both columns would show the same one.
    def mid(idx):
        keep = [int(i) for i in idx if i != out["transition"]]
        return keep[len(keep) // 2] if keep else None

    # grounded: middle of the grounded stretch, away from the touchdown point
    out["grounded"] = mid(grounded)
    # submerged: middle of the suspended stretch, not necessarily the fairlead
    out["submerged"] = mid(subm)
    return out


def prepare(dump, wins, n_rows, dirty=frozenset(), prefer=("ty", "sn", "hi", "un"),
            min_amp=0.20):
    """Annotate windows with contact/sea state, drop unusable ones, pick rows."""
    def rank(w):
        return prefer.index(w["sel"]) if w["sel"] in prefer else len(prefer)

    cand = []
    seen = set()
    for w in wins:
        if (w["loc"], w["case"]) in dirty:
            continue
        key = (w["loc"], w["case"], w["start"])
        if key in seen:
            continue
        seen.add(key)
        true = dump[w["stem"] + "_true"]
        inc, frac = contact_fraction(w["loc"], w["case"], w["start"],
                                     true.shape[0], w["nc"])
        env = sea_state(w["loc"], w["case"])
        if inc is None or env is None:
            continue
        nodes = pick_nodes(frac)
        if nodes["grounded"] is None or nodes["submerged"] is None:
            continue
        # ground-truth-only excitation test -- see the module docstring
        amp = float(true[:, nodes["submerged"]].std()) / 1e3
        if amp < min_amp:
            continue
        w = dict(w, inc=inc, frac=frac, env=env, nodes=nodes, amp=amp,
                 true=true, pred=dump[w["stem"] + "_pred"])
        cand.append(w)
    if not cand:
        return []

    # prefer windows that carry a genuine intermittent node, then spread over Hs
    cand.sort(key=lambda w: w["env"][1])
    real_td = [w for w in cand if not w["nodes"]["transition_fallback"]]
    pool = real_td if len(real_td) >= n_rows else cand

    # one window per (loc, case) -- the best-ranked selector for that case
    by_case = {}
    for w in pool:
        k = (w["loc"], w["case"])
        if k not in by_case or rank(w) < rank(by_case[k]):
            by_case[k] = w
    uniq = sorted(by_case.values(), key=lambda w: w["env"][1])
    if len(uniq) <= n_rows:
        return uniq

    # Split the Hs-sorted candidates into n_rows contiguous bands and take the
    # best-ranked window from each. Spread across sea states is the primary
    # goal -- selector preference only decides WITHIN a band, so a figure never
    # degenerates into three near-identical seas just because they are all "ty".
    bounds = np.linspace(0, len(uniq), n_rows + 1).round().astype(int)
    chosen, used_locs = [], set()
    for a, b in zip(bounds[:-1], bounds[1:]):
        grp = uniq[a:max(int(b), int(a) + 1)]
        # selector rank first, then a location not already on the figure
        w = min(grp, key=lambda w: (rank(w), w["loc"] in used_locs, w["env"][1]))
        used_locs.add(w["loc"])
        chosen.append(w)
    return chosen


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def shade_contact(ax, inc_node, t):
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
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=C_BED, lw=0, zorder=0)


def r2(true, pred):
    ss_tot = float(((true - true.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(((true - pred) ** 2).sum()) / ss_tot


def panel(ax, t, w, regime, show_ylabel):
    i = w["nodes"][regime]
    tr = w["true"][:, i].astype(np.float64) / 1e3      # kN
    pr = w["pred"][:, i].astype(np.float64) / 1e3
    shade_contact(ax, w["inc"][:, i], t)
    ax.plot(t, tr, color=C_TRUE, lw=1.3, zorder=3)
    ax.plot(t, pr, color=C_PRED, lw=1.1, zorder=4)

    # headroom so the stat box never sits on top of the traces
    lo = min(tr.min(), pr.min())
    hi = max(tr.max(), pr.max())
    pad = max(hi - lo, 1e-6)
    ax.set_ylim(lo - 0.06 * pad, hi + 0.34 * pad)
    ax.set_xlim(t[0], t[-1])

    frac = w["frac"][i]
    tag = f"node {i}"
    if regime == "transition":
        if w["nodes"]["transition_fallback"]:
            tag += "  (touchdown-adjacent)"
        else:
            tag += f"  (in contact {100 * frac:.0f}% of window)"
    elif regime == "submerged" and i == w["nc"] - 1:
        tag += "  (fairlead)"

    ax.set_title(tag, fontsize=8.5, color=C_GRAY, pad=3)
    ax.text(0.015, 0.965,
            f"$R^2$ = {r2(tr, pr):.3f}\nMAE = {np.abs(tr - pr).mean() * 1e3:.0f} N",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            color=C_TRUE,
            zorder=6,
            bbox=dict(fc="white", ec=C_GRAY, lw=0.4,
                      boxstyle="round,pad=0.25"))
    if show_ylabel:
        ax.set_ylabel("tension  [kN]", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.18, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def make_figure(N, rows, out_dir, formats):
    t = np.arange(rows[0]["true"].shape[0]) * 0.1        # dt = 0.1 s
    nr, nc = len(rows), len(REGIMES)
    fig, axes = plt.subplots(nr, nc, figsize=(4.05 * nc, 2.15 * nr + 1.05),
                             squeeze=False)

    for r, w in enumerate(rows):
        for c, regime in enumerate(REGIMES):
            panel(axes[r][c], t, w, regime, show_ylabel=(c == 0))
            if r == 0:
                axes[r][c].annotate(
                    REGIME_TITLE[regime],
                    xy=(0.5, 1.30), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=10.5, weight="bold",
                    color=C_TRUE)
            if r == nr - 1:
                axes[r][c].set_xlabel("time  [s]", fontsize=8.5)

        h0, hs, tp = w["env"]
        axes[r][0].annotate(
            f"loc {w['loc']:02d} / case {w['case']:04d}\n"
            f"$H_s$ = {hs:.2f} m\n$T_p$ = {tp:.2f} s\n$h_0$ = {h0:.1f} m",
            xy=(-0.375, 0.5), xycoords="axes fraction",
            ha="center", va="center", fontsize=8.5, color=C_TRUE,
            bbox=dict(fc="#F0F5FF", ec=C_PRED, lw=0.8,
                      boxstyle="round,pad=0.40"))

    fig.legend(handles=[
        Line2D([], [], color=C_TRUE, lw=1.6, label="FE ground truth"),
        Line2D([], [], color=C_PRED, lw=1.6, label="GAT+LSTM prediction"),
        matplotlib.patches.Patch(fc=C_BED, ec=C_GRAY, lw=0.4,
                                 label="node in seabed contact"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=9.5,
        bbox_to_anchor=(0.5, 0.002))

    fig.suptitle(f"Tension time series along the line  —  N = {N} nodes",
                 fontsize=13, weight="bold", color=C_TRUE, y=0.995)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.855 if nr > 2 else 0.80,
                        bottom=0.115, hspace=0.42, wspace=0.22)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Tension_time_series_along_line_n={N}"
    written = []
    for ext in formats:
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=170 if ext == "png" else None,
                    format=ext, bbox_inches=None)
        written.append(p)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--out", default=None, help="output dir (default <dir>/figs_along_line)")
    ap.add_argument("-N", default="all", help="node counts, e.g. 21 or 4,12,21 or all")
    ap.add_argument("--rows", type=int, default=3, help="sea states per figure")
    ap.add_argument("--formats", default="png,eps")
    ap.add_argument("--allow-flagged", action="store_true",
                    help="do NOT exclude the 109 corrupt cases (off by default)")
    ap.add_argument("--prefer", default="ty,sn,hi,un",
                    help="selector preference within a sea-state band")
    ap.add_argument("--min-amp", type=float, default=0.15,
                    help="min std of TRUE submerged-node tension [kN]; 0 disables")
    args = ap.parse_args()

    d = pathlib.Path(args.dump_dir)
    npz = d if d.suffix == ".npz" else d / "test_timeseries_dump.npz"
    if not npz.exists():
        sys.exit(f"not found: {npz}")
    out_dir = pathlib.Path(args.out) if args.out else npz.parent / "figs_along_line"
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    dump = np.load(npz, allow_pickle=True)
    per = collect_windows(dump)
    dirty = set() if args.allow_flagged else flagged_cases()
    prefer = tuple(s.strip() for s in args.prefer.split(",") if s.strip())
    wanted = (sorted(per) if args.N == "all"
              else [int(x) for x in args.N.replace(" ", "").split(",")])

    print(f"dump : {npz}")
    print(f"out  : {out_dir}")
    print(f"cases: {'ALL (flagged included)' if args.allow_flagged else f'{len(dirty)} corrupt cases excluded'}"
          f" | selector preference {'>'.join(prefer)} | min-amp {args.min_amp:.2f} kN\n")
    for N in wanted:
        if N not in per:
            print(f"N={N:<3d} -- no windows in dump")
            continue
        rows = prepare(dump, per[N], args.rows, dirty, prefer, args.min_amp)
        if not rows:
            print(f"N={N:<3d} -- no usable window (all flagged / too calm / cache missing)")
            continue
        written = make_figure(N, rows, out_dir, formats)
        td = sum(not w["nodes"]["transition_fallback"] for w in rows)
        print(f"N={N:<3d} {len(rows)} sea states | "
              f"{td}/{len(rows)} with a genuine intermittent node -> {written[0].name}")
        for w in rows:
            print(f"      {w['sel']:2s} loc{w['loc']:02d}/case{w['case']:04d} "
                  f"start={w['start']:<6d} Hs={w['env'][1]:.2f} m  Tp={w['env'][2]:.2f} s  "
                  f"sigma_T={w['amp']:.2f} kN")


if __name__ == "__main__":
    main()
