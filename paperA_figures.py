"""Generate the Results figures of Paper A (Section 4) in one consistent style.

Local only: reads the .npz test dumps, the snap catalogue and the npy cache (for
seabed contact). Never executes a notebook (memory feedback_never_run_notebook_locally).

Style
  * STIX, the manuscript's own font (cas-dc loads stix); 7.5 pt at print size.
  * Widths match the page: full text width 6.84 in, one column 3.30 in (cas-dc
    geometry: 210 mm paper, 18.1 mm side margins, 18 pt column gap).
  * Palette: the first three slots of the dataviz reference palette (blue, orange,
    aqua), which that palette documents as passing the all-pairs colour-vision checks
    in light mode. Truth is always solid near-black and a prediction always dashed,
    so no comparison relies on colour alone. Seabed contact is a pale surface tint.
  * No in-figure titles; captions carry the description (CLAUDE.md 14.4).

Window choice is by GROUND TRUTH ONLY (site, sea state, selector), never by how well
the model did, and every choice is written down next to the figure that uses it. The one
exception is the pair of snap-station figures, whose panels are chosen as illustrations and
named one by one in SNAP_PANELS_SONE / SNAP_PANELS_STWO; the manuscript says so, and the
capture distribution they illustrate is Table 11, computed over the full test sets.

Inputs are found through paperA_paths.py (development layout, or the unpacked 4TU archives
when PAPERA_DATA is set). Figures 4, 8 and D.1 read the seabed contact from the raw simulations
(MOORING_CACHE_DIR) and are skipped when that cache is absent.

Usage
  python paperA_figures.py            # all figures -> paper_A_manuscript/figures/
  python paperA_figures.py --png DIR  # also write PNG previews into DIR
"""
import argparse
import collections
import csv
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paperA_paths as P
from paperA_paths import npload
import plot_tension_along_line as ALONG   # window collection, flagged cases, contact

ROOT = pathlib.Path(__file__).parent
SF = ROOT / "Results" / "eval_v5_snapfix"
# Every test-set distribution (Figs 5, 7, 10, 11 and the appendix grid) is read from the EVAL_CLEAN
# job (job 402698): the shipped models on the published draws minus the windows the evaluation mask
# covers. Hand-picked windows (Figs 4, 6, 8, 9) stay in the dumps they were chosen from -- a window's
# prediction does not depend on the rest of the draw -- and must contain no evaluation-mask sample.
EVAL_CLEAN = ROOT / "Results" / "SR_RESULTS" / "checkpoints_P2_eval_clean"
T1, T2 = EVAL_CLEAN / "stage1", EVAL_CLEAN
SNAP_CAT = ROOT / "snap_catalogue_cat.csv"
MASK_EVAL = ROOT / "artifact_samples_cat.csv"
OUT = P.OUT_DIR / "figures"
DT = 0.1

FULL_W, COL_W = 6.84, 3.30
C_TRUE = "#1f1f1e"
C_BLUE = "#2a78d6"     # slot 1: the model's prediction / snap windows
C_ORANGE = "#eb6834"   # slot 2: loc10
C_AQUA = "#1baf7a"     # slot 3: loc11
C_MUTED = "#a3a29c"
C_RED = "#d93a35"      # Fig. 5a: windows without a snap event
C_BED = "#efe7d6"
C_GRID = "#e4e3de"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


_MASK = None


def eval_mask_hit(loc, case, start, n):
    """True if the window [start, start + n) holds a sample of the evaluation mask."""
    global _MASK
    if _MASK is None:
        with P.open_(MASK_EVAL, newline="", encoding="utf-8") as fh:
            _MASK = {(int(r["loc"]), int(r["case"])): np.array(sorted(int(x) for x in r["bad_t"].split()))
                     for r in csv.DictReader(fh)}
    b = _MASK.get((int(loc), int(case)))
    if b is None or not b.size:
        return False
    j = np.searchsorted(b, int(start))
    return bool(j < b.size and b[j] < int(start) + int(n))


def r2(t, p):
    sst = float(((t - t.mean()) ** 2).sum())
    return 1.0 - float(((t - p) ** 2).sum()) / sst if sst > 0 else float("nan")


def shade_contact(ax, inc, t):
    """Pale spans wherever the station rests on the seabed."""
    if inc is None or not inc.any():
        return
    d = np.diff(inc.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1) if not inc[0] else [0] + list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1) + ([len(inc)] if inc[-1] else [])
    for a, b in zip(starts, ends):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=C_BED, lw=0, zorder=0)


def save(fig, name, png_dir):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_dir / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[fig] {name}")


# ------------------------------------------------------------------ Stage 1, tension along the line
# One stored test window per layout, chosen on ground truth alone so that no choice depends on how
# well the model did: unbiased 'typical' reservoir windows (ty), except where noted. A window
# qualifies when its simulation is flagged by no artifact rule and the suspended station is excited
# above 0.15 kN (plot_tension_along_line.prepare). The touchdown panel shows the station of the
# window closest to half of it on the seabed (ALONG.pick_nodes), and every row must show a station
# that really leaves the seabed. That excludes N = 4: in all 17 clean stored N = 4 windows of the
# selected checkpoint the last grounded station is on the seabed throughout and the next one is
# suspended throughout (the touchdown point lies between two stations), so the layout is not shown.
# Rows and sites:
#   * N = 6, loc07: 36 % of the window on the seabed. At N = 6 only loc07 has such a window; it is
#     also the calm sea (Hs 1.2 m) that the text uses for the suspended-station R^2;
#   * N = 10, loc11: 94 %. Both typical loc11 windows of this layout have no station leaving the
#     seabed, so the only one that does is used, a snap window (sn) with one catalogued snap. It is
#     the only snap window of the figure;
#   * N = 15, loc10: 89 %, loc10 being the shallow withheld site. Its only clean typical window that
#     is excited and has a station leaving the seabed;
#   * N = 21, loc04: 59 %, the closest to half of the window of any site not already shown.
# Both withheld sites appear and no site twice. Windows are the selected checkpoint's (epoch 32),
# read from two of its test dumps: the run's own 50 000-window dump ("v5") and the eval-only dump
# ("snapfix"). A window in both has identical predictions (140 windows, largest difference 0 N,
# checked 2026-09-19).
DUMPS = {"v5": ROOT / "Results" / "checkpoints_P1_matched_v5" / "test_timeseries_dump.npz",
         "snapfix": SF / "test_timeseries_dump.npz"}
ALONG_ROWS = [(6, "v5", "ty2"),             # loc07/0228, Hs 1.17 m -- calm; station 3, seabed 36 %
              (10, "v5", "nc10_sn0"),       # loc11/0252, Hs 3.80 m -- withheld; station 5, 94 %; one snap
              (15, "snapfix", "nc15_ty1"),  # loc10/0085, Hs 3.07 m -- withheld, shallow; station 11, 89 %
              (21, "snapfix", "nc21_ty1")]  # loc04/0149, Hs 3.23 m -- station 11, seabed 59 %


def fig_along_line(png_dir):
    dumps = {k: npload(v, allow_pickle=True) for k, v in DUMPS.items()}
    dirty = ALONG.flagged_cases()
    fig, axes = plt.subplots(len(ALONG_ROWS), 3, figsize=(FULL_W, 7.3), squeeze=False)
    sites, kinds = [], []
    for r, (N, dump, stem) in enumerate(ALONG_ROWS):
        z = dumps[dump]
        loc, case, nc, start = (int(v) for v in z[stem + "_meta"])
        assert nc == N and (loc, case) not in dirty, (stem, nc, loc, case)
        assert not eval_mask_hit(loc, case, start, z[stem + "_true"].shape[0]), (stem, loc, case, start)
        kinds.append("sn" if "sn" in stem else "ty" if "ty" in stem else "?")
        sites.append(loc)
        true, pred = z[stem + "_true"], z[stem + "_pred"]
        inc, frac = ALONG.contact_fraction(loc, case, start, true.shape[0], N)
        nodes = ALONG.pick_nodes(frac)
        # every row must show a station that really leaves the seabed
        assert not nodes["transition_fallback"], (N, loc, case, frac)
        print(f"[along] N={N:2d} loc{loc:02d}/{case:04d}  touchdown station {nodes['transition']} "
              f"on the seabed {100 * frac[nodes['transition']]:.0f} % of the window")
        h0, hs, tp = ALONG.sea_state(loc, case)
        t = np.arange(true.shape[0]) * DT
        cols = [("grounded", nodes["grounded"]), ("touchdown", nodes["transition"]),
                ("fairlead", N - 1)]
        for c, (kind, i) in enumerate(cols):
            ax = axes[r][c]
            tr, pr = true[:, i] / 1e3, pred[:, i] / 1e3
            shade_contact(ax, inc[:, i], t)
            ax.plot(t, tr, color=C_TRUE, lw=0.9, zorder=3)
            ax.plot(t, pr, color=C_BLUE, lw=0.9, ls=(0, (3.2, 1.6)), zorder=4)
            lo, hi = min(tr.min(), pr.min()), max(tr.max(), pr.max())
            pad = max(hi - lo, 1e-6)
            ax.set_ylim(lo - 0.05 * pad, hi + 0.30 * pad)
            ax.set_xlim(t[0], t[-1])
            if kind == "touchdown":
                where = f"on seabed {100 * frac[i]:.0f}% of window"
            elif kind == "grounded":
                where = "on seabed throughout"
            else:
                where = "suspended"
            ax.text(0.015, 0.97, f"station {i}, {where}\n"
                    f"$R^2$ = {r2(tr, pr):.3f}, MAE = {1e3 * np.abs(tr - pr).mean():.0f} N",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.5, zorder=6)
            if c == 0:
                ax.set_ylabel("Tension [kN]")
            if r == len(ALONG_ROWS) - 1:
                ax.set_xlabel("Time [s]")
            else:
                ax.set_xticklabels([])
        axes[r][0].annotate(
            f"({'abcd'[r]})  $N$ = {N},  loc{loc:02d},  $H_s$ = {hs:.1f} m,  $T_p$ = {tp:.1f} s",
            xy=(0.0, 1.03), xycoords="axes fraction", ha="left", va="bottom", fontsize=7.5)
    assert len(set(sites)) == len(sites) and {10, 11} <= set(sites), sites   # both withheld sites, none twice
    assert set(kinds) <= {"ty", "sn"} and kinds.count("sn") <= 1, kinds      # typical windows, at most one snap
    for c, title in enumerate(("Grounded", "Touchdown", "Suspended (fairlead)")):
        axes[0][c].set_title(title, pad=16, fontsize=8, fontfamily="STIXGeneral")
    fig.legend(handles=[Line2D([], [], color=C_TRUE, lw=1.1, label="Finite-element truth"),
                        Line2D([], [], color=C_BLUE, lw=1.1, ls=(0, (3.2, 1.6)),
                               label="Reconstruction"),
                        Patch(fc=C_BED, ec="none", label="Station on the seabed")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.08,
                        hspace=0.42, wspace=0.18)
    save(fig, "fig_alongline", png_dir)


# ------------------------------------------------------------------ Figure 5
def snap_event_table(z):
    """Per distinct event: median true / predicted line maximum, and the site."""
    g = collections.defaultdict(list)
    for (lc, case, _n, _s, t), a, b in zip(z["snap_ev"], z["snap_true"], z["snap_pred"]):
        if a > 0:
            g[(int(lc), int(case), int(t))].append((float(a), float(b)))
    return {k: (np.median([x for x, _ in v]), np.median([y for _, y in v])) for k, v in g.items()}


def fig_peaks(png_dir):
    z = npload(T1 / "test_timeseries_dump.npz")
    meta, pt, pp = z["meta_all"], z["peak_true_all"] / 1e3, z["peak_pred_all"] / 1e3
    keys = {(int(a), int(b), int(c), int(d)) for a, b, c, d, _t in z["snap_ev"]}
    snap = np.array([(int(m[0]), int(m[1]), int(m[2]), int(m[3])) in keys for m in meta])

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.95))
    top = float(max(pt.max(), pp.max())) * 1.04
    a.scatter(pt[~snap], pp[~snap], s=1.2, color=C_RED, alpha=0.6, lw=0,
              rasterized=True, label="Window without a snap event")
    a.scatter(pt[snap], pp[snap], s=3.0, color=C_BLUE, alpha=0.85, lw=0,
              rasterized=True, label="Window with a snap event")
    a.plot([0, top], [0, top], color=C_TRUE, lw=0.6)
    for q, lab in ((0.90, "$p_{90}$"), (0.99, "$p_{99}$")):
        thr = float(np.quantile(pt, q))
        a.axvline(thr, color=C_TRUE, lw=0.5, ls=(0, (2, 2)))
        a.text(thr, top * 0.985, lab + " ", ha="right", va="top", fontsize=6.5)
    a.set_xlim(0, top); a.set_ylim(0, top); a.set_aspect("equal")
    a.set_xlabel("True fairlead peak [kN]"); a.set_ylabel("Reconstructed fairlead peak [kN]")
    a.legend(loc="lower right", frameon=False, markerscale=3, handletextpad=0.2)
    a.text(0.02, 0.98, "(a)", transform=a.transAxes, ha="left", va="top", fontsize=8)

    ev = snap_event_table(z)
    groups = (("Training sites", lambda lc: lc not in (10, 11), C_BLUE, "o"),
              ("loc10 (withheld)", lambda lc: lc == 10, C_ORANGE, "^"),
              ("loc11 (withheld)", lambda lc: lc == 11, C_AQUA, "s"))
    T = np.array([v[0] for v in ev.values()]) / 1e3
    P = np.array([v[1] for v in ev.values()]) / 1e3
    top_b = float(max(T.max(), P.max())) * 1.05
    for label, sel, col, mk in groups:
        ks = [k for k in ev if sel(k[0])]
        b.scatter([ev[k][0] / 1e3 for k in ks], [ev[k][1] / 1e3 for k in ks], s=9,
                  marker=mk, color=col, alpha=0.8, lw=0, label=f"{label}, $n$ = {len(ks)}")
    b.plot([0, top_b], [0, top_b], color=C_TRUE, lw=0.6)
    b.set_xlim(0, top_b); b.set_ylim(0, top_b); b.set_aspect("equal")
    b.set_xlabel("True line maximum [kN]")
    b.set_ylabel("Reconstructed line maximum [kN]")
    b.legend(loc="upper left", frameon=False, handletextpad=0.2, bbox_to_anchor=(0.06, 1.0))
    b.text(0.02, 0.98, "(b)", transform=b.transAxes, ha="left", va="top", fontsize=8)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig_peaks", png_dir)


# ------------------------------------------------------------------ Figures 6 and 9
# Snap traces at the station where each snap occurs. The capture statistic of Table 10
# is taken on the LINE MAXIMUM at the event sample, and under the fairlead condition of the
# snap rule the fairlead, not the snap, sets that maximum at every catalogued event; a
# line-maximum trace therefore hides the snaps. Each panel shows instead the station where
# the snap occurs.
#
# A window QUALIFIES when (i) it contains no sample of the evaluation mask, i.e. it belongs
# to the clean test draw that Section 4 reports (the window screen of Data 2.5 subsumes the
# old anchor out-pull test), (ii) its catalogued event sits at least 3 s from either window edge,
# and (iii) the layout outputs the snap's OWN station. At matched resolution (iii) holds
# only when node*(N-1)/20 is an integer, which is why the Stage-1 pool is small; Stage 2
# emits the twenty-one stations whatever the input layout, so (iii) always holds there.
#
# Stage 2 adds two further conditions, both on ground truth alone:
#   (iv) the layout must NOT SENSE the station, i.e. node*(N_in-1)/20 is not an integer, so
#        no sensor sits AT any Stage-2 panel's station (a sensor may sit close to it, see
#        the note below). At N_in = 4 that is every station but the anchor and the fairlead;
#    (v) LEGIBILITY -- no unmarked sample of the panel's own trace may out-top the labelled
#        event (by more than 2 %). Without it a taller uncatalogued spike beside the labelled
#        one reads as the model missing a peak it was never asked about. This is what rejects
#        the other two loc10 events: loc10/0275 carries 7.9 kN at the same station 0.7 s
#        before its 4.9 kN labelled event, and loc10/0281 7.7 kN beside 4.5 kN.
#
# WARNING: the panels are ILLUSTRATIONS, not a random draw, and the manuscript says so.
# They were picked from the qualifying windows, one per simulation, to span the range of
# true peak and to cover the withheld sites. Table 11 carries the capture distribution over
# the full test sets, and that is the number to quote. Every panel is named below, so the
# choice is auditable. No qualifying loc10 window exists at matched resolution, which is
# why Figure 6 has no loc10 panel.
#
# ⚠ Figure 9 shows FOUR SENSORS ONLY (author's decision, 2026-09-21). The pool is every
# qualifying four-sensor window in the three dumps of the shipped checkpoint: the run's own dump,
# the withheld-site pass (sr-21) and the EVAL_CLEAN pass (job 402698, whose reservoirs were drawn
# on the clean test draw). Under the evaluation mask and catalogue (2026-09-24) the qualifying
# four-sensor simulations are six, ALL at loc11: loc01/0063 (its 21.0 kN event fails the fairlead
# condition, station 11 at 1.16 x the fairlead, and its window holds evaluation-mask samples) and
# loc10/0187 (one evaluation-mask sample in the window) left the pool, and loc11/0080 entered it
# from the EVAL_CLEAN dump. The figure is all six, one panel per simulation, and the event drawn
# is the LARGEST in its simulation, so that no panel is chosen by how well it was reconstructed.
#
# "Unsensed" means only that no sensor sits AT the station. Sensors are equally spaced in
# arc length and coincide with 21-grid stations only where node*(N_in-1)/20 is an integer,
# so an unsensed station can lie close to a sensor: every panel here sits 1.33-2.33 station
# spacings from the nearest sensor. The manuscript wording is "no sensor sits at the station",
# never "no measurement".
SNAP_PANELS_SONE = [           # (loc, case, N, event row), Stage 1, ordered by true peak
    (11, 252, 21, 489),
    (4, 133, 21, 221),
    (9, 34, 7, 43),
    (1, 243, 12, 352),
    (11, 11, 6, 405),
    (9, 43, 6, 492),
]
SNAP_PANELS_STWO = [           # (loc, case, N_in, event row), Stage 2, ordered by true peak
    (11, 22, 4, 174),            # 12.8 kN
    (11, 11, 4, 93),             # 11.4 kN (its second event, 7.1 kN, is marked too)
    (11, 85, 4, 553),            #  9.5 kN
    (11, 150, 4, 69),            #  8.7 kN
    (11, 80, 4, 526),            #  8.2 kN -- from the EVAL_CLEAN dump
    (11, 89, 4, 344),            #  6.0 kN
]


def snap_station_candidates(z, stage2=False, into=None):
    """Qualifying snap-station windows, keyed by (loc, case, N, event row).

    Conditions (i)-(iii) of the comment above, plus (iv) unsensed and (v) legible for
    Stage 2. One entry per catalogued event, carrying the whole window so that a panel can
    show its full 100 s and mark every catalogued event in it -- without which a second
    snap in the same window reads as the model spiking at nothing.

    `into` accumulates across dumps. When the same physical event is stored at the same
    layout by two dumps, the window whose event sits FURTHEST from either edge wins: the
    model is bidirectional, so an event near a window edge has lost half its context.
    """
    import re
    cat = {(int(r["loc"]), int(r["case"]), int(r["t"])): r
           for r in csv.DictReader(P.open_(SNAP_CAT, encoding="utf-8"))}
    edge = int(round(3.0 / DT))
    out = {} if into is None else into
    stems = r"^((?:nc\d+_)?(?:sn|ty|hi|un)\d+|site\d+_n\d+_(?:cl|sn)\d+)_evrows$"
    for key in z.files:
        m = re.match(stems if stage2 else r"^((?:nc\d+_)?(?:sn|ty|hi|un)\d+)_evrows$", key)
        if not m:
            continue
        stem = m.group(1)
        loc, case, nc, start = (int(v) for v in z[stem + "_meta"])
        true, pred = z[stem + "_true"], z[stem + "_pred"]
        if eval_mask_hit(loc, case, start, true.shape[0]):
            continue                                    # condition (i)
        evrows = np.atleast_1d(z[key]).astype(int)
        marks = [int(e) for e in evrows if (loc, case, start + int(e)) in cat]
        for e in evrows:
            c = cat.get((loc, case, start + int(e)))
            if c is None or not (edge <= e < len(true) - edge):
                continue                                # condition (ii)
            node = int(c["node"])
            if stage2:
                i = node
                if (node * (nc - 1)) % 20 == 0:
                    continue                            # condition (iv): sensed station
                ts = true[:, i]
                taller = np.setdiff1d(np.where(ts > ts[e] * 1.02)[0], np.asarray(marks, int))
                if taller.size:
                    continue                            # condition (v): legibility
            else:
                if (node * (nc - 1)) % 20:
                    continue                            # condition (iii)
                i = node * (nc - 1) // 20
            rec = dict(loc=loc, case=case, nc=nc, row=int(e), station=i, node=node,
                       true=true, pred=pred, marks=marks, peak=float(true[e, i]),
                       capture=float(pred[e, i] / true[e, i]),
                       centred=min(int(e), len(true) - 1 - int(e)))
            # one entry per (simulation, layout, PHYSICAL event); keep the best-centred window
            ev = (loc, case, nc, start + int(e))
            prev = out.get(ev)
            if prev is None or rec["centred"] > prev["centred"]:
                out[ev] = rec
    return out


def snap_panels(stage2=False):
    """The named panels, in order, with their traces attached, and the pool size.

    The Stage-2 pool is the union of the qualifying windows of the three evaluation passes
    of the shipped checkpoint: the run's own dump, the withheld-site pass (patch sr-21,
    which adds per-site reservoirs at four sensors) and the EVAL_CLEAN pass (job 402698).
    They score the same weights on the same test draw (EVAL_CLEAN on its clean subset) and
    differ only in which reservoir windows they kept, so the union is simply a larger sample
    of stored windows; condition (i) keeps only windows of the clean draw.
    """
    cand = {}
    for d in ((S2, S2W, T2) if stage2 else (SF,)):
        f = d / "test_timeseries_dump.npz"
        if P.exists(f):
            snap_station_candidates(npload(f, allow_pickle=True), stage2, cand)
    by_row = {(r["loc"], r["case"], r["nc"], r["row"]): r for r in cand.values()}
    keys = SNAP_PANELS_STWO if stage2 else SNAP_PANELS_SONE
    missing = [k for k in keys if k not in by_row]
    if missing:
        raise SystemExit(f"snap panels not in the qualifying pool: {missing}")
    snap_panels.pool = by_row              # the whole qualifying pool, for the events left out
    return [by_row[k] for k in keys], len({(k[0], k[1]) for k in cand})


def _snap_grid(png_dir, stage2, name, nrow, ncol, height):
    panels, n_sims = snap_panels(stage2)
    sym = r"$N_\mathrm{in}$" if stage2 else "$N$"
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_W, height), squeeze=False)
    for k, r in enumerate(panels):
        ax = axes[k // ncol][k % ncol]
        i, e = r["station"], r["row"]
        ts, ps = r["true"][:, i] / 1e3, r["pred"][:, i] / 1e3
        t = np.arange(len(ts)) * DT
        top = max(ts.max(), ps.max()) * 1.45
        # Every catalogued event in the window is marked, or a second snap in the same
        # window reads as the model spiking at nothing; the one the label quantifies is
        # the solid line, the others are faint. The lines stop below the label band and
        # the data below that again, so nothing is ever written over.
        for mk in r["marks"]:
            ax.axvline(mk * DT, color=C_ORANGE, lw=0.8, ymax=0.82,
                       alpha=0.9 if mk == e else 0.3, zorder=1)
        ax.plot(t, ts, color=C_TRUE, lw=0.8, zorder=3)
        ax.plot(t, ps, color=C_BLUE, lw=0.8, ls=(0, (3.2, 1.6)), zorder=4)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(min(0.0, ts.min(), ps.min()), top)
        ax.text(0.015, 0.98,
                f"({'abcdefghi'[k]})  loc{r['loc']:02d}, {sym} = {r['nc']}, "
                f"station {i}\npeak {ts[e]:.1f} kN, reconstructed {ps[e]:.1f} kN",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
        if k % ncol == 0:
            ax.set_ylabel("Tension [kN]")
        if k // ncol == nrow - 1:
            ax.set_xlabel("Time [s]")
    fig.legend(handles=[Line2D([], [], color=C_TRUE, lw=1.1, label="Finite-element truth"),
                        Line2D([], [], color=C_BLUE, lw=1.1, ls=(0, (3.2, 1.6)),
                               label="Reconstruction"),
                        Line2D([], [], color=C_ORANGE, lw=1.1, label="Catalogued snap event")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.98,
                        bottom=0.17 if nrow == 2 else 0.125, hspace=0.3, wspace=0.2)
    save(fig, name, png_dir)
    print(f"      {len(panels)} panels of {n_sims} qualifying simulations")
    for r in panels:
        print(f"      loc{r['loc']:02d}/{r['case']:04d} N={r['nc']} station {r['station']} "
              f"(node {r['node']}) true {r['peak']:.0f} N capture {r['capture']:.3f}")


def fig_snap_traces(png_dir):
    _snap_grid(png_dir, False, "fig_snaptraces", 2, 3, 3.7)


def fig_snap_traces_s2(png_dir):
    _snap_grid(png_dir, True, "fig_snaptraces_s2", 2, 3, 3.7)


# ================================================================ Section 4.2
S2 = ROOT / "Results" / "SR_RESULTS" / "checkpoints_P2_final_warm(run2)"
# The withheld-site evaluation pass of the SAME checkpoint (patch sr-21): identical
# weights, identical 50 000-window test draw -- every deterministic array in the two
# dumps is bit-identical -- with per-site reservoirs added at N_in = 4. Only the
# reservoir draws (ty, sn) differ, so anything reservoir-based must name its dump.
S2W = ROOT / "Results" / "SR_RESULTS" / "checkpoints_P2_eval_withheld"
LAYOUTS = (4, 5, 6, 7, 8, 10, 12, 15, 18, 21)


def _grid(name):
    """{(N_in, N_out): row} from one of the Stage-2 grid CSVs."""
    with P.open_(T2 / name, newline="", encoding="utf-8") as fh:
        return {(int(r["N_in"]), int(r["N_out"])): r for r in csv.DictReader(fh)}


# ------------------------------------------------------------------ Figure 7
def fig_sweep(png_dir):
    """MAE and skill score against the number of sensed stations, at N_out = 21."""
    g, gb = _grid("superres_grid.csv"), _grid("superres_grid_baseline.csv")
    x = np.array(LAYOUTS)
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.9), sharex=True)
    for ax, key, lab in ((a, "MAE_tension", "MAE [N]"),
                         (b, "temporal_diff_skill_tension", "Skill score $S$ (lower is better)")):
        d = np.array([float(g[(n, 21)][key]) for n in LAYOUTS])
        bl = np.array([float(gb[(n, 21)][key]) for n in LAYOUTS])
        ax.plot(x, bl, color=C_ORANGE, lw=1.0, ls=(0, (3.2, 1.6)), marker="^", ms=3.2,
                label="Interpolated Stage 1")
        ax.plot(x, d, color=C_BLUE, lw=1.0, marker="o", ms=3.0, label="Stage 2")
        ax.axhline(float(gb[(21, 21)][key]), color=C_TRUE, lw=0.6, ls=(0, (1, 1.5)),
                   label="Stage 1, all 21 stations sensed")
        ax.set_ylabel(lab)
    a.set_yscale("log")
    a.set_yticks([30, 50, 100, 200])
    a.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.legend(loc="upper right", frameon=False, handlelength=2.6)
    b.set_xticks(LAYOUTS)
    b.set_xlabel("Sensed stations $N_{\\mathrm{in}}$")
    for ax, letter in ((a, "(a)"), (b, "(b)")):
        ax.text(-0.2, 1.0, letter, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    fig.subplots_adjust(hspace=0.12)
    save(fig, "fig_sweep", png_dir)


# ------------------------------------------------------------------ Figure 8
# Four sensors -> the 21 native stations, in two test windows, at a grounded, a touchdown and a
# suspended station. Only stations the layout does NOT sense are shown: at N_in = 4 the sensors
# sit at s = 0, 1/3, 2/3 and 1, and only s = 0 and s = 1 fall on a native station, so every
# station except the anchor and the fairlead is unsensed. One station per region, picked from the
# true contact state by the same rule as Figure 4 (ALONG.pick_nodes): the middle of the grounded
# stretch, the most genuinely intermittent station, and the middle of the suspended stretch.
#
# ★ BOTH WINDOWS ARE WITHHELD SITES. They come from the per-site reservoirs of the
# withheld-site evaluation pass (patch sr-21), which draws snap-free, artifact-free,
# residue-free windows at N_in = 4 separately for loc10 and loc11 -- something the pooled and
# per-layout reservoirs of the earlier dumps could not do: of the ten windows they store at
# N_in = 4, the four from loc10 all carry out-pulling samples (Appendix D) and the single
# loc11 one carries a snap.
#
# Choice within each site, on ground truth alone: of the 24 reservoir windows per site, the
# ones where a station really leaves the seabed (ALONG.pick_nodes transition_fallback False)
# are 7 at loc10 and 11 at loc11 -- the shallow site's touchdown point often falls between two
# of the 21 stations -- and of those the window taken is the one whose touchdown station is the
# most genuinely intermittent, i.e. whose time on the seabed is closest to half the window.
# assert_four_window_clean re-checks what the reservoir already enforced: no masked artifact
# sample, no out-pulling sample and no catalogued snap event anywhere in either window, snap
# loads being the subject of Figures 6 and 9.
FOUR_WINDOWS = ["site10_n04_cl21",   # loc10/0268, Hs 4.0 m -- withheld, shallow; touchdown station off the seabed 19 % of the window
                "site11_n04_cl17"]   # loc11/0221, Hs 3.2 m -- withheld; off the seabed 17 %
FOUR_STEM = FOUR_WINDOWS[0]


def _samples_in_window(path, col, loc, case, start, n):
    """Flagged samples of this simulation that fall inside the window."""
    hit = []
    if not path.exists():
        return hit
    with P.open_(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["loc"]) != loc or int(r["case"]) != case:
                continue
            raw = r[col]
            for x in (raw.split() if " " in raw else [raw]):
                if x:
                    k = int(x) - start
                    if 0 <= k < n:
                        hit.append(k)
    return sorted(hit)


def assert_four_window_clean(loc, case, start, n):
    """No evaluation-mask sample, no out-pulling sample and no catalogued snap in the window."""
    art = _samples_in_window(MASK_EVAL, "bad_t", loc, case, start, n)
    out = _samples_in_window(ROOT / "anchor_outpull_samples.csv", "t", loc, case, start, n)
    snap = _samples_in_window(SNAP_CAT, "t", loc, case, start, n)
    assert not art and not out and not snap, (loc, case, start, art, out, snap)


def four_sensor_window(stem=FOUR_STEM):
    z = npload(S2W / "test_timeseries_dump.npz", allow_pickle=True)
    loc, case, nin, start = (int(v) for v in z[stem + "_meta"])
    assert nin == 4, nin
    true, pred = z[stem + "_true"], z[stem + "_pred"]
    assert_four_window_clean(loc, case, start, true.shape[0])
    inc, frac = ALONG.contact_fraction(loc, case, start, true.shape[0], 21)
    interp = ALONG.resample(ALONG.resample(true, 4), 21)   # true tension, 4 stations, back to 21
    nodes = ALONG.pick_nodes(frac)
    stations = [nodes["grounded"], nodes["transition"], nodes["submerged"]]
    assert not nodes["transition_fallback"], (stem, frac)          # a station really lifts off
    assert all(i is not None and i not in (0, 20) for i in stations), (stem, stations)
    h0, hs, tp = ALONG.sea_state(loc, case)
    return dict(loc=loc, case=case, start=start, true=true, pred=pred, interp=interp,
                inc=inc, frac=frac, stations=stations, td=[nodes["transition"]],
                hs=hs, tp=tp, h0=h0)


def fig_four_sensor(png_dir):
    ws = [four_sensor_window(s) for s in FOUR_WINDOWS]
    fig, axes = plt.subplots(len(ws), 3, figsize=(FULL_W, 4.0), squeeze=False)
    for r, w in enumerate(ws):
        t = np.arange(w["true"].shape[0]) * DT
        for c, i in enumerate(w["stations"]):
            ax = axes[r][c]
            tr, pr = w["true"][:, i] / 1e3, w["pred"][:, i] / 1e3
            shade_contact(ax, w["inc"][:, i], t)
            ax.plot(t, tr, color=C_TRUE, lw=0.9, zorder=3)
            ax.plot(t, pr, color=C_BLUE, lw=0.9, ls=(0, (3.2, 1.6)), zorder=4)
            lo, hi = min(tr.min(), pr.min()), max(tr.max(), pr.max())
            pad = max(hi - lo, 1e-6)
            ax.set_ylim(lo - 0.05 * pad, hi + 0.30 * pad)
            ax.set_xlim(t[0], t[-1])
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
            where = ("on seabed throughout" if w["frac"][i] >= 0.98 else "suspended"
                     if w["frac"][i] <= 0.02 else f"on seabed {100 * w['frac'][i]:.0f}% of window")
            ax.text(0.015, 0.97, f"station {i}, {where}\n"
                    f"$R^2$ = {r2(tr, pr):.3f}, MAE = {1e3 * np.abs(tr - pr).mean():.0f} N",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.5, zorder=7)
            if c == 0:
                ax.set_ylabel("Tension [kN]")
            if r == len(ws) - 1:
                ax.set_xlabel("Time [s]")
            else:
                ax.set_xticklabels([])
        axes[r][0].annotate(
            f"({'ab'[r]})  loc{w['loc']:02d},  $H_s$ = {w['hs']:.1f} m,  $T_p$ = {w['tp']:.1f} s",
            xy=(0.0, 1.03), xycoords="axes fraction", ha="left", va="bottom", fontsize=7.5)
        print(f"      loc{w['loc']:02d}/{w['case']:04d} start {w['start']} Hs {w['hs']:.2f} m "
              f"Tp {w['tp']:.1f} s, stations {w['stations']}")
    assert len({w["loc"] for w in ws}) == len(ws), [w["loc"] for w in ws]   # no site twice
    for c, title in enumerate(("Grounded", "Touchdown", "Suspended")):
        axes[0][c].set_title(title, pad=16, fontsize=8, fontfamily="STIXGeneral")
    fig.legend(handles=[Line2D([], [], color=C_TRUE, lw=1.1, label="Finite-element truth"),
                        Line2D([], [], color=C_BLUE, lw=1.1, ls=(0, (3.2, 1.6)),
                               label="Stage 2, four sensors"),
                        Patch(fc=C_BED, ec="none", label="Station on the seabed")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.135,
                        hspace=0.30, wspace=0.18)
    save(fig, "fig_foursensor", png_dir)


# ------------------------------------------------------------------ Figure 10 (envelope)
def fig_envelope(png_dir):
    """Share of snap events covered against the multiplier applied to the prediction,
    on the events common to the Stage-1 and Stage-2 test draws (per distinct event)."""
    import paperA_numbers as PN
    ev1 = PN.snap_events(npload(T1 / "test_timeseries_dump.npz"))
    z2 = npload(T2 / "test_timeseries_dump.npz")
    ev21, ev4 = PN.snap_events(z2, 21), PN.snap_events(z2, 4)
    common = sorted(set(ev1) & set(ev21) & set(ev4))
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    for ev, lab, col, ls in ((ev1, "Stage 1, matched", C_AQUA, "-"),
                             (ev21, "Stage 2, 21 sensors", C_BLUE, (0, (3.2, 1.6))),
                             (ev4, "Stage 2, four sensors", C_ORANGE, (0, (5, 1.5, 1, 1.5)))):
        m = np.sort([1.0 / ev[k][2] for k in common])
        cov = 100.0 * np.arange(1, len(m) + 1) / len(m)
        ax.step(np.r_[0.5, m], np.r_[0.0, cov], where="post", color=col, lw=1.1, ls=ls,
                label=lab)
    for q in (50, 75, 90):
        ax.axhline(q, color=C_MUTED, lw=0.5, zorder=0)
    ax.set_xlim(0.8, 2.4); ax.set_ylim(0, 101)
    ax.set_yticks([0, 25, 50, 75, 90, 100])
    ax.set_xlabel("Multiplier on the predicted line maximum")
    ax.set_ylabel("Snap events covered [%]")
    ax.legend(loc="lower right", frameon=False, handlelength=2.8)
    save(fig, "fig_envelope", png_dir)
    print(f"      {len(common)} common events")


# ------------------------------------------------------------------ Figure 11
# Motion-measurement noise, absolute levels (the eval-only job 236311). Colour encodes
# the ordered noise level on a one-hue lightness ramp, grey for no noise; line style
# encodes the model (Stage 2 solid, interpolated Stage 1 dashed), and each level has
# its own marker in (b), so nothing relies on colour alone.
SN = T2          # the noise levels, re-run on the clean Stage-2 draw by the EVAL_CLEAN job
NOISE_STYLE = (("", 0.0, C_MUTED, "o"), ("_p010", 0.01, "#86b6ea", "s"),
               ("_p050", 0.05, "#4a8fdc", "^"), ("_p100", 0.10, "#2566b8", "D"),
               ("_p250", 0.25, "#0f2f5e", "v"))


def _noise_curves(tag):
    with P.open_(SN / f"test_metrics_by_node_count{tag}.csv", newline="", encoding="utf-8") as fh:
        d = {int(r["node_count"]): float(r["MAE_tension"]) for r in csv.DictReader(fh)}
    with P.open_(SN / f"superres_grid_baseline{tag}.csv", newline="", encoding="utf-8") as fh:
        b = {int(r["N_in"]): float(r["MAE_tension"]) for r in csv.DictReader(fh) if int(r["N_out"]) == 21}
    return np.array([d[n] for n in LAYOUTS]), np.array([b[n] for n in LAYOUTS])


def fig_noise(png_dir):
    """(a) MAE at the 21 native stations against N_in, no noise and two noise levels;
    (b) the Stage-2 / interpolated MAE ratio for every absolute level."""
    x = np.array(LAYOUTS)
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 4.0), sharex=True)
    for tag, s, col, mk in NOISE_STYLE:
        d, bl = _noise_curves(tag)
        lab = "no noise" if s == 0 else f"$\\sigma$ = {s:.2f} m"
        if tag in ("", "_p050", "_p250"):
            a.plot(x, bl, color=col, lw=1.0, ls=(0, (3.2, 1.6)), marker=mk, ms=3.0, mfc="white")
            a.plot(x, d, color=col, lw=1.0, marker=mk, ms=3.0)
        b.plot(x, d / bl, color=col, lw=1.0, marker=mk, ms=3.2, label=lab)
    b.axhline(1.0, color=C_TRUE, lw=0.7, ls=(0, (1, 1.5)))
    a.set_yscale("log"); b.set_yscale("log")
    a.set_ylim(28, 1150)
    a.set_yticks([30, 50, 100, 200, 500])
    b.set_yticks([0.2, 0.3, 0.5, 1.0, 2.0])
    for ax in (a, b):
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_ylabel("MAE [N]")
    b.set_ylabel("MAE ratio, Stage 2 / interpolated")
    keys = [Line2D([], [], color=C_TRUE, lw=1.0, label="Stage 2"),
            Line2D([], [], color=C_TRUE, lw=1.0, ls=(0, (3.2, 1.6)), label="Interpolated Stage 1")]
    a.legend(handles=keys, loc="upper center", ncol=2, frameon=False, handlelength=2.6)
    b.legend(loc="lower right", frameon=False, ncol=2, handlelength=2.0, columnspacing=1.0)
    b.set_xticks(LAYOUTS)
    b.set_xlabel("Sensed stations $N_{\\mathrm{in}}$")
    for ax, letter in ((a, "(a)"), (b, "(b)")):
        ax.text(-0.2, 1.0, letter, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    fig.subplots_adjust(hspace=0.12)
    save(fig, "fig_noise", png_dir)


# ------------------------------------------------------------------ Appendix grid
def fig_grid(png_dir):
    """N_in x N_out MAE surfaces: Stage 2, interpolated Stage 1, and their ratio."""
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    g, gb = _grid("superres_grid.csv"), _grid("superres_grid_baseline.csv")
    n = len(LAYOUTS)
    D = np.array([[float(g[(a, b)]["MAE_tension"]) for b in LAYOUTS] for a in LAYOUTS])
    B = np.array([[float(gb[(a, b)]["MAE_tension"]) for b in LAYOUTS] for a in LAYOUTS])
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.0))
    norm = LogNorm(vmin=min(D.min(), B.min()), vmax=max(D.max(), B.max()))
    ongrid = [b for b in LAYOUTS if 20 % (b - 1) == 0]
    panels = ((D, "cividis", norm, "{:.0f}", "Stage 2, MAE [N]"),
              (B, "cividis", norm, "{:.0f}", "Interpolated Stage 1, MAE [N]"),
              (D / B, "PuOr_r", TwoSlopeNorm(vcenter=1.0, vmin=(D / B).min(),
                                             vmax=max(1.05, (D / B).max())),
               "{:.2f}", "Ratio, Stage 2 / interpolated"))
    for k, (ax, (M, cmap, nm, fmt, lab)) in enumerate(zip(axes, panels)):
        im = ax.imshow(M, origin="lower", cmap=cmap, norm=nm, aspect="equal")
        for i in range(n):
            for j in range(n):
                v = nm(M[i, j])
                dark = (v < 0.45) if cmap == "cividis" else (abs(v - 0.5) > 0.32)
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=4.3,
                        color="white" if dark else C_TRUE)
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{b}*" if b in ongrid else f"{b}" for b in LAYOUTS])
        ax.set_yticks(range(n)); ax.set_yticklabels(LAYOUTS)
        ax.set_xlabel("Output stations $N_{\\mathrm{out}}$")
        if k == 0:
            ax.set_ylabel("Sensed stations $N_{\\mathrm{in}}$")
        ax.grid(False)
        ax.tick_params(labelsize=6)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.2,
                          aspect=28)
        if cmap == "cividis":
            cb.set_ticks([35, 50, 70, 100, 150, 200])
            cb.set_ticklabels(["35", "50", "70", "100", "150", "200"])
            cb.minorticks_off()
        cb.ax.tick_params(labelsize=6)
        cb.set_label(lab, fontsize=6.5)
        ax.text(-0.02, 1.02, f"({'abc'[k]})", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=8)
    fig.subplots_adjust(wspace=0.22)
    save(fig, "fig_grid", png_dir)


# ------------------------------------------------------------------ Appendix E (linear regression)
# Stage 1 against LR4, the longest-memory linear model of Linear_Regression.ipynb (cluster job
# 885495), in one Stage-1 test window at the shallow withheld site loc10 (user, 2026-10-02: one
# row only; further windows go to the data repository). The window is chosen on the ground truth
# alone: among the stored reservoir windows (ty, sn: unbiased draws) of the screened Stage-1 dump at
# loc10 with at least ten stations, from a simulation no artifact rule flags, the one with the
# largest standard deviation of the true fairlead tension (the most dynamic sea state). LR3 is not
# drawn: its traces differ from LR4's by 11-52 N on average in such windows.
LR = ROOT / "Results" / "LR_baseline"
LINEAR_SITES = ((10, 1),)                       # (site, windows)
LINEAR_WINDOWS = [(10, 158, 10, 9884)]          # loc10/0158, typical window
C_LR = "#eb6834"


def linear_windows(z):
    """Re-derive the window choice from the rule above and check it against LINEAR_WINDOWS."""
    dirty = ALONG.flagged_cases()
    cand = {}
    for f in sorted(z.files):
        if not f.endswith("_meta") or f[:-5].split("_")[-1][:2] not in ("ty", "sn"):
            continue
        m = tuple(int(v) for v in z[f])
        if (m[2] >= 10 and m[0] in dict(LINEAR_SITES) and (m[0], m[1]) not in dirty
                and not eval_mask_hit(m[0], m[1], m[3], z[f[:-5] + "_true"].shape[0])):
            cand.setdefault(m, f[:-5])
    chosen = []
    for site, k in LINEAR_SITES:
        pool = sorted((m for m in cand if m[0] == site),
                      key=lambda m: -float(z[cand[m] + "_true"][:, -1].std()))
        sims = []
        for m in pool:
            if m[1] not in sims:
                sims.append(m[1])
                chosen.append(m)
            if len(sims) == k:
                break
    assert chosen == LINEAR_WINDOWS, chosen
    return [(m, cand[m]) for m in chosen]


def fig_linear(png_dir):
    import csv as _csv
    import linear_regression_lib as L       # torch; only this figure needs it
    z = npload(T1 / "test_timeseries_dump.npz")
    wins = linear_windows(z)
    w = npload(LR / "weights.npz")
    s3, s4 = npload(LR / "standardiser.npz"), npload(LR / "standardiser_lr4.npz")
    pred = L.predict_windows(str(ALONG.CACHE_DIR), [m for m, _ in wins],
                             {"LR4_lagged40": w["LR4_lagged40"]}, s3["mu"], s3["sd"],
                             lags=L.LAGS_LONG, stand={"LR4_lagged40": (s4["mu"], s4["sd"])})
    with P.open_(LR / "LR4_lagged40" / "test_per_window_stats.csv", newline="", encoding="utf-8") as fh:
        job_mae = {(int(r["lc_id"]), int(r["case_id"]), int(r["n_in"]), int(r["start_idx"])):
                   float(r["mae"]) for r in _csv.DictReader(fh)}
    fig, axes = plt.subplots(len(wins), 3, figsize=(FULL_W, 2.45), squeeze=False)
    for r, (m, stem) in enumerate(wins):
        loc, case, N, start = m
        true, s1, lr = z[stem + "_true"], z[stem + "_pred"], pred[m]["LR4_lagged40"]
        # the recomputed traces are the job's: same truth, same per-window MAE
        assert np.array_equal(pred[m]["true"], true), m
        assert abs(float(np.abs(lr - true).mean()) - job_mae[m]) <= 1e-5 * job_mae[m], m
        inc, frac = ALONG.contact_fraction(loc, case, start, true.shape[0], N)
        nodes = ALONG.pick_nodes(frac)
        h0, hs, tp = ALONG.sea_state(loc, case)
        t = np.arange(true.shape[0]) * DT
        for c, i in enumerate((nodes["grounded"], nodes["transition"], N - 1)):
            ax = axes[r][c]
            tr, p1, p4 = true[:, i] / 1e3, s1[:, i] / 1e3, lr[:, i] / 1e3
            shade_contact(ax, inc[:, i], t)
            ax.plot(t, tr, color=C_TRUE, lw=0.9, zorder=3)
            ax.plot(t, p4, color=C_LR, lw=0.8, ls=(0, (4.0, 1.2, 1.0, 1.2)), zorder=4)
            ax.plot(t, p1, color=C_BLUE, lw=0.9, ls=(0, (3.2, 1.6)), zorder=5)
            lo = min(tr.min(), p1.min(), p4.min())
            hi = max(tr.max(), p1.max(), p4.max())
            pad = max(hi - lo, 1e-6)
            ax.set_ylim(lo - 0.05 * pad, hi + 0.42 * pad)
            ax.set_xlim(t[0], t[-1])
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
            where = ("on seabed throughout" if frac[i] >= 0.98 else "suspended"
                     if frac[i] <= 0.02 else f"on seabed {100 * frac[i]:.0f}% of window")
            ax.text(0.015, 0.97, f"station {i}, {where}\n"
                    f"MAE: Stage 1 {1e3 * np.abs(p1 - tr).mean():.0f} N, "
                    f"LR4 {1e3 * np.abs(p4 - tr).mean():.0f} N",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.5, zorder=7)
            if c == 0:
                ax.set_ylabel("Tension [kN]")
            if r == len(wins) - 1:
                ax.set_xlabel("Time [s]")
            else:
                ax.set_xticklabels([])
        role = "shallow training site" if loc == 6 else "withheld site"
        tag = f"({'abcde'[r]})  " if len(wins) > 1 else ""
        axes[r][0].annotate(
            f"{tag}loc{loc:02d}, {role},  $N$ = {N},  $H_s$ = {hs:.1f} m,  $T_p$ = {tp:.1f} s",
            xy=(0.0, 1.03), xycoords="axes fraction", ha="left", va="bottom", fontsize=7.5)
        print(f"[linear] loc{loc:02d}/{case:04d} N={N} start {start}: stations "
              f"{nodes['grounded']}/{nodes['transition']}/{N - 1}, touchdown station on the seabed "
              f"{100 * frac[nodes['transition']]:.0f} %")
    for c, title in enumerate(("Grounded", "Near touchdown", "Fairlead")):
        axes[0][c].set_title(title, pad=16, fontsize=8, fontfamily="STIXGeneral")
    fig.legend(handles=[Line2D([], [], color=C_TRUE, lw=1.1, label="Finite-element truth"),
                        Line2D([], [], color=C_BLUE, lw=1.1, ls=(0, (3.2, 1.6)), label="Stage 1"),
                        Line2D([], [], color=C_LR, lw=1.1, ls=(0, (4.0, 1.2, 1.0, 1.2)),
                               label="LR4 (linear, lags to $\\pm$40 s)"),
                        Patch(fc=C_BED, ec="none", label="Station on the seabed")],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.80, bottom=0.265, hspace=0.42, wspace=0.18)
    save(fig, "fig_linear", png_dir)


FIGURES = {"alongline": fig_along_line, "peaks": fig_peaks, "snaptraces": fig_snap_traces,
           "snaptraces_s2": fig_snap_traces_s2,
           "sweep": fig_sweep, "foursensor": fig_four_sensor,
           "noise": fig_noise, "grid": fig_grid, "linear": fig_linear}
# Figures 4, 8 and D.1 take the true seabed contact (and Figure D.1 the linear models' inputs)
# from the raw simulations.
RAW_FIGURES = {"alongline", "foursensor", "linear"}
# fig_envelope (the old Fig. 10, coverage of the line maximum) is no longer drawn: under the
# fairlead condition the line maximum at every snap event is the fairlead's (user, 2026-09-24).

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None, help="also write PNG previews into this directory")
    ap.add_argument("--only", default=None, help="comma-separated subset of " + ",".join(FIGURES))
    args = ap.parse_args()
    png = pathlib.Path(args.png) if args.png else None
    for name, fn in FIGURES.items():
        if args.only and name not in args.only.split(","):
            continue
        if name in RAW_FIGURES and not P.CACHE.is_dir():
            print(f"[fig] skipped {name}: it reads the raw simulations; set MOORING_CACHE_DIR")
            continue
        fn(png)
