"""
pub_select_epoch.py -- post-hoc checkpoint selection for the Publication runs.

SELECTION v4 (user directive 2026-07-27). Supersedes the v3 two-rate box.

Why the rule changed
--------------------
The FINAL run (job 10466291, checkpoints_P1_matched) exposed the old rule as
unsatisfiable: it required 0 < bias_p90 <= 200 N AND 0 < bias_p99 <= 400 N, but
val_peak_tension_bias_p99 was NEGATIVE in all 55 epochs (-3785 .. -764 N). No
epoch was ever feasible, so save_feasible_checkpoints never fired and the only
checkpoint on disk was the fallback best-R2 file -- epoch 25, the WORST
tail-safety epoch of the run (up90 96.9%, up99 92.5%), which then went to test
(test up90 94.74%, up99 97.49%).

Root cause of the persistently negative bias: the p99 tail error distribution is
extremely skewed. ~90% of tail entries are over-predicted by a few hundred N,
while a handful are under-predicted by 10-20 kN, and those few dominate the
mean. Rate and magnitude are ANTI-correlated across epochs, so a bias gate is
effectively a gate on the rare catastrophic miss, not on typical conservatism.

The rule
--------
  Gate A (accuracy): val_global_R2_tension >= run_max - delta      [delta 0.0010]
  Gate B (safety)  : underpred_p90 < up90_cap                      [10 %]
                     -- p90 ONLY. The p99 cap was REMOVED in v4: on the corrected
                     pipeline the p99 tail is 43 % (fairlead peak) to 82 % (line
                     max) snap-composed, and snap magnitude is aliasing-bounded at
                     the 0.1 s export interval (rise=1 is 76 % of events, median
                     capture 0.592). up99 < 20 % is therefore not reachable by
                     construction, and gating on it produced ZERO feasible epochs
                     in jobs 10466291 and 10521852 -- both then shipped the
                     safety-blind 10-R2 fallback. p90 is only 10-15 % snap-composed
                     and is the tau=0.9 design exceedance, so it stays the gate.
  Objective        : among epochs passing A AND B, pick the MINIMUM
                     val_temporal_diff_skill_tension (lower = better tracking).

  Fallback (SAFETY-INFEASIBLE): no Gate-A epoch meets the p90 cap -> report the
  epoch with the lowest up90 as least-unsafe, and flag the run. On the cluster
  the training loop now ABORTS at cfg.sel_abort_epoch instead of reaching this.

bias_p90/p99 and undermag_p90/p99 are still REPORTED for every pick -- they are
diagnostics, not gates. undermag = mean |error| over the tail entries that
actually miss low (added by patch 10); watch it, because the epoch with the best
underprediction RATE often has the worst conditional miss MAGNITUDE.

Usage:
  python pub_select_epoch.py [dir ...] [--delta 0.001]
         [--up90-cap 10] [--require-checkpoint]
         [--stale-hours 6]
"""
import sys, csv, os, math, argparse

sys.stdout.reconfigure(encoding="utf-8")

FIELDS = {
    "r2":    "val_global_R2_tension",
    "skill": "val_temporal_diff_skill_tension",
    "up90":  "val_peak_tension_underpred_rate_p90",
    "b90":   "val_peak_tension_bias_p90",
    "up99":  "val_peak_tension_underpred_rate_p99",
    "b99":   "val_peak_tension_bias_p99",
    "mae90": "val_peak_tension_MAE_p90",
    "mae99": "val_peak_tension_MAE_p99",
    "thr90": "val_peak_tension_thr_p90",
    "thr99": "val_peak_tension_thr_p99",
    "um90":  "val_peak_tension_undermag_p90",   # patch 10; absent in older runs
    "um99":  "val_peak_tension_undermag_p99",
}


def load_history(d):
    rows = []
    with open(os.path.join(d, "metrics_history.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            row = {"epoch": int(float(r["epoch"]))}
            for k, c in FIELDS.items():
                try:
                    row[k] = float(r.get(c, ""))
                except (TypeError, ValueError):
                    row[k] = float("nan")
            rows.append(row)
    return rows


def fresh_checkpoint_epochs(d, rows, stale_hours):
    """Epochs with a non-stale .pt in dir (feasible/periodic + the best file)."""
    csv_m = os.path.getmtime(os.path.join(d, "metrics_history.csv"))
    ok = {}
    for fn in os.listdir(d):
        if fn.startswith("checkpoint_epoch") and fn.endswith(".pt"):
            ep = int(fn[len("checkpoint_epoch"):-3])
        elif fn.startswith("best_") and fn.endswith(".pt"):
            ep = max(rows, key=lambda r: r["r2"])["epoch"]
        else:
            continue
        fresh = (csv_m - os.path.getmtime(os.path.join(d, fn))) <= stale_hours * 3600
        ok[ep] = ok.get(ep, False) or fresh
    return {e for e, f in ok.items() if f}


def select(rows, delta, up90_cap, allowed=None):
    mx = max(r["r2"] for r in rows if not math.isnan(r["r2"]))
    A = [r for r in rows if r["r2"] >= mx - delta]
    if allowed is not None:
        A = [r for r in A if r["epoch"] in allowed]

    def safe(r):
        # v4: p90 rate only. up99 is reported everywhere but never gates.
        return (not math.isnan(r["up90"]) and r["up90"] < up90_cap)

    B = [r for r in A if safe(r)]
    if B:
        return "OK", min(B, key=lambda r: r["skill"]), B, A, mx
    least = min(A, key=lambda r: r["up90"]) if A else None
    return "SAFETY-INFEASIBLE", least, [], A, mx


def fmt(r):
    um = ""
    if not math.isnan(r.get("um99", float("nan"))):
        um = f"  undermag90={r['um90']:.0f} undermag99={r['um99']:.0f}"
    return (f"ep{r['epoch']:>2}  R2={r['r2']:.4f}  skill={r['skill']:.4f}  "
            f"up90={r['up90']:4.1f}%  up99={r['up99']:4.1f}%  "
            f"[diag b90={r['b90']:+6.1f}  b99={r['b99']:+7.1f}  "
            f"mae90={r['mae90']:.0f}  mae99={r['mae99']:.0f}{um}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=["Results/checkpoints_P1_matched_v2"])
    ap.add_argument("--delta", type=float, default=0.0010)
    ap.add_argument("--up90-cap", type=float, default=10.0)   # selection v4
    ap.add_argument("--require-checkpoint", action="store_true")
    ap.add_argument("--stale-hours", type=float, default=6.0)
    args = ap.parse_args()

    print(f"rule v4: R2 >= max-{args.delta} | up90 < {args.up90_cap:.0f}% "
          f"(p90 ONLY) | min skill  (up99/bias/undermag = diagnostics)"
          + ("  [restricted to fresh on-disk checkpoints]" if args.require_checkpoint else ""))
    print("=" * 108)
    ranking = []
    for d in args.dirs:
        name = os.path.basename(d.rstrip("/\\")).replace("checkpoints_P1_HPO_", "")
        rows = load_history(d)
        allowed = fresh_checkpoint_epochs(d, rows, args.stale_hours) if args.require_checkpoint else None
        level, pick, feas, A, mx = select(rows, args.delta, args.up90_cap,
                                          allowed)
        r0 = rows[0]
        old = max(rows, key=lambda r: r["r2"])   # the max-R2 rule, for contrast
        print(f"\n### {name}  (epochs={len(rows)}, maxR2={mx:.4f}, "
              f"thr_p90={r0['thr90']:.0f} N, thr_p99={r0['thr99']:.0f} N)")
        print(f"  gateA epochs: {len(A)}   feasible: {len(feas)} "
              f"{sorted(r['epoch'] for r in feas)}   level: {level}")
        if pick is not None:
            ck = "n/a"
            if allowed is None:
                have = fresh_checkpoint_epochs(d, rows, args.stale_hours)
                ck = "on disk" if pick["epoch"] in have else "MISSING"
            print(f"  PICK [{level}]: {fmt(pick)}   ckpt: {ck}")
        print(f"  (max-R2 rule would pick: {fmt(old)})")

        # Full menu of passing epochs. The rule ranks on skill, but undermag
        # (mean |error| over tail entries that MISS LOW) is anti-correlated with
        # the underprediction rate -- the best-rate epoch often has the worst
        # catastrophic miss -- so the choice among feasible epochs is a real
        # judgement call and every candidate is listed here.
        if feas:
            on_disk = fresh_checkpoint_epochs(d, rows, args.stale_hours)
            print(f"\n  ALL {len(feas)} FEASIBLE EPOCHS (ranked by skill; "
                  f"'*' = the rule's pick):")
            print("    ep  |   R2    |  skill  | up90  | up99  |  b90   |   b99   "
                  "| undermag90 | undermag99 | ckpt")
            print("    " + "-" * 100)
            for r in sorted(feas, key=lambda r: r["skill"]):
                star = "*" if r["epoch"] == pick["epoch"] else " "
                um90 = "-" if math.isnan(r["um90"]) else f"{r['um90']:10.0f}"
                um99 = "-" if math.isnan(r["um99"]) else f"{r['um99']:10.0f}"
                disk = "on disk" if r["epoch"] in on_disk else "-"
                print(f"  {star} {r['epoch']:>3} | {r['r2']:.5f} | {r['skill']:.4f}  "
                      f"| {r['up90']:5.2f}%| {r['up99']:5.2f}%| {r['b90']:+7.1f}"
                      f"| {r['b99']:+8.1f}| {um90} | {um99} | {disk}")
            if all(math.isnan(r["um90"]) for r in feas):
                print("    (undermag columns are blank: this run predates patch 10)")
            print("\n  To inspect any of these locally:  python plot_test_timeseries.py "
                  f"{d} --epoch <N>")
        if level == "SAFETY-INFEASIBLE":
            print("  [WARN] no epoch met the p90 rate cap -- the shipped model "
                  "is NOT tail-safe; loosen the cap or retrain.")
        ranking.append((name, level, pick))

    order = {"OK": 0, "SAFETY-INFEASIBLE": 1}
    ranking.sort(key=lambda t: (order[t[1]], t[2]["skill"] if t[2] else 9e9))
    print("\n" + "=" * 108)
    print("RANKING (level, then skill at pick):")
    for i, (name, level, pick) in enumerate(ranking, 1):
        s = f"{pick['skill']:.4f} @ ep{pick['epoch']}" if pick else "-"
        print(f"  {i}. {name:28s} [{level}]  skill {s}")


if __name__ == "__main__":
    main()
