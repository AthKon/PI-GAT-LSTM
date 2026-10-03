"""Lay out a working directory for Linear_Regression.ipynb from this repository and the archives.

    python make_lr_workdir.py --data /path/to/DATA --out lr_run
    cd lr_run
    export MOORING_CACHE_DIR=/path/to/cache_npy
    LR_RUN_MODE=smoke jupyter nbconvert --to notebook --execute Linear_Regression.ipynb \
        --output Linear_Regression_executed.ipynb --ExecutePreprocessor.timeout=-1

--data is the folder holding the unpacked 4TU archives (stage1_matched/, linear_baseline/).
The layout is the one the cluster job ran in (job 885495, run_LR_baseline.slurm): the notebook
and its two modules, the three derived tables, and under Results/ the Stage 1 configuration,
test dumps and per-window files it compares itself against. Results/LR_ref_872463/ holds the
LR0-LR3 metrics of the first linear-regression job, which the full run must reproduce (GATE 8);
they are byte-identical to the deposited LR0-LR3 metrics, which is where they are taken from.
The notebook then writes Results/LR_baseline/ (Results/LR_baseline/_smoke/ in smoke mode).
"""
import argparse
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
S1 = "stage1_matched"
SCR, UNS = "evaluation_screened_(published)", "evaluation_unscreened"
EC1 = "Results/SR_RESULTS/checkpoints_P2_eval_clean/stage1"

CODE = ["Linear_Regression.ipynb", "linear_regression_lib.py", "stage1_verbatim.py"]
TABLES = ["artifact_samples.csv", "artifact_samples_cat.csv", "snap_catalogue_cat.csv"]
FROM_ARCHIVES = [   # (path in the working directory, path under --data)
    ("Results/checkpoints_P1_matched_v5/config.json", f"{S1}/config.json"),
    ("Results/checkpoints_P1_matched_v5/test_timeseries_dump.npz",
     f"{S1}/{UNS}/test_timeseries_dump_training_job.npz"),
    ("Results/eval_v5_snapfix/test_timeseries_dump.npz", f"{S1}/{UNS}/test_timeseries_dump.npz"),
    (f"{EC1}/test_metrics.csv", f"{S1}/{SCR}/test_metrics.csv"),
    (f"{EC1}/test_metrics_by_node_count.csv", f"{S1}/{SCR}/test_metrics_by_node_count.csv"),
    (f"{EC1}/test_per_window_stats.csv", f"{S1}/{SCR}/test_per_window_stats.csv"),
    (f"{EC1}/test_timeseries_dump.npz", f"{S1}/{SCR}/test_timeseries_dump.npz"),
    (f"{EC1}/published/test_per_window_stats.csv", f"{S1}/{UNS}/test_per_window_stats.csv"),
] + [(f"Results/LR_ref_872463/{m}/{sub}test_metrics.csv",
      f"linear_baseline/{m}/{UNS if sub else SCR}/test_metrics.csv")
     for m in ("LR0_static", "LR1_pointwise", "LR2_line", "LR3_lagged") for sub in ("", "published/")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="folder holding the unpacked archives")
    ap.add_argument("--out", required=True, help="working directory to create")
    a = ap.parse_args()
    data, out = pathlib.Path(a.data), pathlib.Path(a.out)
    pairs = [(f, HERE / f) for f in CODE] + [(f, HERE / "data" / f) for f in TABLES] + \
            [(d, data / s) for d, s in FROM_ARCHIVES]
    missing = [str(s) for _d, s in pairs if not s.is_file()]
    if missing:
        raise SystemExit("missing inputs:\n  " + "\n  ".join(missing))
    for dst, src in pairs:
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out / dst)
    print(f"[lr-workdir] {len(pairs)} files in {out}")


if __name__ == "__main__":
    main()
