# PI-GAT-LSTM: mooring line tension reconstruction from motion sensing

Code for the manuscript *Spatial Super-Resolution of Mooring Tension using a Topology-General,
Physics-Informed Graph Attention Network* by Athanasios Konstantaras, Juan P. Aguilar-López and
Oriol Colomés (Delft University of Technology), in preparation for submission to *Ocean Engineering*.

| | |
|---|---|
| Trained models, input normalisers, logs and evaluation outputs | 4TU.ResearchData, https://doi.org/10.4121/e3f167ac-2925-4815-956e-5ef3d883f1a5 (CC BY 4.0) |
| Code (this repository) | MIT licence |
| Code version used for the deposited files | release **v1.0** of this repository |
| Contact | Athanasios Konstantaras, thanoskonstantaras@gmail.com |

The repository holds the notebooks that trained and evaluated the models, the scripts that turn
the evaluation outputs into every number, table and figure of the manuscript, and the small
derived tables they read. The trained checkpoints and the evaluation outputs are in the 4TU
dataset; its own README (deposited next to the archives) documents every file in it. This README
explains the code and how the two fit together.

---

## 1. What the model does

A mooring line is a chain graph of stations from the anchor to the fairlead. A GATv2 spatial
encoder couples neighbouring stations, a bidirectional LSTM follows each station in time, and a
shared head returns the **axial tension along the line** over a 100 s window (1000 samples at
0.1 s). The inputs are the **motion** at the sensed stations: horizontal and vertical position
and the channels derived from them (velocities, seabed contact, drag), plus the static line
description and the sea state. The dynamic tension is never an input.

* **Stage 1, matched resolution**: tension at the same N stations at which motion is given,
  N = 4, 5, 6, 7, 8, 10, 12, 15, 18 or 21, with one set of weights for all ten layouts.
* **Stage 2, spatial super-resolution**: a query-based cross-attention decoder between the
  encoder and the LSTM returns the tension at the 21 native stations from fewer sensed stations
  (for example 4 in, 21 out). It is warm-started from Stage 1.

Training and test data are finite-element simulations at ten offshore sites in the southern
North Sea (water depth 14 to 45 m, line length 75 or 105 m, 300 sea states per site). Two sites,
loc10 and loc11, are never seen in training.

The loss has eight terms, weighted by learned (Kendall) uncertainties: mean squared error, mean
absolute error, temporal gradient, spectral amplitude, incremental EA·strain (constitutive),
catenary equilibrium, nodal momentum balance, and a τ = 0.9 pinball term on the high-tension band.

---

## 2. Code and dataset: what lives where

| | Where | Section |
|---|---|---|
| Model definitions, training and evaluation notebooks | this repository | 3 |
| Scripts for the manuscript's numbers, tables and figures | this repository | 6 |
| Derived tables (artifact masks, snap catalogues, case scans) | this repository, `data/` | 6, 9 |
| Trained checkpoints (Stage 1 epoch 32, Stage 2 epoch 76) | 4TU, `stage1_matched.zip`, `stage2_superres.zip` | 5 |
| Input normalisers, configurations, training logs | 4TU, the same two archives | 5 |
| Evaluation outputs (metrics, per-window statistics, stored predictions, layout grid, noise sweep) | 4TU, the same two archives | 6, 7 |
| Linear-regression baselines (weights, standardisers, evaluation outputs) | 4TU, `linear_baseline.zip` | 8 |
| Raw finite-element simulations (about 13 GB preprocessed) | not deposited, available on request | 9 |

Each archive unpacks into one folder. Unpack all three side by side in one folder, called `DATA`
below:

```
DATA/stage1_matched/    checkpoint, input_normalisers.npz/.json, config.json, training/,
                        evaluation_screened_(published)/, evaluation_unscreened/
DATA/stage2_superres/   the same, plus provenance/ (executed notebooks and job logs)
DATA/linear_baseline/   weights.npz, standardiser*.npz, LR0_static/ ... LR4_lagged40/, provenance/
```

`evaluation_screened_(published)` is the test draw the manuscript reports; `evaluation_unscreened`
is the same model on the test draw as originally made (dataset README, Section 6.4).

---

## 3. Repository contents

**Notebooks**

| File | Role |
|---|---|
| `GAT_LSTM_NN_Publication_superresolution.ipynb` | Model definitions for **both stages** (a Stage 1 checkpoint has `use_query_decoder = false`), Stage 2 training, and every evaluation. One run cell selects the mode (`_MODE`): `FINAL_WARM` trained Stage 2; `EVAL_CLEAN`, the active mode, wrote both `evaluation_screened_(published)/` folders and the noise sweep; `EVAL_SNAPROWS` wrote the withheld-site windows; `BASELINE`, `HPO_*`, `FINAL_SCRATCH` and `EVAL_NOISE` are the other runs of the study. |
| `GAT_LSTM_NN_Publication.ipynb` | Stage 1 training (the run that produced the epoch-32 checkpoint). |
| `GAT_LSTM_NN_Publication_EVAL.ipynb` | Stage 1 evaluation with the corrected snap catalogue; it wrote `stage1_matched/evaluation_unscreened/test_timeseries_dump.npz`. |
| `Linear_Regression.ipynb` | The five linear-regression baselines LR0 to LR4 on Stage 1's split and test windows (Section 8). |

The code cells of `GAT_LSTM_NN_Publication_superresolution.ipynb` and `Linear_Regression.ipynb`
are identical to those of the executed notebooks deposited in the archives' `provenance/` folders
(the evaluation job 402698 and the linear-regression job 885495). Stage 2 was trained with an
earlier state of the same notebook, before the evaluation-only modes were added; its executed copy
is `stage2_superres/provenance/GAT_LSTM_NN_Publication_SR_FINAL_WARM_executed_154471.ipynb`. The
Stage 1 notebook differs from the one that trained the epoch-32 checkpoint only in the snap
catalogue it embeds and in how its test dump selects stored windows; the data pipeline, model,
loss, training loop and evaluation cells are unchanged.

**Using the checkpoints**

| File | Role |
|---|---|
| `load_checkpoint_example.py` | Builds the model from a checkpoint's configuration, loads the weights and runs it once (Section 5). |
| `export_input_normalisers.py` | Rebuilds the five input normalisers of a checkpoint from the training data, as deposited in `input_normalisers.npz` (Section 5.2). |

**Manuscript numbers, tables and figures** (Section 6)

| File | Role |
|---|---|
| `paperA_numbers.py` | Writes every number quoted in the manuscript (`numbers.tex`, one LaTeX macro each) and every table. |
| `paperA_figures.py` | Writes every figure. |
| `paperA_paths.py` | Where both find their inputs: the archives (`PAPERA_DATA`), `data/` and the raw simulations (`MOORING_CACHE_DIR`). |
| `exact_split_accounting.py` | Snap events reachable by the training, validation and test windows under each exclusion policy. |
| `pub_select_epoch.py` | Re-applies the checkpoint-selection rule to a training log. |
| `plot_test_timeseries.py`, `plot_node_timeseries.py`, `plot_tension_along_line.py` | Plots of stored test windows; `plot_tension_along_line.py` also supplies the seabed-contact helpers of the figures. |

**Building the derived tables from the raw simulations** (Section 9): `scan_corrupt_cases.py`,
`scan_artifact_samples.py`, `scan_snap_events.py`.

**Linear-regression baselines** (Section 8): `Linear_Regression.ipynb`, `linear_regression_lib.py`,
`stage1_verbatim.py` (Stage 1's data pipeline copied verbatim from the notebook; generated),
`_build_linear_regression_nb.py` and `_Linear_Regression_cells.py` (build the notebook and
`stage1_verbatim.py`), `make_lr_workdir.py`.

**`cluster/`**: the Slurm job scripts of the runs (DelftBlue, one NVIDIA A100 80 GB), kept as a
record of how each was run. `run_pub_FINAL_v5.slurm` trained Stage 1, `run_pub_EVAL_v5_snapfix.slurm`
evaluated it, `run_sr_FINAL_WARM.slurm` trained Stage 2, `run_sr_EVAL_clean.slurm` ran the
evaluation behind the reported results, `run_LR_baseline.slurm` fitted the linear baselines.
`run_sr_*.slurm` are generated by `_make_sr_slurm.py` (with `sr_ops_preflight.py`, the GPU
preflight they embed). They contain cluster paths and module names and are not meant to run
elsewhere unchanged.

**`data/`**: derived tables, about 1.7 MB.

| File | Content |
|---|---|
| `artifact_samples.csv` | the training mask: samples dropped as solver artifacts (1,643 samples in 109 simulations) |
| `artifact_samples_cat.csv` | the evaluation mask: the training mask plus the samples at which a station other than the fairlead carries more than 1 kN and more than the fairlead |
| `snap_catalogue_cat.csv`, `snap_by_case_cat.csv` | the 902 snap events the manuscript reports, and a per-simulation summary |
| `snap_catalogue.csv`, `snap_by_case.csv` | the earlier catalogue (1,388 events), read for the comparison in the manuscript's Section 2.5 |
| `corrupt_scan.csv`, `ceiling_derivation.csv`, `anchor_outpull_samples.csv` | per-simulation scans behind the curation rules |
| `derived_inputs.json`, `derived_inputs_arrays.npz` | stored results that need inputs not in the archives (Section 6) |

---

## 4. Installation

Python 3.11. The versions that produced the deposited results are pinned in `requirements.txt`:

```bash
git clone https://github.com/AthKon/PI-GAT-LSTM && cd PI-GAT-LSTM
pip install -r requirements.txt
```

Loading a checkpoint needs only `torch`, `torch_geometric` and `numpy`. It was tested with
Python 3.11.9, PyTorch 2.11.0 (CPU) and PyTorch Geometric 2.7.0. The manuscript scripts need only
NumPy and Matplotlib; they ran with NumPy 2.4.4 and Matplotlib 3.10.9.

| | Stage 2 training, every evaluation, the linear baselines | Stage 1 training |
|---|---|---|
| Python | 3.11.9 | 3.10.13 |
| PyTorch | 2.5.1, CUDA 12.5 | 2.1.0 |
| PyTorch Geometric | 2.7.0 | 2.x |
| NumPy, pandas, Matplotlib | 1.26.4, 2.2.3, 3.9.2 | |
| hardware | one NVIDIA A100 80 GB, 16 CPU cores | same |

On that cluster stack cuDNN rejected the LSTM (`CUDNN_STATUS_BAD_PARAM`), and the runs used
PyTorch's own LSTM kernels by setting `SR_DISABLE_CUDNN=1`; the results agree with cuDNN to
floating-point tolerance. Set the same variable if you meet that error.

---

## 5. Loading a trained model

Do not execute the notebooks top to bottom on a laptop: their data cells build tens of thousands
of datasets and are meant for a GPU node. `load_checkpoint_example.py` executes only the
notebook's 22 definition cells, which define classes and read no data, builds the model from the
checkpoint's configuration, loads the weights with `strict=True` and runs it once on random
inputs of the right shapes:

```bash
python load_checkpoint_example.py DATA/stage1_matched/best_mooring_gat_lstm_seed_42.pt
python load_checkpoint_example.py DATA/stage2_superres/best_mooring_gat_lstm_seed_42.pt
```

Expected output:

```
epoch 32  parameters 5,670,017  query decoder: False
output (100, 4, 1)
epoch 76  parameters 5,825,153  query decoder: True
output (100, 21, 1)
```

### 5.1 Inputs and outputs

For a window of W samples at N sensed stations, ordered from the anchor (0) to the fairlead (N-1):

| Argument | Shape | Content |
|---|---|---|
| `graph.x` | [N, 12] | static node features, standardised |
| `graph.x_raw` | [N, 12] | the same before standardising; column 0, the normalised arc length, is read by the Stage 2 decoder |
| `graph.edge_index` | [2, 2(N-1)] | both directions of every segment |
| `graph.edge_attr` | [2(N-1), 10] | static edge features, standardised |
| `x_seq` | [W, N, 22] | dynamic node features (motion-derived), standardised |
| `edge_seq` | [W, 2(N-1), 4] | dynamic edge features, standardised |
| `query_graph` | the same fields at 21 stations | Stage 2 only: the output stations |

`model(graph, x_seq, edge_seq, query_graph)` returns standardised tension of shape [W, N_out, 1],
with N_out = N for Stage 1 and 21 for Stage 2, over the same window. The channel lists and their
units are in Appendix A of the manuscript. The dataset class `MooringSequenceDatasetPositionTension`
of the notebook builds all of them from the x and z positions of a simulation and the static line
description.

### 5.2 Normalisation

Standardisation is part of the model. Each model archive holds `input_normalisers.npz` (and the
same values in `input_normalisers.json`): for each group `static_node`, `static_edge`,
`dynamic_node`, `dynamic_edge` and `target`, the channels it applies to (`<group>_feature_idx`),
their mean and their standard deviation. Channels not listed (binary flags) are left as they are.
Use the file from the same archive as the checkpoint.

```python
import numpy as np, torch
from load_checkpoint_example import load_model

model, ck = load_model("DATA/stage2_superres/best_mooring_gat_lstm_seed_42.pt")
nz = np.load("DATA/stage2_superres/input_normalisers.npz")

def standardise(x, group):                                 # last axis = channels
    idx = torch.as_tensor(nz[f"{group}_feature_idx"])
    x = x.clone()
    x[..., idx] = (x[..., idx] - torch.as_tensor(nz[f"{group}_mean"])) / torch.as_tensor(nz[f"{group}_std"])
    return x

graph.x_raw = graph.x                                      # raw static features (s/L in column 0)
graph.x = standardise(graph.x, "static_node")
graph.edge_attr = standardise(graph.edge_attr, "static_edge")
# Stage 2 only: the same three lines for query_graph, the graph of the 21 output stations
with torch.no_grad():
    y = model(graph, standardise(x_seq, "dynamic_node"), standardise(edge_seq, "dynamic_edge"), query_graph)
tension_N = y[..., 0] * float(nz["target_std"][0]) + float(nz["target_mean"][0])
```

The notebook refits these statistics on 5,000 training windows each time it builds its data
loaders, and the checkpoints store only the tension and position statistics. The deposited files
were therefore made afterwards by `export_input_normalisers.py`, which rebuilds the training split,
checks that the rebuilt test draw equals the deposited one window for window, runs the notebook's
own fitting function, and checks the tension and position statistics against the checkpoint. With
the raw simulations it reproduces the deposited file bit for bit:

```bash
export MOORING_CACHE_DIR=/path/to/cache_npy
python export_input_normalisers.py stage1 --zip /path/to/stage1_matched.zip --out normalisers_stage1
```

---

## 6. Regenerating the manuscript's numbers, tables and figures

With the three archives unpacked into `DATA`, from the repository root:

```bash
export PAPERA_DATA=/path/to/DATA
python paperA_numbers.py      # paperA_output/numbers.tex (561 macros) and paperA_output/tables/ (13 tables)
python paperA_figures.py      # paperA_output/figures/
```

The output folder can be changed with `PAPERA_OUT`. `paperA_paths.py` translates every input path
to the archive file holding the same content (each pair was checked by SHA-256). Run this way,
without the raw simulations, `numbers.tex` and all 13 tables come out byte-identical to the
manuscript's, in about 30 seconds.

**What comes from where.** Almost every result is computed from the archives and `data/`. A few
need inputs that are not deposited: the raw simulations (site depths and line lengths, sea-state
ranges, fairlead motion statistics, the share of tension samples above the pinball threshold, snap
reachability, the catenary-condition study of Section 2.5, the true seabed contact behind the
Figure 8 values of Section 4.2.2 and a true-contact check of the regional R²), the training logs of
the global-context ablation, two Stage 2 per-checkpoint test dumps (a comparison of epochs 33 and
76), and the file times behind the compute times of Section 3.6. These were computed in
the development tree and stored in `data/derived_inputs.json` and `data/derived_inputs_arrays.npz`
(`python paperA_numbers.py --write-derived`). With `PAPERA_DATA` set they are read from there;
three macro groups come from them entirely (the simulation campaign, 14 macros; the catenary
condition, 23; compute times, 6). With the raw simulations available, `PAPERA_RECOMPUTE=1`
recomputes the ones that read them.

**Figures.** Figures 5, 6, 7, 9 and 10 (`fig_peaks`, `fig_snaptraces`, `fig_sweep`,
`fig_snaptraces_s2`, `fig_noise`) are drawn from the archives and come out identical to the
manuscript's, as does `fig_grid` (the full 10 x 10 layout grid, not in the manuscript). Figures 4, 8 and D.1 (`fig_alongline`, `fig_foursensor`, `fig_linear`) read
the true seabed contact from the raw simulations; without `MOORING_CACHE_DIR` they are skipped with
a message.

Without `PAPERA_DATA` the scripts read the development layout the results were produced in
(`Results/...`, see `paperA_paths.py`) and write into the manuscript folder.

`pub_select_epoch.py` re-applies the checkpoint-selection rule to a training log. It picks epoch
32 for Stage 1 and, with the Stage 2 safety gate switched off as in that run, epoch 76 for Stage 2:

```bash
python pub_select_epoch.py DATA/stage1_matched/training
python pub_select_epoch.py --up90-cap 1000 DATA/stage2_superres/training
```

---

## 7. Re-running the evaluation and the training

Both need the raw simulations (Section 9) and a GPU. The evaluation behind the reported results
took about 11 h on one A100.

1. `export MOORING_CACHE_DIR=/path/to/cache_npy`.
2. In the working directory, put the checkpoints where the notebook expects them:
   `./checkpoints_P1_matched_v5/best_mooring_gat_lstm_seed_42.pt` (Stage 1) and
   `./checkpoints_P2_final_warm/best_mooring_gat_lstm_seed_42.pt` (Stage 2).
3. The active run-cell line of `GAT_LSTM_NN_Publication_superresolution.ipynb` is
   `_MODE = ("EVAL_CLEAN", "./checkpoints_P2_final_warm/best_mooring_gat_lstm_seed_42.pt")`.
4. Execute it:
   ```bash
   jupyter nbconvert --to notebook --execute GAT_LSTM_NN_Publication_superresolution.ipynb \
       --output evaluation_executed.ipynb --ExecutePreprocessor.timeout=-1
   ```

It rebuilds the splits, refits the normalisers, scores Stage 2 on its layout-paired draw and
Stage 1 on its own draw, both unscreened and screened, then runs the 10 x 10 layout grid, the
interpolation baseline and the ten noise levels, into `./checkpoints_P2_eval_clean/`. The executed
notebook prints the draw fingerprints and a `[check]` block comparing eight pooled metrics with the
shipped values; every line must read `OK`. The deposited run's output is
`stage2_superres/provenance/SR_EVAL_CLEAN_402698.out`.

To train, use mode `FINAL_WARM` (Stage 2, warm-started from the Stage 1 checkpoint) or
`GAT_LSTM_NN_Publication.ipynb` (Stage 1). Training resumes from the newest checkpoint in its
output folder.

---

## 8. Linear-regression baselines

`Linear_Regression.ipynb` fits five least-squares models on Stage 1's training split and scores
them on Stage 1's test windows with Stage 1's metric code: LR0, the static pretension; LR1, the
station's own inputs; LR2, plus the fairlead, the neighbouring stations and the line mean; LR3,
plus lags of ±0.1 to 5 s; LR4, plus lags every 1 s out to ±40 s. Its outputs are the
`linear_baseline` archive. It needs the raw simulations and, for the full run, a many-core node
(48 cores and about 190 GB of memory on the cluster, about 4 h from scratch; a laptop can run
the smoke mode on six simulations).

`make_lr_workdir.py` lays out the working directory the cluster job ran in, from this repository
and the archives (every file it places is byte-identical to the job's own input):

```bash
python make_lr_workdir.py --data /path/to/DATA --out lr_run
cd lr_run
export MOORING_CACHE_DIR=/path/to/cache_npy
LR_RUN_MODE=smoke jupyter nbconvert --to notebook --execute Linear_Regression.ipynb \
    --output Linear_Regression_executed.ipynb --ExecutePreprocessor.timeout=-1
```

`LR_RUN_MODE=full` runs the fits reported in the manuscript; results go to
`Results/LR_baseline/`. The job script is `cluster/run_LR_baseline.slurm`. To apply the deposited
weights to new data, build the inputs with the feature construction of `linear_regression_lib.py`
and standardise them with the archive's `standardiser*.npz`.

---

## 9. The raw simulations

The simulations come from the finite-element mooring solver of Agarwal, Sánchez Gómez and Colomés
(*Ocean Engineering*, 2026, https://doi.org/10.1016/j.oceaneng.2026.127337), at the sites and sea
states of Santjer et al. (*Applied Ocean Research* 165, 104809, 2025,
https://doi.org/10.1016/j.apor.2025.104809). They are not deposited: they are large and jointly
held by the solver's developers. They are available from the corresponding author on reasonable
request, as the preprocessed cache the code reads (about 13 GB) or as solver output.
`MOORING_CACHE_DIR` must point to:

```
cache_npy/loc01/case_0001/{x_abs,z_abs,tension,reference}.npy    float32, [13801, 21]
                         /env.npy                                float32, [13]
... for loc01, 03, 04, 05, 06, 07, 08, 09, 10, 11 and case_0001 ... case_0300
```

`x_abs`, `z_abs`: position (m); `tension`: axial tension (N); `reference`: unstretched arc length
of each station (m); `env`: [h0, Hs, Tp, then depth (m) and speed (m/s) at five current levels].
The function `preprocess_all_data` of the notebook builds this cache from the solver files.

`scan_corrupt_cases.py`, `scan_artifact_samples.py` and `scan_snap_events.py` build the tables of
`data/` from the cache. They write next to the scripts; move the results into `data/` to use them.

---

## 10. Citation and licence

Please cite the manuscript and the dataset (https://doi.org/10.4121/e3f167ac-2925-4815-956e-5ef3d883f1a5),
and for the simulations Agarwal, Sánchez Gómez and Colomés (2026) and Santjer et al. (2025).

This work extends the MSc thesis *Graph-Attention LSTM Network (GAT+LSTM) for Mooring Line
Load-Field Estimation Across Variable Topologies* (TU Delft, 2026), whose code is at
https://github.com/AthKon/Thesis_GAT_LSTM.

Code: MIT, see `LICENSE`. Data and checkpoints: CC BY 4.0.
