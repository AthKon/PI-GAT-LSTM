# GAT-LSTM Mooring Line Tension Reconstruction

Code accompanying the manuscript *"Spatial Super-Resolution of Mooring Tension using a Topology-General, Physics-Informed Graph Attention Network"* (Konstantaras, Colomés Gené, Aguilar Lopez;
submitted to *Ocean Engineering*, mooring/anchoring/cable-systems special issue).

A mooring line is represented as a chain graph. A GATv2 spatial encoder captures coupling between
neighbouring stations, a per-node LSTM captures each station's temporal evolution, and a shared
head reconstructs the whole-line tension field from motion alone (x, z position and velocity;
no tension input). One trained weight set spans ten sensor layouts (4 to 21 stations) and ten
offshore locations, two of which are held out to test generalisation.

## Notebooks

### `GAT_LSTM_NN_Publication.ipynb` — Stage 1: matched-resolution reconstruction

Motion at N stations in → tension at the same N stations out, trained jointly across all ten
layouts and eight of the ten locations. An 8-term physics-informed loss (data fit, temporal
smoothness, spectral, constitutive strain, catenary equilibrium, nodal momentum balance, and a
quantile term on the upper tension tail) with Kendall uncertainty weighting between terms.

### `GAT_LSTM_NN_Publication_superresolution.ipynb` — Stage 2: spatial super-resolution

Adds a query-based cross-attention decoder between the spatial encoder and the temporal model, so
the network can be queried at more output stations than it was given as input (e.g. 4 sensors in,
21-station tension field out). Warm-started from the Stage-1 weights.

## Analysis and figure/table scripts

Every number, table and figure reported in the manuscript is produced by one of these scripts from
the models' saved evaluation outputs (`.csv`/`.npz`):

| script | purpose |
|---|---|
| `paperA_numbers.py` | generates every quoted number and LaTeX table in the manuscript |
| `paperA_figures.py` | generates every manuscript figure |
| `scan_snap_events.py` | catalogues snap-load (slack-to-taut) events in the raw simulation data |
| `scan_corrupt_cases.py` | flags FE solver blowups via a spatial-coherence criterion |
| `scan_artifact_samples.py` | builds the per-sample artifact/exclusion mask used in training |
| `exact_split_accounting.py` | reachability accounting for snap events across the train/val/test split |
| `pub_select_epoch.py` | re-applies the checkpoint-selection rule to a training run's logged metrics |
| `plot_test_timeseries.py` | re-plots stored test-set prediction windows (peaks, snaps, per-topology) |
| `plot_node_timeseries.py` | per-node time-series plots along the mooring line |
| `plot_tension_along_line.py` | tension-profile-along-the-line snapshots |

⚠ A few scripts (`paperA_numbers.py`, `exact_split_accounting.py`, `plot_tension_along_line.py`)
have local filesystem paths set for the original development environment; adjust the path
constants near the top of each file to reproduce elsewhere.

## Data

These notebooks and scripts expect a cache of finite-element mooring simulations (motion, tension
and contact time series for ten offshore locations) that is **not included in this repository**.
The simulation data was generated with the geometrically nonlinear finite-element mooring model of
Agarwal, Sánchez Gómez & Colomés (*Ocean Engineering*, 2026) and is not currently public owing to
its size (on the order of tens of gigabytes) and because it is jointly held by the model's
developers. It is available from the corresponding author on reasonable request.

Trained model checkpoints and the evaluation outputs the analysis scripts consume are archived
separately at 4TU.ResearchData: [DOI pending].

## Reference

This code extends the methodology of the MSc thesis *Graph-Attention LSTM Network (GAT+LSTM) for
Mooring Line Load-Field Estimation Across Variable Topologies* (TU Delft, 2026), whose original
(pre-publication) code is archived at
[github.com/AthKon/Thesis_GAT_LSTM](https://github.com/AthKon/Thesis_GAT_LSTM).

## License

MIT — see `LICENSE`.
