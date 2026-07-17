# STELLAR

Benchmark implementation for **STELLAR**, a graph-based method for cell-type annotation and novel-cell-type discovery in spatially resolved single-cell data.

Because STELLAR is distributed as research code rather than a packaged Python library, the relevant source code is vendored here under its original license. Instructions for installing the required environment and dependencies are provided in the Installation section.

**Reference.** Brbić, M., Cao, K., Hickey, J.W. et al. Annotation of spatially resolved single-cell data with STELLAR. Nat Methods 19, 1411–1418 (2022). https://doi.org/10.1038/s41592-022-01651-8

**Source.** https://github.com/snap-stanford/stellar

## Contents

Benchmark-specific code:

- `run_benchmark.py` — defines `STELLAR_main`, the benchmark function called by `../main.py`.

## Usage

For each random seed, `STELLAR_main` initializes `STELLAR(args, dataset)` using the shared preprocessed reference and target spatial graphs. The method is run across the number of random seeds specified by `--rounds`.
Results are saved to `results/<name>/STELLAR/seed{i}/`. Each seed-specific directory contains:

- `prediction.npy` — predicted cell labels
- `ground_truth.npy` — ground-truth cell labels
- `feat.csv`, `out_feat.csv` — learned cell embeddings
