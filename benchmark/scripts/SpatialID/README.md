# Spatial-ID

Benchmark implementation for **Spatial-ID**, a transfer-learning method for cell type annotation in spatial transcriptomics.

For reproducibility, the Spatial-ID implementation used in this benchmark is vendored here under its original license. Instructions for installing the required environment and dependencies are provided in the [Installation](#installation) section.

**Reference.** Shen, R., Liu, L., Wu, Z. et al. Spatial-ID: a cell typing method for spatially resolved transcriptomics via transfer learning and spatial embedding. Nat Commun 13, 7640 (2022). https://doi.org/10.1038/s41467-022-35288-0

**Source.** https://github.com/TencentAILabHealthcare/spatialID

## Contents

Benchmark-specific code:

- `run_benchmark.py` — defines `SpatialID_main`, the benchmark function called by `../main.py`.

## Usage

For each random seed, `SpatialID_main` invokes `spatial_classification_tool(...)` using the shared reference and target datasets. The workflow proceeds in three stages:

1. `learn_sc` trains a DNN classifier on the annotated reference dataset.
2. `sc2st` transfers the reference-derived predictions to the spatial target as pseudo-labels.
3. `annotation` refines the pseudo-labels using the spatial graph model.

Results are saved to `results/<name>/SpatialID/seed{i}/`. Each seed-specific directory contains:

* `annotation.h5ad` — the annotated target dataset, with predicted cell-type labels stored in `obs["celltype_pred"]`
* `model.csv` — cell identifiers and their predicted cell-type labels
