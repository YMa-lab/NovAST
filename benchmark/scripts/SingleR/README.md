# SingleR

Benchmark implementation for **SingleR**, a reference-based cell-type annotation baseline.

Because SingleR is an R package distributed through Bioconductor, it is executed through a file-based workflow rather than directly through `main.py`. The Python and R components exchange data through intermediate files and run in separate processes.

**Reference.** Aran, D., Looney, A.P., Liu, L. et al. Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage. Nat Immunol 20, 163–172 (2019). https://doi.org/10.1038/s41590-018-0276-y

**Source.** https://bioconductor.org/packages/SingleR

## Contents

- `run_singler.R` — runs cell-type classification using Bioconductor SingleR.
- `singler_io.py` — provides the Python I/O bridge through its `prepare` and `convert` modes.

## Installation

The R workflow requires R and the following packages: `SingleR`, `SingleCellExperiment`, `scuttle`, `Matrix`, `jsonlite`, and `optparse`.

## Usage

The workflow comprises three steps. Here, `SAVEDIR` denotes the directory containing the preprocessed inputs for one benchmark run:

```bash
python singler_io.py prepare --savedir $SAVEDIR    # h5ad  -> .mtx + .csv
Rscript  run_singler.R      --savedir $SAVEDIR     # run classification
python singler_io.py convert --savedir $SAVEDIR    # .csv  -> .npy
```

Results are saved to `results/<name>/SingleR/`. This directory contains `prediction.npy` and `ground_truth.npy` in the layout expected by the evaluation notebook.

Because this benchmark workflow involves no random initialization or seed-dependent training, SingleR is run once per dataset. Its outputs are therefore stored directly in the method directory rather than in `seed{i}/` subdirectories.