# scBOL

Benchmark implementation for **scBOL**, a universal cell type identification framework for single-cell and spatial transcriptomics.

Because scBOL is released as research code rather than a packaged Python library, the relevant source code is vendored here under its original license. Our benchmark uses the spatial transcriptomics implementation from the [`code/spatial`](https://github.com/aimeeyaoyao/scBOL/tree/main/code/spatial) directory of the official scBOL repository. Instructions for installing the required environment and dependencies are provided in the [Installation](#installation) section.

**Reference.** Yuyao Zhai, Liang Chen, Minghua Deng, scBOL: a universal cell type identification framework for single-cell and spatial transcriptomics data, Briefings in Bioinformatics, Volume 25, Issue 3, May 2024, bbae188, https://doi.org/10.1093/bib/bbae188

## Contents

Benchmark-specific code:

- `run_benchmark.py` — defines `scBOL_main`, the benchmark function called by `../main.py`.

## Usage

`scBOL_main` builds `BOL(args, dataset)` from the shared, preprocessed spatial graph and runs `--rounds` seeds.
Results are written to `results/<name>/scBOL/seed{i}/`, with each directory containing:

- `prediction.npy` — predicted cell labels
- `ground_truth.npy` — ground-truth cell labels
- `out_feat.npy` — learned cell embeddings