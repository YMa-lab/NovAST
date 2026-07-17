# Spatial Cell-Type Annotation Benchmark
This directory contains the reproducible benchmarking workflow used to evaluate **NovAST** against five existing cell-type annotation methods:

* SingleR
* Tangram
* Spatial-ID
* STELLAR
* scBOL

The workflow uses the MERFISH mouse primary motor cortex (MOp) dataset as a worked example. One spatial section is used as the reference dataset and another as the target dataset.

## Directory structure

```
benchmark/
├── scripts/                       # Benchmarking pipeline and method implementations
│   ├── main.py                    # Entry point for preprocessing and method execution
│   ├── datasets.py, preprocess.py, utils.py   # shared pipeline helpers
│   ├── NovAST_py/ STELLAR_py/ SpatialID/ Tangram/ scBOL/   # main.py methods
│   └── SingleR/                   # R baseline, run separately (see its README)
├── results/                       # Benchmark outputs: results/<name>/...
├── 01_run_merfish_split.ipynb     # Preprocessing and method execution
└── 02_evaluate_merfish_split.ipynb# Metric calculation and visualization
```
Additional implementation details and method-specific instructions are provided in [`scripts/README.md`](scripts/README.md).

## Quick start

1. Open **`01_run_merfish_split.ipynb`** and run it top to bottom. It preprocesses
   the data once, then runs each method. Outputs go to
   `results/merfish_split_<setting>/` (e.g. `results/merfish_split_most/` for the
   default `SETTING = "most"`).
2. Open **`02_evaluate_merfish_split.ipynb`** to compute accuracy / weighted F1 /
   ARI / macro F1 and plot the method comparison.

## The pipeline in one line

```bash
# stage 1 — preprocess (once per setting)
python scripts/main.py --stage preprocess --savedir results/ --name merfish_split_most \
    --reference_path /path/to/merfish_split1.h5ad --target_path /path/to/merfish_split2.h5ad \
    --region_name_reference slice --region_name_target slice \
    --celltype_name_reference cell_type --celltype_name_target cell_type \
    --remove-celltype --remove-celltype-type most

# stage 2 — run one method (repeat per method)
python scripts/main.py --stage run --savedir results/ --name merfish_split_most --method NovAST
```

`--remove-celltype --remove-celltype-type {most,least,closest,furthest}` at
preprocess holds out one cell type from the Reference to simulate a novel cell
type. Omit both flags (as in the `none` setting) to keep every cell type shared.

Raw data: `<data_dir>/merfish_split{1,2}.h5ad` — set `<data_dir>` to your local
copy (split1 = reference, split2 = target; both spatial → `st_to_st`).
