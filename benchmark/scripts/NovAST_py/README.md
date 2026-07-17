# NovAST

Benchmark implementation for **NovAST**. 

The officially released NovAST package is installed separately and invoked through its public API.

## Contents

- `run_benchmark.py` — defines `NovAST_main`, the benchmark function called by `../main.py`.
- `__init__.py`

## Installation

Clone the official NovAST repository and create a dedicated Python 3.11 environment:

```bash
# Clone the official NovAST repository
git clone https://github.com/YMa-lab/NovAST.git

# Create a Python 3.11 virtual environment
python -m venv NovAST_env

# Activate the virtual environment
source NovAST_env/bin/activate

# Install the package
pip install .
```

Verify the installation:

```bash
python -c "import NovAST; print(NovAST.run_NovAST)"
```

## Usage

`NovAST_main` runs `NovAST.run_NovAST(...)` using the shared, preprocessed benchmark data.
Results are written to: `results/<name>/NovAST/seed{i}/`. Each output directory includes `adata_unlabeled_final.h5ad`, which contains the following columns used by the evaluation notebook:

- `obs['voted_final_prediction']` — predicted cell labels
- `obs['ground_truth']` — ground-truth cell labels

By default, method-specific hyperparameters are loaded from the package's `default_config.yaml`. Alternative values can be supplied through the `overrides={...}` argument of `NovAST_main`.


## Benchmark-specific settings

The benchmark implementation modifies only the following dataset-level settings:

```python
preprocess_skip = True
remove_celltype = False
uncontrolled = True
````

These settings prevent NovAST from repeating data preparation steps that have already been performed upstream and applied consistently across all benchmarking methods.