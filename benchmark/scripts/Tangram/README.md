# Tangram

Benchmark implementation for **Tangram**, a method for aligning single-cell and spatial transcriptomes.

The officially released Tangram package is installed separately and invoked through its public API.

**Reference.** Biancalani, T., Scalia, G., Buffoni, L. et al. Deep learning and alignment of spatially resolved single-cell transcriptomes with Tangram. Nat Methods 18, 1352–1362 (2021). https://doi.org/10.1038/s41592-021-01264-7

**Source.** https://github.com/broadinstitute/Tangram

## Contents

Benchmark-specific code:

- `Tangram_main.py` — provides the `run_pair` and `chunk_and_run` wrappers around the Tangram API.
- `run_benchmark.py` — defines `Tangram_main`, the benchmark function called by `../main.py`.

## Installation

Install the released package into the benchmark environment:

```bash
pip install tangram-sc
```

## Usage

For each random seed, `Tangram_main` maps the reference cells to spatial locations in the target dataset.

Results are saved to `results/<name>/Tangram/`. Predictions for each target slice are stored as `seed{i}_pred_te_*.npy`, alongside corresponding ground-truth labels in `seed{i}_gt_te_*.npy`.

Because cell-level Tangram mapping can require substantial memory for large target sections, the benchmark wrapper supports spatial K-means chunking through `chunk_and_run`. The target section is partitioned into spatially coherent subsets, Tangram is applied to each subset independently, and the resulting predictions are combined in the original cell order.