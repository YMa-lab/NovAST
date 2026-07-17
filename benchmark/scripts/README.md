# Benchmark methods

Reference implementations of the cell-type annotation methods compared in this
benchmark. Each subfolder wraps one method behind a common `run_benchmark.py`
entry point that consumes the shared preprocessed data and writes a uniform
per-method output layout.

| Folder | Method | Source | Integration |
|--------|--------|--------|-------------|
| [`NovAST_py`](NovAST_py) | NovAST | first-party (installed package) | thin adapter |
| [`STELLAR_py`](STELLAR_py) | STELLAR | [snap-stanford/STELLAR](https://github.com/snap-stanford/STELLAR) | vendored |
| [`SpatialID`](SpatialID) | Spatial-ID | [JackieHanLab/Spatial-ID](https://github.com/JackieHanLab/Spatial-ID) | vendored |
| [`scBOL`](scBOL) | scBOL | research code (not packaged) | vendored |
| [`Tangram`](Tangram) | Tangram | [`tangram-sc`](https://pypi.org/project/tangram-sc/) | thin wrapper |
| [`SingleR`](SingleR) | SingleR | [Bioconductor](https://bioconductor.org/packages/SingleR) | R workflow |

**Vendored** methods are GitHub-only research code with no installable release,
so a copy is checked in under each project's original license; see the per-folder
README for the citation, upstream link, and the modifications applied.
**Installed** methods (NovAST, Tangram) are called from their released packages,
so only a driver lives here.

## Pipeline

`main.py` runs in two stages:

1. `--stage preprocess` — gene intersection, normalization, optional held-out
   cell type, and spatial-graph construction, shared by every method.
2. `--stage run --method <M>` — dispatches to `<M>/run_benchmark.py`.

SingleR is the exception: it is an R workflow driven directly (see
[`SingleR/README.md`](SingleR)), not through `main.py`.
