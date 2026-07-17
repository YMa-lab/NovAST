# NovAST: Novel cell type discovery and Annotation in Spatial Transcriptomics
This repository contains the source code for the following study: 

**NovAST: A Deep Learning Framework for Automated Label Transfer and Novel Cell Type Discovery in Spatial Transcriptomics.**

## Python dependencies
The required Python packages are listed in `requirements.txt`.

## Jupyter notebook
Example workflows are provided and documented in the accompanying Jupyter notebook.

## Installation
Clone the official NovAST repository and create a dedicated Python 3.11 environment:
```
# Clone the official NovAST repository
git clone https://github.com/YMa-lab/NovAST.git

# Create a Python 3.11 virtual environment
python -m venv NovAST_env

# Activate the virtual environment
source NovAST_env/bin/activate

# Install the package
pip install .
```

## Tutorial
The tutorial provides step-by-step instructions for preparing input data, configuring NovAST, running the package on individual datasets, and interpreting the resulting cell type annotations and novel cell populations.

For detailed usage instructions and dataset-specific examples, see the [NovAST documentation](https://novast.readthedocs.io/en/latest/).

## Benchmarking
The `benchmark/` directory contains the scripts used to benchmark NovAST against five existing methods:

* SingleR
* Tangram
* Spatial-ID
* STELLAR
* scBOL

The benchmarking workflow includes method-specific implementations, shared data preprocessing, model execution, and performance evaluation. Each method is run through its released package or public implementation using a consistent set of preprocessed inputs whenever possible.

See the README files within `benchmark/` and its method-specific subdirectories for installation instructions, usage details, and output formats.