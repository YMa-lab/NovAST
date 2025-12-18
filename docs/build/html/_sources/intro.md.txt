# Introduction to NovAST

NovAST is a deep-learning framework for automated label transfer and novel cell type discovery in spatial transcriptomics.

## Installation
Here is the sample command to install packages into a python environment:
```
# clone the code from GitHub
git clone https://github.com/YMa-lab/NovAST.git
# create a new environemnbt with python==3.11
python -m venv NovAST_env
# activate the python environemtn 
source NovAST_env/bin/activate
# install the package
pip install .
```

## Quickstart

NovAST requires both **reference** and **target** datasets in **AnnData (.h5ad)** format. 
We need to first set up some basic path :

```python
savedir: "./"
train_path="path_to_train_dataset.h5ad"
test_path="path_to_train_dataset.h5ad"
training_mode = "exploration"

# Column name in the reference AnnData that stores cell-type annotations
celltype_name_train = "cell_type"
name = "demo_exploration"
dataset = "Xenium_IPF_lung"
```

All remaining hyperparameters are defined in the file **`default_config.yaml`**.  
Users may override any of them directly when calling `run_NovAST()` if customization is needed.

You can now run NovAST using the specified settings as follows:

```python
args = run_NovAST(
    training_mode=training_mode,
    train_path=train_path,
    test_path=test_path,
    celltype_name_train=celltype_name_train,
    name=name,
    dataset=dataset,
    rounds=10
)
```
For each training round, the pipeline saves all outputs to the specified directory, with each random seed assigned its own subfolder. This includes the trained model, the Stage-1 loss values, and a final result file named **`adata_unlabeled_final.h5ad`**, which stores the latent embeddings in **`.obsm['X_latent']`** and the final predicted labels in **`.obs['voted_final_prediction']`**.

Running the following line of code will generate UMAP visualizations as well as spatial plots of the predicted cell types and their associated confidence scores, and save them to each individual seed’s output directory.

```python
NovAST_plot(args)
```

## Demo
NovAST provides two operation modes:
- **Exploration mode** — recommended for applying NovAST on new datasets **without ground-truth labels** in the target dataset.
- **Evaluation mode** — reproduces the benchmarking pipeline used in the manuscript (requires ground-truth labels).

Below, we provide separate, step-by-step demos illustrating the workflow and expected outputs for each mode.