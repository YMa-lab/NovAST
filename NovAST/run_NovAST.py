import argparse
import numpy as np
import os
import torch
import random
import yaml
import importlib.resources as pkg_resources
import NovAST
from .datasets import read_dataset
from .utils import detect_spatial_info
from .preprocess import preprocess
from .main_functions import *
from .plot_eval import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
from anndata import ImplicitModificationWarning
import pandas as pd

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging

logging.getLogger("anndata").setLevel(logging.ERROR)
logging.getLogger("scanpy").setLevel(logging.ERROR)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

# Config keys were renamed train -> reference and test -> target, so the names
# say WHICH DATASET a value describes ("reference"/"target") rather than the
# overloaded "train" (which also means the act of training -- cf. NovAST_train,
# train_epoch, which keep their names). The old names still work for one release
# and emit a DeprecationWarning.
_LEGACY_ALIASES = {
    "train_path": "reference_path",
    "test_path": "target_path",
    "dataset_train": "dataset_reference",
    "dataset_test": "dataset_target",
    "celltype_name_train": "celltype_name_reference",
    "celltype_name_test": "celltype_name_target",
    "celltype_name_train_select": "celltype_name_reference_select",
    "celltype_name_test_select": "celltype_name_target_select",
    "region_name_train": "region_name_reference",
    "region_name_test": "region_name_target",
}


def _apply_legacy_aliases(kwargs):
    """Accept the pre-rename kwarg names, warn, and translate them."""
    out = {}
    for key, value in kwargs.items():
        new_key = _LEGACY_ALIASES.get(key)
        if new_key is None:
            out[key] = value
            continue
        if new_key in kwargs:
            raise TypeError(
                f"run_NovAST() got both '{key}' (deprecated) and '{new_key}'. "
                f"Pass only '{new_key}'."
            )
        warnings.warn(
            f"run_NovAST(): '{key}' is deprecated and will be removed in a future "
            f"release; use '{new_key}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        out[new_key] = value
    return out


def run_NovAST(config_file="default_config.yaml", **override_kwargs):
    """
    Main NovAST pipeline callable as a function.
    Loads defaults from a YAML config, applies user overrides,
    and executes the full pipeline (train + identify novel cells + voting).

    Dataset-naming: use `reference_path` / `target_path` and
    `celltype_name_reference` / `celltype_name_target`. The old train/test names
    still work but raise a DeprecationWarning (see `_LEGACY_ALIASES`).
    """
    # ====================== Load configuration ======================
    if config_file == "default_config.yaml":
        with pkg_resources.files(NovAST).joinpath("default_config.yaml").open("r") as f:
            cfg = yaml.safe_load(f)
    else:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
    override_kwargs = _apply_legacy_aliases(override_kwargs)
    cfg.update(override_kwargs)  # update the user defined paramters
    class Args: pass
    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)

    # Device
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # ====================== Setup output folder ======================
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)
    print(f"The saving directory set to {args.savedir}", flush=True)

    # ====================== Load datasets ======================
    if args.reference_path and args.target_path:
        adata_reference = read_dataset(data_type='reference', reference_path=args.reference_path)
        adata_target = read_dataset(data_type='target', target_path=args.target_path)
    else:
        raise ValueError("You must provide both --reference_path and --target_path.")
    
    # ====================== Validate obs fields ======================
    if args.celltype_name_reference not in adata_reference.obs.columns:
        raise ValueError(f"{args.celltype_name_reference} not found in training dataset obs.")

    if "evaluation" in args.training_mode: # check testing if in the evaluation mode
        if args.celltype_name_target not in adata_target.obs.columns:
            raise ValueError(f"{args.celltype_name_target} not found in testing dataset obs.")

    if args.celltype_name_reference_select == "None": # select removed gene given 'celltype_name_reference/test' if not further specified
        args.celltype_name_reference_select = args.celltype_name_reference
    if args.celltype_name_target_select == "None":
        args.celltype_name_target_select = args.celltype_name_target
    
    print(args.celltype_name_reference_select)

    has_spatial_reference, reference_loc = detect_spatial_info(adata_reference)
    has_spatial_target, target_loc = detect_spatial_info(adata_target)
    args.target_spatial_loc = target_loc

    if not has_spatial_target:
        raise ValueError("Spatial coordinates ('spatial') not found in target dataset.")
    
    # ====================== Predefine subsampling and hvg selection if needed ======================
    if has_spatial_reference:
        args.training_type = "st_to_st"
        args.gmm_thresh = 0.12
    else:
        # sampling training cells if it is scRNA-seq and has significanly larger amount
        args.training_type = "sc_to_st"
        if adata_reference.n_obs > 3 * adata_target.n_obs and args.sampling_cells==None:
            args.sampling_cells = 1
        args.gmm_thresh = 0.06

    # if number of genes is high, 
    if adata_reference.n_vars > 5000 and adata_target.n_vars > 5000 and args.sampling_genes==None:
        args.sampling_genes = 5000

    if "exploration" in args.training_mode:
        args.no_gt = True
        args.remove_celltype = False
        args.uncontrolled = True
        print("The training mode set to exploration!", flush=True)

    # ====================== Preprocessing ======================
    if args.preprocess_skip != True:
        adata_reference, adata_target = preprocess(adata_reference, adata_target, args.reference_path, args.target_path, args.sampling_genes)
        print("Datasets have been preprocessed!")
    
    # ====================== Prepare final dataset for training ======================
    dataset, labeled_y, unlabeled_y, _, adata_reference, adata_target = load_paired_dataset(adata_reference, adata_target, args)

    # ====================== Multi-round training loop ======================
    for i in range(1, args.rounds+1):
        print("──────────────────────────────────────────")
        print(f"Starting training for seed {i}...")        
        set_seed(i+1)
        filedic = os.path.join(args.savedir,f"seed{i}")
        os.makedirs(filedic, exist_ok=True)

        # ---- Step I: Train AE with regularization ----
        labeled_y, z_reference, unlabeled_y, z_target = training_model(i, args, dataset, labeled_y, unlabeled_y, filedic)

        # ---- Step II: Novel cell discovery ----
        if 'exploration' in args.training_mode:
            adata_all, adata_unlabeled, predicted_labels, labeled_y, z_target, _ = identify_novel_cells_exploration(labeled_y, z_reference, unlabeled_y, z_target, args, filedic, adata_reference, adata_target)

        elif 'evaluation' in args.training_mode:
            adata_all, adata_unlabeled, predicted_labels, labeled_y, z_target, unlabeled_y_name, _ = identify_novel_cells_evaluation(labeled_y, z_reference, unlabeled_y, z_target, args, filedic, adata_reference, adata_target)

    # ====================== Voting step ======================
    args.vote_thresh = int(args.rounds) // 2 
    vote(args)
         
    return args

def NovAST_plot(args):
    """
    Generate exploration-mode plots for each trained seed.
    This function loads precomputed outputs from run_NovAST()
    and produces visualization without retraining.
    """
    if "exploration" in args.training_mode:
        print(f"Exploration mode detected. Generating plots for {args.rounds} seeds...\n")

        for i in range(1, args.rounds+1):
            if i == 1:
                show = True
            else:
                show = False
            print(f"──────────────────────────────────────────")
            print(f"Starting plotting for seed {i}...")

            set_seed(i+1)
            filedic = os.path.join(args.savedir, f"seed{i}")
            adata_all, adata_unlabeled = load_exist_data_exploration(args, filedic)

            print(f"Generating UMAP plot...")
            plot_umap(adata_all, adata_unlabeled, save_path=filedic, show=show)
            print(f"UMAP plot saved.")

            print(f"Generating spatial plot...")
            plot_spatial(args, adata_unlabeled, save_path=filedic, show=show)
            print(f"Spatial plot saved.\n")
    
        print("──────────────────────────────────────────")
        print("All seeds plotted successfully.\n")
    else:
        print("Dataset is not in exploration mode; skipping exploration plots.", flush=True)

def NovAST_evaluation(args):
    """
    Perform evaluation-mode accuracy assessment and plotting.
    Loads saved prediction outputs from each seed, computes accuracy
    (or other evaluation metrics), and saves aggregated results.
    """
    
    if "evaluation" in args.training_mode:
        print(f"Evaluation mode detected. Processing {args.rounds} seeds...\n")

        for i in range(1, args.rounds+1):
            print("──────────────────────────────────────────")
            print(f"Starting evaluation for seed {i}...")
            set_seed(i+1)
            # Train the part one model (an autoencoder with additional losses)
            filedic = os.path.join(args.savedir, f"seed{i}")
            adata_all, adata_unlabeled, inverse_dict = load_exist_data_evaluation(args, filedic)

            print(f"Generating UMAP plot with ground truth...")
            plot_umap(adata_all, adata_unlabeled, save_path=filedic, ground_truth=True)
            print(f"UMAP plot saved.")

            print(f"Generating spatial plot (region key = {args.region_name_target})...")
            plot_spatial(args, adata_unlabeled, save_path=filedic, ground_truth=True)
            print(f"Spatial plot saved.\n")

        print("──────────────────────────────────────────")
        print("All seeds processed. Computing aggregated metrics...\n")

        metrics_df = gather_metrics(args.savedir, args.dataset, max_seed=args.rounds)
        plot_representative(args, metrics_df, metric="accuracy")
    else:
        print("Dataset is not in evaluation mode; skipping evaluation and plotting.", flush=True)
    return metrics_df
