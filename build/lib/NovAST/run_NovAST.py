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

def run_NovAST(config_file="default_config.yaml", **override_kwargs):
    """
    Main NovAST pipeline callable as a function.
    Loads defaults from a YAML config, applies user overrides,
    and executes the full pipeline (train + identify novel cells + voting).
    """
    # ====================== Load configuration ======================
    if config_file == "default_config.yaml":
        with pkg_resources.files(NovAST).joinpath("default_config.yaml").open("r") as f:
            cfg = yaml.safe_load(f)
    else:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
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
    if args.train_path and args.test_path:
        adata_train = read_dataset(data_type='train', train_path=args.train_path)
        adata_test = read_dataset(data_type='test', test_path=args.test_path)
    else:
        raise ValueError("You must provide both --train_path and --test_path.")
    
    # ====================== Validate obs fields ======================
    if args.celltype_name_train not in adata_train.obs.columns:
        raise ValueError(f"{args.celltype_name_train} not found in training dataset obs.")

    if "evaluation" in args.training_mode: # check testing if in the evaluation mode
        if args.celltype_name_test not in adata_test.obs.columns:
            raise ValueError(f"{args.celltype_name_test} not found in testing dataset obs.")

    has_spatial_train, train_loc = detect_spatial_info(adata_train)
    has_spatial_test, test_loc = detect_spatial_info(adata_test)
    args.test_spatial_loc = test_loc

    if not has_spatial_test:
        raise ValueError("Spatial coordinates ('spatial') not found in target dataset.")
    
    # ====================== Predefine subsampling and hvg selection if needed ======================
    if has_spatial_train:
        args.training_type = "st_to_st"
        args.gmm_thresh = 0.12
    else:
        # sampling training cells if it is scRNA-seq and has significanly larger amount
        args.training_type = "sc_to_st"
        if adata_train.n_obs > 3 * adata_test.n_obs and args.sampling_cells==None:
            args.sampling_cells = 1
        args.gmm_thresh = 0.06

    # if number of genes is high, 
    if adata_train.n_vars > 5000 and adata_test.n_vars > 5000 and args.sampling_genes==None:
        args.sampling_genes = 5000

    if "exploration" in args.training_mode:
        args.no_gt = True
        args.remove_celltype = False
        args.uncontrolled = True
        print("The training mode set to exploration!", flush=True)

    # ====================== Preprocessing ======================
    if args.preprocess_skip != True:
        adata_train, adata_test = preprocess(adata_train, adata_test, args.train_path, args.test_path, args.sampling_genes)
        print("Datasets have been preprocessed!")
    
    # ====================== Prepare final dataset for training ======================
    dataset, labeled_y, unlabeled_y, _, adata_train, adata_test = load_paired_dataset(adata_train, adata_test, args)

    # ====================== Multi-round training loop ======================
    for i in range(1, args.rounds+1):
        print("──────────────────────────────────────────")
        print(f"Starting training for seed {i}...")        
        set_seed(i+1)
        filedic = os.path.join(args.savedir,f"seed{i}")
        os.makedirs(filedic, exist_ok=True)

        # ---- Step I: Train AE with regularization ----
        labeled_y, z_train, unlabeled_y, z_test = training_model(i, args, dataset, labeled_y, unlabeled_y, filedic)

        # ---- Step II: Novel cell discovery ----
        if 'exploration' in args.training_mode:
            adata_all, adata_unlabeled, predicted_labels, labeled_y, z_test, _ = identify_novel_cells_exploration(labeled_y, z_train, unlabeled_y, z_test, args, filedic, adata_train, adata_test)

        elif 'evaluation' in args.training_mode:
            adata_all, adata_unlabeled, predicted_labels, labeled_y, z_test, unlabeled_y_name, _ = identify_novel_cells_evaluation(labeled_y, z_train, unlabeled_y, z_test, args, filedic, adata_train, adata_test)

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
            print(f"──────────────────────────────────────────")
            print(f"Starting plotting for seed {i}...")

            set_seed(i+1)
            filedic = os.path.join(args.savedir, f"seed{i}")
            adata_all, adata_unlabeled = load_exist_data_exploration(args, filedic)

            print(f"Generating UMAP plot...")
            plot_umap(adata_all, adata_unlabeled, filedic)
            print(f"UMAP plot saved.")

            print(f"Generating spatial plot...")
            plot_spatial(args, adata_unlabeled, filedic)
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
            plot_umap(adata_all, adata_unlabeled, filedic, ground_truth=True)
            print(f"UMAP plot saved.")

            print(f"Generating spatial plot (region key = {args.region_name_test})...")
            plot_spatial(args, adata_unlabeled, filedic, ground_truth=True)
            print(f"Spatial plot saved.\n")

        print("──────────────────────────────────────────")
        print("All seeds processed. Computing aggregated metrics...\n")

        metrics_df = gather_metrics(args.savedir, args.dataset, max_seed=args.rounds)
    else:
        print("Dataset is not in evaluation mode; skipping evaluation and plotting.", flush=True)
    return metrics_df
