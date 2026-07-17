import argparse
import sys
import time
import pandas as pd
import numpy as np
import scanpy as sc
import os, pickle, json
from datasets import read_dataset, load_dataset_adata
import torch
import resource  
from utils import has_spatial, set_seed, load_complete_dataset
from preprocess import preprocess
import ast
# Each method's benchmark runner (*_main) now lives in its own package folder and
# is imported lazily at dispatch time (below), so a single-method run only pulls
# in that method's dependencies — in particular Tangram's TensorFlow import stays
# deferred until a Tangram run actually needs it.

def build_args():
    parser = argparse.ArgumentParser(
        description="Preprocess + run one of {NovAST,SpatialID,STELLAR,Tangram}"
    )
    # fundamental setup / input & output
    parser.add_argument('--savedir', type=str, default='./')
    parser.add_argument('--name', type=str, default='run')
    parser.add_argument('--training_mode', type=str, default="evaluation_full",help="Available modes are 'exploration_full', 'exploration_step2', 'exploration_step3', 'exploration_plot', 'evaluation_full', 'evaluation_step1', 'evaluation_step2', 'evaluation_step3', 'evaluation_plot'")
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--dataset_reference', type=str, default='train')
    parser.add_argument('--dataset_target', type=str, default='test')
    parser.add_argument('--reference_path', type=str, default=None)
    parser.add_argument('--target_path', type=str, default=None)
    parser.add_argument('--celltype_name_reference', type=str, default='cell_type', help='the obs_name for cell types to use for reference')
    parser.add_argument('--celltype_name_target', type=str, default='cell_type', help='the obs_name for cell types to use for target')
    parser.add_argument('--celltype_name_reference_select', type=str, default=None, help='the obs_name for cell types to use to select cells to remove')
    parser.add_argument('--celltype_name_target_select', type=str, default=None, help='the obs_name for cell types to use to select novel cell types')
    parser.add_argument('--region_name_reference', type=str, default=None, help='the obs_name for regions to use for reference')
    parser.add_argument('--region_name_target', type=str, default=None, help='the obs_name for regions to use for target')
    parser.add_argument('--region_name_reference_select', type=str, default='brain_section_label', help='the obs_name for regions to use for reference select')
    parser.add_argument('--region_name_target_select', type=str, default='brain_section_label', help='the obs_name for regions to use for target select')
    parser.add_argument('--exploration', type=bool, default=False)

    # dataset control (the same for all method)
    parser.add_argument('--remove-celltype', action='store_true', default=False, help='If present, drop the specified cell type in preprocess')
    parser.add_argument('--remove-celltype-type', type=str, default="most")
    parser.add_argument('--remove_celltype_name', type=str, default=None,
                        help='Explicit cell-type name(s) in the *_select column to drop from the Reference '
                             '(comma-separated for multiple). When set, this OVERRIDES the '
                             'closest/furthest UMAP-centroid selection so the held-out type is '
                             'reproducible instead of depending on stochastic UMAP.')
    parser.add_argument('--uncontrolled', type=bool, default=False)
    parser.add_argument('--no_gt', type=bool, default=False)
    parser.add_argument('--sampling_cells', type=int, default=None)
    parser.add_argument('--sampling_genes', type=str, default=None)
    parser.add_argument('--sampling_cells_method', type=str, default='prop')
    parser.add_argument('--select', type=str, default=None)
    parser.add_argument('--rm_ref', type=bool, default=False)

    # preprcoessing or choices of method to test
    parser.add_argument(
      "--stage",
      choices=["preprocess","run",'stellar_add'],
      required=True,
      help="'preprocess': do data prep only; 'run': apply methods to prepped data"
    )
    parser.add_argument('--method', type=str)
    parser.add_argument('--graph', type=bool, default=False)
    parser.add_argument('--graph_method', type=str, default='radius', choices=['radius', 'brute'],
                        help="Spatial graph construction at preprocess: 'radius' (tree-based "
                             "radius_neighbors_graph, default) or 'brute' (upstream STELLAR's "
                             "O(n^2) dense pairwise_distances matrix).")

    ### NovAST parameteres
    # training details
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('-b', '--batch-size', default=500, type=int)
    parser.add_argument('--k_neighbors', type=int, default=10)
    # hyperparameters
    parser.add_argument('--gamma', type=float, default=2)    
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--mmf_k', type=int, default=100)    
    parser.add_argument('--mmf_margin', type=float, default=2.0)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--dbscan', type=int, default=150)
    parser.add_argument('--gmm_ncomponent', type=int, default=8)
    # model arch
    parser.add_argument('--GAT_heads', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=64) 
    parser.add_argument('--hidden_dims', type=ast.literal_eval, default=[128, 128])
    parser.add_argument('--dropout', type=float, default=None)

    ### STELLAR paramters
    parser.add_argument('--stellar-epochs',      type=int,   default=20,    help='STELLAR: number of epochs')
    parser.add_argument('--stellar-lr',          type=float, default=1e-3,  help='STELLAR: learning rate')
    parser.add_argument('--stellar-wd',          type=float, default=5e-2,  help='STELLAR: weight decay')
    parser.add_argument('--stellar-batch-size',  type=int,   default=1,     help='STELLAR: mini-batch size')
    parser.add_argument('--num-heads', type=int, default=45)
    parser.add_argument('--num-seed-class', type=int, default=0)
    parser.add_argument('--sample-rate', type=float, default=0.5)
    parser.add_argument('--num_parts_reference', default=10, type=int)
    parser.add_argument('--num_parts_target', default=10, type=int)
    parser.add_argument('--distance_thres_reference', default=50, type=float)
    parser.add_argument('--distance_thres_target', default=50, type=float)

    ### scBOL parameters
    parser.add_argument('--scbol-lr',       type=float, default=1e-3,  help='scBOL: learning rate')
    parser.add_argument('--scbol-wd',       type=float, default=5e-2,  help='scBOL: weight decay')
    parser.add_argument('--scbol-nbr',      type=int,   default=30,    help='scBOL: number of ssKM prototype initializations')
    parser.add_argument('--scbol-quantile',  type=float, default=0.8,  help='scBOL: confidence quantile threshold')

    return parser.parse_args()

def main():
    args = build_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if args.celltype_name_reference_select is None:
        args.celltype_name_reference_select = args.celltype_name_reference
    if args.celltype_name_target_select is None:
        args.celltype_name_target_select = args.celltype_name_target

    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)
    print(f"The saving directory set to {args.savedir}", flush=True)

    if args.stage == 'preprocess':
        # Skip if preprocess outputs already exist. Both the loader pickle (consumed
        # by NovAST/STELLAR/scBOL) and the per-split h5ads (consumed by SpatialID/
        # Tangram) need to be present, alongside the args JSON written last.
        # The guard lists EVERY stage-1 artefact. If any one is missing the whole
        # stage re-runs -- that is what repairs a run dir left half-written by an
        # older/renamed version of this script, instead of silently skipping and
        # failing later in a method that expects the missing file.
        required = [
            os.path.join(args.savedir, f) for f in (
                "preprocess_args.json",
                "loader_output.pkl",
                "reference_preprocessed.h5ad", "target_preprocessed.h5ad",
                "reference_normalized.h5ad",   "target_normalized.h5ad",
                "inverse_dict_reference.pkl",  "inverse_dict_target.pkl",
            )
        ]
        missing = [p for p in required if not os.path.exists(p)]
        if not missing:
            print(f"Preprocess outputs already exist under {args.savedir}, skipping.", flush=True)
            return
        if len(missing) < len(required):
            print("Re-running preprocess; missing stage-1 outputs: "
                  + ", ".join(os.path.basename(p) for p in missing), flush=True)

        ###### load dataset
        # Seed the run and create saving directory
        _pre_start = time.time()
        set_seed(args.seed)

        if args.exploration:
            args.uncontrolled = True
            
        # read datasets
        if args.reference_path and args.target_path:
            adata_reference = read_dataset(data_type='reference', reference_path=args.reference_path)
            adata_target = read_dataset(data_type='target', target_path=args.target_path)
        else:
            raise ValueError("You must provide either --dataset or both --reference_path and --target_path.")
        
        # check whether cell type found in reference dataset
        if args.celltype_name_reference not in adata_reference.obs.columns:
            raise ValueError(
                f"'{args.celltype_name_reference}' not found in reference dataset. Please check whether the cell type exists, or if the 'celltype_name_reference' argument is correctly specified - especially if an alterantive name is used!"
            )
        # check whether cell type found in target dataset for any mode other than exploration
        if args.celltype_name_target not in adata_target.obs.columns:
            raise ValueError(
                f"'{args.celltype_name_target}' not found in target dataset. Please check whether the 'celltype_name_target' argument or switch to exploration mode where cell types are not required.!"
            )

        if has_spatial(adata_target) == False:
            raise ValueError("Spatial coordinates ('spatial') not found in target dataset.")
        
        if has_spatial(adata_reference):
            args.training_type = "st_to_st"
            args.gmm_thresh = 0.12
            args.graph = True
        else:
            # sampling training cells if it is scRNA-seq and has significanly larger amount
            args.training_type = "sc_to_st"
            if adata_reference.n_obs > 3 * adata_target.n_obs and args.sampling_cells==None:
                args.sampling_cells = 1
            args.gmm_thresh = 0.06

        if adata_reference.n_vars > 5000 and adata_target.n_vars > 5000 and args.sampling_genes==None:
            args.sampling_genes = 5000

        # preprocess dataset by taking the subset between ref and target, normalize by total, log1p, and scale
        adata_reference, adata_target = preprocess(adata_reference, adata_target, args.reference_path, args.target_path, args.sampling_genes)

        loader_out = load_dataset_adata(args, adata_reference, adata_target)
        with open(os.path.join(args.savedir, 'loader_output.pkl'), 'wb') as f:
            pickle.dump(loader_out, f)

        save_args = vars(args).copy()
        save_args["device"] = str(save_args["device"])
        with open(os.path.join(args.savedir, "preprocess_args.json"), "w") as f:
            json.dump(save_args, f, indent=2)

        # record total preprocess wall time + peak RSS (includes the graph build)
        pre_stats = {
            "preprocess_total_sec": time.time() - _pre_start,
            "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "graph_method": getattr(args, "graph_method", "radius") or "radius",
        }
        with open(os.path.join(args.savedir, "preprocess_stats.json"), "w") as f:
            json.dump(pre_stats, f, indent=2)

        print(f"Preprocess + loader output saved under {args.savedir} "
              f"(total {pre_stats['preprocess_total_sec']:.1f}s, peak RSS {pre_stats['peak_rss_kb']} KB)")
        return

    if args.stage == "run":
        # load previously saved paramters
        with open(os.path.join(args.savedir, 'preprocess_args.json')) as f:
            pre_args = json.load(f)

        for k, v in pre_args.items():
            # skip anything you're explicitly overriding in run
            if k in ["stage", "method", "num_seed_class", "rounds",
                      "scbol_lr", "scbol_wd", "scbol_nbr", "scbol_quantile"]:
                continue
            setattr(args, k, v)

        print(args)
        # previousy saved the device to str and now convert it back
        if isinstance(args.device, str):
            args.device = torch.device(args.device)

        # load data
        if args.method in ["SpatialID", 'Tangram']:
            # counts-flavoured h5ads: these methods normalize internally
            adata_reference = sc.read_h5ad(os.path.join(args.savedir, f"reference_preprocessed.h5ad"))
            adata_target  = sc.read_h5ad(os.path.join(args.savedir, f"target_preprocessed.h5ad"))

            if args.method == 'SpatialID':
                from SpatialID.run_benchmark import SpatialID_main
                SpatialID_main(args, adata_reference, adata_target)
            else:
                from Tangram.run_benchmark import Tangram_main
                Tangram_main(args, adata_reference, adata_target)

        elif args.method == 'NovAST':
            # NovAST is an installed package: its run_NovAST builds its own
            # dataset from the stage-1 `*_normalized.h5ad`, so no loader_output.
            from NovAST_py.run_benchmark import NovAST_main
            NovAST_main(args)

        else: # STELLAR or scBOL -- graph methods fed by the shared loader_output
            loader_out = pickle.load(open(os.path.join(args.savedir, "loader_output.pkl"), "rb"))

            if args.method == "STELLAR":
                if args.training_type == "st_to_st":
                    from STELLAR_py.run_benchmark import STELLAR_main
                    dataset, labeled_y, unlabeled_y = load_complete_dataset(loader_out, args.training_type, graph=True)
                    STELLAR_main(args, dataset, labeled_y, unlabeled_y)
                else:
                    print("The reference is scRNA-seq, STELLAR cannot run!")

            elif args.method == "scBOL":
                if args.training_type == "st_to_st":
                    from scBOL.run_benchmark import scBOL_main
                    dataset, labeled_y, unlabeled_y = load_complete_dataset(loader_out, args.training_type, graph=True)
                    # Map scBOL-specific args
                    args.nbr = args.scbol_nbr
                    args.quantile = args.scbol_quantile
                    scBOL_main(args, dataset, labeled_y, unlabeled_y)
                else:
                    print("The reference is scRNA-seq, scBOL cannot run (requires spatial graph)!")
        return

    if args.stage == "stellar_add":
        # load previously saved paramters
        with open(os.path.join(args.savedir, 'preprocess_args.json')) as f:
            pre_args = json.load(f)
        
        for k, v in pre_args.items():
            # skip anything you're explicitly overriding in run
            if k in ["stage", "method", "num_seed_class"]:  
                continue
            setattr(args, k, v)
        
        # previousy saved the device to str and now convert it back
        if isinstance(args.device, str):
            args.device = torch.device(args.device)

        loader_out = pickle.load(open(os.path.join(args.savedir, "loader_output.pkl"), "rb"))

        from STELLAR_py.run_benchmark import STELLAR_main
        dataset, labeled_y, unlabeled_y = load_complete_dataset(loader_out, args.training_type, graph=True)
        STELLAR_main(args, dataset, labeled_y, unlabeled_y)

        return

if __name__ == '__main__':
    main()
    # Force-exit to avoid the interpreter hanging on TF/DataLoader-worker
    # cleanup after all real work has completed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
