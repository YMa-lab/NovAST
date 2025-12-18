import os, pickle
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
)
from .utils import color_dict, calculate_optimal_accuracy_final

def plot_umap(adata_all, adata_unlabeled, save_path, ground_truth=False, colormap=None):
    """
    Generate a multi-panel UMAP visualization summarizing the NovAST pipeline.
    Panels include:
        (1) Latent space colored by DataType (Reference vs Target)
        (2) Label Spreading assignments
        (3) Confidence score (combined heuristic)
        (4) Final predicted labels
        (5) Ground-truth labels (optional; only if ground_truth=True)
        (6) Combined legend showing all cell types used across panels
    """
    n_cols = 5 if ground_truth else 4
    width_ratios = [1] * n_cols + [1.5]
    fig = plt.figure(figsize=(30, n_cols + 1))
    gs  = gridspec.GridSpec(1, n_cols + 1, width_ratios=width_ratios, wspace=0.1)

    # make palette and colors
    preds = adata_unlabeled.obs['voted_final_prediction'].astype(str).unique()
    if ground_truth:
        gts = adata_unlabeled.obs['ground_truth'].astype(str).unique()
        combined_labels = sorted(set(preds) | set(gts))
    else:
        combined_labels = sorted(set(preds))
    colormap_dict = color_dict(combined_labels, colormap_dict=colormap)
    handles = [plt.Line2D([], [], color=colormap_dict[ct], marker="o", linestyle="", markersize=8) 
           for ct in combined_labels]

    # --- Plot 1: DataType (Reference vs Target) ----------------------
    ax0 = fig.add_subplot(gs[0, 0])
    perm = np.random.permutation(adata_all.n_obs)
    ad_shuf = adata_all[perm].copy()
    ad_shuf.obs['DataType'] = ad_shuf.obs['labeled_or_not']\
        .map(lambda x: 'Reference' if x=='labeled' else 'Target')
    palette_dt = ['#0b3954','#bfd7ea']
    sc.pl.umap(ad_shuf, color='DataType', palette=palette_dt,
            ax=ax0, show=False, sort_order=False, size=2, title='Latent Space')
    ax0.set_box_aspect(1/1); ax0.axis('off')
    handles0, labels0 = ax0.get_legend_handles_labels()
    ax0.legend(handles0, labels0, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)

    # --- Plot 2: Label Spreading -------------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    lbls = adata_unlabeled.obs['label_prop'].astype(str).unique()
    subpal = {l: colormap_dict[l] for l in lbls}
    sc.pl.umap(adata_unlabeled, color='label_prop', palette=subpal,
            ax=ax1, show=False, legend_loc=None,
            size=2, title='Label Spreading')
    ax1.set_box_aspect(1/1); ax1.axis('off')

    # --- Plot 3: Combined score --------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    sc.pl.umap(adata_unlabeled, color='combined_score', ax=ax2, show=False, sort_order=False,
            legend_loc=None, colorbar_loc=None, cmap='magma', vmin=0, vmax=1,
            size=2, title='Confidence Score')
    ax2.set_box_aspect(1/1); ax2.axis('off')
    pos = ax2.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.08, pos.width, 0.02])
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap="magma", norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal',
                        ticks=np.linspace(0,1,6), format='%.1f')
    cbar.ax.tick_params(labelsize=14)

    # --- Plot 4: Prediction ------------------------------------------
    ax3 = fig.add_subplot(gs[0, 3])
    preds = adata_unlabeled.obs['voted_final_prediction'].astype(str).unique()
    subpal = {p: colormap_dict[p] for p in preds}
    sc.pl.umap(adata_unlabeled, color='voted_final_prediction', palette=subpal,
            ax=ax3, show=False, legend_loc=None, size=2, title='Final Prediction')
    ax3.set_box_aspect(1/1); ax3.axis('off')

    # --- Plot 5: Ground Truth ----------------------------------------
    if ground_truth:
        ax4 = fig.add_subplot(gs[0, 4])
        gts = adata_unlabeled.obs['ground_truth'].astype(str).unique()
        subpal = {g: colormap_dict[g] for g in gts}
        sc.pl.umap(adata_unlabeled, color='ground_truth', palette=subpal,
                ax=ax4, show=False, legend_loc=None,
                size=2, title='Ground Truth')
        ax4.set_box_aspect(1/1); ax4.axis('off')

    # --- Plot 6: Legend ----------------------------------------------
    ax5 = fig.add_subplot(gs[0, n_cols])
    ax5.axis('off')
    ax5.legend(
        handles        = handles,
        labels         = combined_labels,
        title          = "Cell Type",
        loc            = "center",
        frameon        = False,
        fontsize       = 12,
        title_fontsize = 16,
        markerscale    = 1.5,
        ncol           = 4,
        handletextpad  = 0.1,
        columnspacing  = 0.3,
        handlelength   = 1.0,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'NovAST_umap.pdf'), dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(save_path, f'NovAST_umap.png'), dpi=600, bbox_inches='tight')
    plt.show() 

def plot_spatial(args, adata, save_path, region_key=None, ground_truth=False, colormap=None):
    """
    Generate a multi-panel UMAP visualization summarizing the NovAST pipeline.
    Panels include:
        (1) Latent space colored by DataType (Reference vs Target)
        (2) Label Spreading assignments
        (3) Confidence score (combined heuristic)
        (4) Final predicted labels
        (5) Ground-truth labels (optional; only if ground_truth=True)
        (6) Combined legend showing all cell types used across panels
    """
    save_path = os.path.join(save_path, 'NovAST_spatial')
    os.makedirs(save_path, exist_ok=True)
    loc_type, key = args.test_spatial_loc
    region_key = args.region_name_test
    spot_size = args.spot_size
    

    def _plot(ad, spot_size, title_suffix=""):
        n_cols = 4 if ground_truth else 3
        width_ratios = [1] * n_cols + [1.5]
        fig = plt.figure(figsize=(30, n_cols + 1))
        gs  = gridspec.GridSpec(1, n_cols + 1, width_ratios=width_ratios, wspace=0.1)
        if loc_type == "obsm":
            coords = ad.obsm[key]
        elif loc_type == "obs":
            x_col, y_col = key
            coords = ad.obs[[x_col, y_col]].to_numpy()

        # make palette and colors
        preds = ad.obs['voted_final_prediction'].astype(str).unique()
        if ground_truth:
            gts = ad.obs['ground_truth'].astype(str).unique()
            combined_labels = sorted(set(preds) | set(gts))
        else:
            combined_labels = sorted(set(preds))
        colormap_dict = color_dict(combined_labels, colormap_dict=colormap)
        handles = [plt.Line2D([], [], color=colormap_dict[ct], marker="o", linestyle="", markersize=8) 
            for ct in combined_labels]

        # --- Plot 1: Prediction ------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        pred_labels = ad.obs['voted_final_prediction'].astype(str)
        colors = pred_labels.map(colormap_dict)
        ax1.scatter(coords[:,0], coords[:,1], c=colors, s=args.spot_size, linewidths=0, alpha=0.9)
        ax1.set_title('Final Prediction', fontsize=26, pad=20)
        ax1.set_box_aspect(1/1); ax1.axis('off')

        # --- Plot 5: Ground Truth ----------------------------------------
        if ground_truth:
            ax2 = fig.add_subplot(gs[0, 1])
            gt_labels = ad.obs['ground_truth'].astype(str)
            gt_colors = gt_labels.map(colormap_dict)
            ax2.scatter(coords[:,0], coords[:,1], c=gt_colors, s=args.spot_size, linewidths=0, alpha=0.9)
            ax2.set_title('Ground Truth', fontsize=26, pad=20)
            ax2.set_box_aspect(1/1); ax2.axis('off')

        # --- Plot 6: Legend ----------------------------------------------
        ax3 = fig.add_subplot(gs[0, n_cols-2])
        ax3.axis('off')
        ax3.legend(
            handles        = handles,
            labels         = combined_labels,
            title          = "Cell Type",
            loc            = "center",
            frameon        = False,
            fontsize       = 12,
            title_fontsize = 16,
            markerscale    = 1.5,
            ncol           = 4,
            handletextpad  = 0.1,
            columnspacing  = 0.3,
            handlelength   = 1.0,
        )

        # --- Plot 3: Combined score --------------------------------------
        ax4 = fig.add_subplot(gs[0, n_cols-1])
        ax4.scatter(coords[:,0], coords[:,1], c=ad.obs["combined_score"], cmap="magma", s=spot_size, linewidths=0, alpha=0.9)
        ax4.set_title('Confidence Score', fontsize=26, pad=20)
        ax4.set_box_aspect(1/1); ax4.axis('off')
        pos = ax4.get_position()
        cax = fig.add_axes([pos.x0, pos.y0 - 0.08, pos.width, 0.02])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(cmap="magma", norm=norm)
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal',
                            ticks=np.linspace(0,1,6), format='%.1f')
        cbar.ax.tick_params(labelsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'NovAST_spatial_{title_suffix}.pdf'), dpi=600, format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(save_path, f'NovAST_spatial_{title_suffix}.png'), dpi=600, bbox_inches='tight')
        plt.show() 

    if region_key:
        for region in adata.obs[region_key].unique():
            subset = adata[adata.obs[region_key] == region]
            _plot(subset, spot_size, title_suffix=str(region))
    else:
        _plot(adata, spot_size)
    
def gather_metrics(savedir, dataset, max_seed=10):
    """
    Compute evaluation metrics (accuracy, weighted F1, ARI, macro F1)
    across multiple seeds for a given dataset.
    """
    with open(os.path.join(savedir, "inverse_dict_train.pkl"), "rb") as f:
        inverse_dict = pickle.load(f)

    total_accuracy    = {}
    total_weighted_F1 = {}
    total_ari         = {}
    total_macro_F1    = {}

    for i in range(1, max_seed+1):
        seed_key = f"seed_{i}"
        ad = sc.read_h5ad(os.path.join(savedir, f"seed{i}/adata_unlabeled_final.h5ad"))
        preds = ad.obs["voted_final_prediction"].to_numpy()
        gts   = ad.obs["ground_truth"].to_numpy()

        # remap labels
        preds = np.array([inverse_dict.get(x, x) for x in preds], dtype=str)
        preds, _ = calculate_optimal_accuracy_final(gts, preds, inverse_dict)

        # compute metrics
        acc  = float((gts == preds).mean())
        wf1  = float(f1_score(gts, preds, average="weighted"))
        ari  = float(adjusted_rand_score(gts, preds))
        mF1  = float(f1_score(gts, preds, average="macro"))

        total_accuracy[seed_key]      = acc
        total_weighted_F1[seed_key]   = wf1
        total_ari[seed_key]           = ari
        total_macro_F1[seed_key]      = mF1

    df = pd.DataFrame({
        "seed": list(total_accuracy.keys()),
        "accuracy": list(total_accuracy.values()),
        "weighted_F1": list(total_weighted_F1.values()),
        "ARI": list(total_ari.values()),
        "macro_F1": list(total_macro_F1.values())
    })

    csv_path = os.path.join(savedir, f"metrics_summary.csv")
    df.to_csv(csv_path, index=False)

    print("\n========== Mean Metrics Across Seeds ==========")
    print(df.mean(numeric_only=True).round(4))
    print("==============================================\n")

    return df
