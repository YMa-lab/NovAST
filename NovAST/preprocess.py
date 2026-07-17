import scanpy as sc
import numpy as np
import scipy.sparse as sp

def normalize_log_scale(adata, target_sum, int_tol=1e-6):
    """
    If adata.X is integer counts, do normalize_total + log1p, 
    then always do scale(); otherwise just scale().
    """
    # pull out the raw array
    X = adata.X
    vals = X.data if sp.issparse(X) else X

    # if likely to be an integer, do normalize + log1p
    if np.allclose(vals, np.round(vals), atol=int_tol):
        # print("take log first")
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

    # always scale (zero_center, no clipping)
    sc.pp.scale(adata, zero_center=True, max_value=None, copy=False)

def preprocess(adata_reference, adata_target, filedic_reference, filedic_target, hvg=None, target_sum=1e4):
    """
    If preprocessed files already exist, load and return them.
    Otherwise, perform preprocessing, save to new files, and return the subsets.
    """
    # Construct output paths
    output_path_reference = filedic_reference.replace(".h5ad", "_full_preprocess.h5ad")
    output_path_target = filedic_target.replace(".h5ad", "_full_preprocess.h5ad")

    # Only keep the overlapp gene 
    genes_reference = set(adata_reference.var_names)
    genes_target = set(adata_target.var_names)
    overlapping_genes = genes_reference.intersection(genes_target)
    print(f'Number of overlapped genes: {len(overlapping_genes)}')
    adata_reference_subset = adata_reference[:, adata_reference.var_names.isin(overlapping_genes)]
    adata_target_subset = adata_target[:, adata_target.var_names.isin(overlapping_genes)]

    if hvg:
        # select hvg for each dataset separately then take the intersect
        sc.pp.highly_variable_genes(adata_reference_subset, flavor='seurat_v3', n_top_genes=int(hvg))
        sc.pp.highly_variable_genes(adata_target_subset, flavor='seurat_v3', n_top_genes=int(hvg))
        hvg_list_reference = np.array(adata_reference_subset.var[adata_reference_subset.var['highly_variable']].index.tolist())
        hvg_list_target = np.array(adata_target_subset.var[adata_target_subset.var['highly_variable']].index.tolist())
        hvg_intersect = np.intersect1d(hvg_list_reference, hvg_list_target)
        print(f"Top {hvg} highly variable genes have been selected!", flush=True)
        print(f"Number of genes inversect is {len(hvg_intersect)}", flush=True)

    # Preserve raw counts in a dedicated layer
    adata_reference_subset.layers["counts"] = adata_reference_subset.X.copy()
    adata_target_subset.layers["counts"]  = adata_target_subset.X.copy()
    
    # normalize
    normalize_log_scale(adata_reference_subset, target_sum=target_sum)
    normalize_log_scale(adata_target_subset,  target_sum=target_sum)

    # then subset again if hvg selected
    if hvg: 
        adata_reference_subset = adata_reference_subset[:, adata_reference_subset.var_names.isin(hvg_intersect)]
        adata_target_subset = adata_target_subset[:, adata_target_subset.var_names.isin(hvg_intersect)] 

    adata_reference_subset.layers["normalized"] = adata_reference_subset.X.copy()
    adata_target_subset.layers["normalized"]  = adata_target_subset.X.copy()
    
    # save the dataset
    adata_reference_subset.write_h5ad(output_path_reference)
    adata_target_subset.write_h5ad(output_path_target)

    return adata_reference_subset, adata_target_subset