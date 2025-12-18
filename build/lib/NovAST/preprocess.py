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

def preprocess(adata_train, adata_test, filedic_train, filedic_test, hvg=None, target_sum=1e4):
    """
    If preprocessed files already exist, load and return them.
    Otherwise, perform preprocessing, save to new files, and return the subsets.
    """
    # Construct output paths
    output_path_train = filedic_train.replace(".h5ad", "_full_preprocess.h5ad")
    output_path_test = filedic_test.replace(".h5ad", "_full_preprocess.h5ad")

    # Only keep the overlapp gene 
    genes_train = set(adata_train.var_names)
    genes_test = set(adata_test.var_names)
    overlapping_genes = genes_train.intersection(genes_test)
    print(f'Number of overlapped genes: {len(overlapping_genes)}')
    adata_train_subset = adata_train[:, adata_train.var_names.isin(overlapping_genes)]
    adata_test_subset = adata_test[:, adata_test.var_names.isin(overlapping_genes)]

    if hvg:
        # select hvg for each dataset separately then take the intersect
        sc.pp.highly_variable_genes(adata_train_subset, flavor='seurat_v3', n_top_genes=int(hvg))
        sc.pp.highly_variable_genes(adata_test_subset, flavor='seurat_v3', n_top_genes=int(hvg))
        hvg_list_train = np.array(adata_train_subset.var[adata_train_subset.var['highly_variable']].index.tolist())
        hvg_list_test = np.array(adata_test_subset.var[adata_test_subset.var['highly_variable']].index.tolist())
        hvg_intersect = np.intersect1d(hvg_list_train, hvg_list_test)
        print(f"Top {hvg} highly variable genes have been selected!", flush=True)
        print(f"Number of genes inversect is {len(hvg_intersect)}", flush=True)

    # Preserve raw counts in a dedicated layer
    adata_train_subset.layers["counts"] = adata_train_subset.X.copy()
    adata_test_subset.layers["counts"]  = adata_test_subset.X.copy()
    
    # normalize
    normalize_log_scale(adata_train_subset, target_sum=target_sum)
    normalize_log_scale(adata_test_subset,  target_sum=target_sum)

    # then subset again if hvg selected
    if hvg: 
        adata_train_subset = adata_train_subset[:, adata_train_subset.var_names.isin(hvg_intersect)]
        adata_test_subset = adata_test_subset[:, adata_test_subset.var_names.isin(hvg_intersect)] 

    adata_train_subset.layers["normalized"] = adata_train_subset.X.copy()
    adata_test_subset.layers["normalized"]  = adata_test_subset.X.copy()
    
    # save the dataset
    adata_train_subset.write_h5ad(output_path_train)
    adata_test_subset.write_h5ad(output_path_test)

    return adata_train_subset, adata_test_subset