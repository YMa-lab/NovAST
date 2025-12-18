
import pickle 
import os
import numpy as np
import torch
import time
import pandas as pd
import scanpy as sc
import re
from sklearn.cluster import DBSCAN
from .train import NovAST_train
from .datasets import *
from .utils import *

def load_paired_dataset(adata_train, adata_test, args):
    # load datasets
    outputs = load_dataset_adata(args, adata_train, adata_test)
    (labeled_X, labeled_y, inverse_dict_train, unlabeled_X, unlabeled_y, unlabeled_y_c, inverse_dict_test, adata_train, adata_test) = outputs
    dataset = Dataset(labeled_X, labeled_y, unlabeled_X, unlabeled_y_c)
    
    with open(os.path.join(args.savedir, 'inverse_dict_train.pkl'), 'wb') as f:
        pickle.dump(inverse_dict_train, f)

    if args.uncontrolled == False:
        with open(os.path.join(args.savedir, 'inverse_dict_test.pkl'), 'wb') as f:
            pickle.dump(inverse_dict_test, f)

    return dataset, labeled_y, unlabeled_y, unlabeled_X, adata_train, adata_test

def training_model(seed, args, dataset, labeled_y, unlabeled_y, filepath):
    start_time = time.time()
    print(f"Part I training started! Round {seed}")
    novaset = NovAST_train(args, dataset)
    z_train, z_test, total_losses, recons_losses, sim_losses, mmf_losses, model_cpu = novaset.train()
    torch.save(model_cpu, os.path.join(filepath, 'model.pth'))
    
    np.savetxt(os.path.join(filepath, 'total_losses.csv'), total_losses, delimiter=",")
    np.savetxt(os.path.join(filepath, 'recons_losses.csv'), recons_losses, delimiter=",")
    np.savetxt(os.path.join(filepath, 'sim_losses.csv'), sim_losses, delimiter=",")
    np.savetxt(os.path.join(filepath, 'mmf_losses.csv'), mmf_losses, delimiter=",")
    
    print("Part I training done",flush=True)
    print("--- %s seconds ---" % (time.time() - start_time),flush=True)

    return labeled_y, z_train, unlabeled_y, z_test

def identify_novel_cells_evaluation(labeled_y, z_train, unlabeled_y, z_test, args, filepath, adata_train, adata_test):
    start_time = time.time()
    print("Step2 started!", flush=True)

    # get combined latent space, transform it to an anndata, and add feautres (i.e., labled or not)
    adata_all = sc.concat(
        [adata_train, adata_test],
        join="outer",            # keep all obs columns (safe)
        axis=0,                  # stack rows (cells)
        label=None,              # no batch label added automatically
        index_unique=None        # keep original indices
    )
    z_all = np.concatenate((z_train, z_test), axis=0)
    unlabeled_y_modified = np.full(unlabeled_y.shape, -1) 
    y_all = np.concatenate((labeled_y, unlabeled_y_modified))
    adata_all.obsm['X_latent'] = z_all
    n_labeled = len(labeled_y)
    n_unlabeled = len(unlabeled_y)
    labeled_or_not = ['labeled'] * n_labeled + ['unlabeled'] * n_unlabeled
    adata_all.obs['labeled_or_not'] = pd.Categorical(labeled_or_not)

    start_time2 = time.time()
    if len(y_all) < 100000:
        sc.pp.neighbors(adata_all, n_neighbors=15, use_rep='X_latent')
        sc.tl.umap(adata_all)
    else:
        print("Cells more than 100k, use the faster version!", flush=True)
        sc.pp.neighbors(adata_all, n_neighbors=30, use_rep='X_latent', method='umap')
        sc.tl.umap(
            adata_all,
            min_dist=0.3,
            spread=1.0,
            init_pos='spectral',
            random_state=42
        )
    print("UMAPing takes %s seconds" % (time.time() - start_time2),flush=True)

    ############################## labelspreading ######################################################
    # label spreading
    start_time2 = time.time()
    if len(y_all) < 100000:
        from sklearn.semi_supervised import LabelSpreading
        label_spread = LabelSpreading(kernel='knn', n_neighbors=10)
    else:
        label_spread = FastLabelSpreading()
    label_spread.fit(z_all, y_all)
    predicted_labels = label_spread.transduction_
    print("Labelspreading takes %s seconds" % (time.time() - start_time2),flush=True)

    # add labels to adata
    with open(os.path.join(args.savedir, 'inverse_dict_train.pkl'), 'rb') as f:
        inverse_dict = pickle.load(f)
    with open(os.path.join(args.savedir, 'inverse_dict_test.pkl'), 'rb') as f:
        inverse_dict_test = pickle.load(f)
    unlabeled_y_name = unlabeled_y.astype('object')
    for i in range(len(unlabeled_y_name)):
        if unlabeled_y_name[i] in inverse_dict_test.keys():
            unlabeled_y_name[i] = inverse_dict_test[unlabeled_y_name[i]]
    shared_list = set(inverse_dict.values()).intersection(inverse_dict_test.values())
    unique_list = set(inverse_dict_test.values()) - shared_list
    labeled_y_name = labeled_y.astype('object')
    for i in range(len(labeled_y_name)):
        if labeled_y_name[i] in inverse_dict.keys():
            labeled_y_name[i] = inverse_dict[labeled_y_name[i]]
    y_all_true =  np.concatenate((labeled_y_name, unlabeled_y_name))
    novel_status = []
    for y in y_all_true:
        if y in unique_list:
            novel_status.append('novel')
        else:
            novel_status.append('non-novel')
    adata_all.obs['novel_status'] = pd.Categorical(novel_status)

    # transfrom the y_all_true and predicted_label from number to names
    predicted_labels = predicted_labels.astype('object')
    for i in range(len(predicted_labels)):
        if predicted_labels[i] in inverse_dict.keys():
            predicted_labels[i] = inverse_dict[predicted_labels[i]]

    # add to anndata
    adata_all.obs['label_prop'] = predicted_labels
    adata_all.obs['label_prop'] = adata_all.obs['label_prop'].astype('category')
    adata_all.obs['ground_truth'] = pd.Categorical(y_all_true)
    adata_all.write_h5ad(os.path.join(filepath, 'adata_all_end_of_labelspreading.h5ad'))

    ############################## calculate confidence scores ###################################
    # confidence score
    adata_all, unlabeled_data, unlabeled_mask = confidence_score(adata_all, args)
    unlabeled_data, gmm_mean_min = conf_GMM(unlabeled_data, args)
    adata_unlabeled = adata_all[unlabeled_mask]
    adata_unlabeled.obs['predicted_novel'] = unlabeled_data['predicted_novel']
    adata_novel = adata_unlabeled[adata_unlabeled.obs['predicted_novel'] == True]
    sc.pp.neighbors(adata_novel, use_rep='X_umap')
    umap_coords = adata_novel.obsm['X_umap']
    dbscan = DBSCAN(eps=0.5, min_samples=args.dbscan).fit(umap_coords)
    adata_novel.obs['dbscan'] = dbscan.labels_
    adata_unlabeled.obs['dbscan_novel'] = 'known'
    adata_unlabeled.obs.loc[adata_novel.obs.index, 'dbscan_novel'] = adata_novel.obs['dbscan']

    adata_unlabeled.obs['dbscan_novel'] = adata_unlabeled.obs['dbscan_novel'].astype(str)
    adata_unlabeled.obs['dbscan_novel'] = adata_unlabeled.obs['dbscan_novel'].replace('-1', 'Noise')

    # combine the results for louvain clustering and label spreading
    predicted_labels_target = predicted_labels[labeled_y.shape[0]:]
    predicted_labels_target = predicted_labels_target.astype('object')
    for i in range(len(predicted_labels_target)):
        if predicted_labels_target[i] in inverse_dict.keys():
            predicted_labels_target[i] = inverse_dict[predicted_labels_target[i]] 
    dbscan_labels = adata_unlabeled.obs['dbscan_novel'].to_numpy()
    unique_dbscan_labels = np.unique(dbscan_labels)
    novel_mapping = {}
    novel_label_counter = 1
    for label in unique_dbscan_labels:
        if label not in ['Noise', 'known']:
            novel_mapping[label] = f"novel{novel_label_counter}"
            novel_label_counter += 1
    relabeled_dbscan = pd.Series(dbscan_labels).map(lambda x: novel_mapping[x] if x in novel_mapping else x).to_numpy()
    combined_labels = relabeled_dbscan.copy()
    for i, (dbscan_label, predicted_label) in enumerate(zip(relabeled_dbscan, predicted_labels_target)):
        if dbscan_label in ['Noise', 'known']:
            combined_labels[i] = predicted_label
    adata_unlabeled.obs['combined_labels'] = combined_labels

    # reject novel identified if gmm_mean_min > args.gmm_thresh
    adata_unlabeled.obs['gmm_mean_min'] = gmm_mean_min
    adata_unlabeled.obs['reject_novel'] = gmm_mean_min > args.gmm_thresh
    if gmm_mean_min > args.gmm_thresh:
        adata_unlabeled.obs['final_prediction'] = adata_unlabeled.obs['label_prop']
    else:
        adata_unlabeled.obs['final_prediction'] = adata_unlabeled.obs['combined_labels']
    adata_unlabeled.write_h5ad(os.path.join(filepath, 'adata_unlabeled_end_of_step2.h5ad'))
  
    print("Part II training done",flush=True)
    print("--- %s seconds ---" % (time.time() - start_time),flush=True)

    return adata_all, adata_unlabeled, predicted_labels_target, labeled_y, z_test, unlabeled_y_name, inverse_dict

def load_exist_data_evaluation(args, filepath):
    adata_all = sc.read_h5ad(os.path.join(filepath, 'adata_all_end_of_labelspreading.h5ad'))
    adata_unlabeled = sc.read_h5ad(os.path.join(filepath, 'adata_unlabeled_final.h5ad'))
    with open(os.path.join(args.savedir, 'inverse_dict_train.pkl'), 'rb') as f:
        inverse_dict = pickle.load(f)
    return adata_all, adata_unlabeled, inverse_dict


def identify_novel_cells_exploration(labeled_y, z_train, unlabeled_y, z_test, args, filepath, adata_train, adata_test):
    start_time = time.time()
    print("Step2 started!")
    # get combined latent space, transform it to an anndata, and add feautres (i.e., labled or not)
    adata_all = sc.concat(
        [adata_train, adata_test],
        join="outer",            # keep all obs columns (safe)
        axis=0,                  # stack rows (cells)
        label=None,              # no batch label added automatically
        index_unique=None        # keep original indices
    )
    z_all = np.concatenate((z_train, z_test), axis=0)
    unlabeled_y_modified = np.full(unlabeled_y.shape, -1)
    unlabeled_y_modified = unlabeled_y_modified.squeeze()
    unlabeled_y = unlabeled_y.squeeze()
    y_all = np.concatenate((labeled_y, unlabeled_y_modified))
    adata_all.obsm['X_latent'] = z_all
    sc.pp.neighbors(adata_all, n_neighbors=15, use_rep='X_latent')
    n_labeled = len(labeled_y)
    n_unlabeled = len(unlabeled_y)
    labeled_or_not = ['labeled'] * n_labeled + ['unlabeled'] * n_unlabeled
    adata_all.obs['labeled_or_not'] = pd.Categorical(labeled_or_not)

    ############################## labelspreading ######################################################
    # label spreading
    start_time2 = time.time()
    if len(y_all) < 100000:
        from sklearn.semi_supervised import LabelSpreading
        label_spread = LabelSpreading(kernel='knn', n_neighbors=10)
    else:
        label_spread = FastLabelSpreading()
    label_spread.fit(z_all, y_all)
    predicted_labels = label_spread.transduction_
    np.save(os.path.join(filepath, 'labelspreading_predictions.npy'), predicted_labels)
    print("Labelspreading takes %s seconds" % (time.time() - start_time2),flush=True)

    # add labels to adata
    with open(os.path.join(args.savedir, 'inverse_dict_train.pkl'), 'rb') as f:
        inverse_dict = pickle.load(f)
    shared_list = set(inverse_dict.values())
    labeled_y_name = labeled_y.astype('object')
    for i in range(len(labeled_y_name)):
        if labeled_y_name[i] in inverse_dict.keys():
            labeled_y_name[i] = inverse_dict[labeled_y_name[i]]
    y_all_true =  np.concatenate((labeled_y_name, unlabeled_y))

    # transfrom the y_all_true and predicted_label from number to names
    predicted_labels = predicted_labels.astype('object')
    for i in range(len(predicted_labels)):
        if predicted_labels[i] in inverse_dict.keys():
            predicted_labels[i] = inverse_dict[predicted_labels[i]]

    # add to anndata
    adata_all.obs['label_prop'] = predicted_labels
    adata_all.obs['label_prop'] = adata_all.obs['label_prop'].astype('category')
    adata_all.obs['ground_truth'] = pd.Categorical(y_all_true.astype(str))
    sc.tl.umap(adata_all)
    adata_all.write_h5ad(os.path.join(filepath, 'adata_all_end_of_labelspreading.h5ad'))

    ############################## calculate confidence scores ###################################
    # confidence score
    adata_all, unlabeled_data, unlabeled_mask = confidence_score(adata_all, args)
    unlabeled_data, gmm_mean_min = conf_GMM(unlabeled_data, args)
    adata_unlabeled = adata_all[unlabeled_mask]
    adata_unlabeled.obs['predicted_novel'] = unlabeled_data['predicted_novel']
    adata_novel = adata_unlabeled[adata_unlabeled.obs['predicted_novel'] == True]
    sc.pp.neighbors(adata_novel, use_rep='X_umap')
    umap_coords = adata_novel.obsm['X_umap']
    dbscan = DBSCAN(eps=0.5, min_samples=args.dbscan).fit(umap_coords)
    adata_novel.obs['dbscan'] = dbscan.labels_
    adata_unlabeled.obs['dbscan_novel'] = 'known'
    adata_unlabeled.obs.loc[adata_novel.obs.index, 'dbscan_novel'] = adata_novel.obs['dbscan']

    adata_unlabeled.obs['dbscan_novel'] = adata_unlabeled.obs['dbscan_novel'].astype(str)
    adata_unlabeled.obs['dbscan_novel'] = adata_unlabeled.obs['dbscan_novel'].replace('-1', 'Noise')

    # combine the results for louvain clustering and label spreading
    predicted_labels_target = predicted_labels[labeled_y.shape[0]:]
    predicted_labels_target = predicted_labels_target.astype('object')
    for i in range(len(predicted_labels_target)):
        if predicted_labels_target[i] in inverse_dict.keys():
            predicted_labels_target[i] = inverse_dict[predicted_labels_target[i]] 
    dbscan_labels = adata_unlabeled.obs['dbscan_novel'].to_numpy()
    unique_dbscan_labels = np.unique(dbscan_labels)
    novel_mapping = {}
    novel_label_counter = 1
    for label in unique_dbscan_labels:
        if label not in ['Noise', 'known']:
            novel_mapping[label] = f"novel{novel_label_counter}"
            novel_label_counter += 1
    relabeled_dbscan = pd.Series(dbscan_labels).map(lambda x: novel_mapping[x] if x in novel_mapping else x).to_numpy()
    combined_labels = relabeled_dbscan.copy()
    for i, (dbscan_label, predicted_label) in enumerate(zip(relabeled_dbscan, predicted_labels_target)):
        if dbscan_label in ['Noise', 'known']:
            combined_labels[i] = predicted_label
    adata_unlabeled.obs['combined_labels'] = combined_labels

    # reject novel identified if gmm_mean_min > args.gmm_thresh
    adata_unlabeled.obs['gmm_mean_min'] = gmm_mean_min
    adata_unlabeled.obs['reject_novel'] = gmm_mean_min > args.gmm_thresh
    if gmm_mean_min > args.gmm_thresh:
        adata_unlabeled.obs['final_prediction'] = adata_unlabeled.obs['label_prop']
    else:
        adata_unlabeled.obs['final_prediction'] = adata_unlabeled.obs['combined_labels']

    adata_unlabeled.write_h5ad(os.path.join(filepath, 'adata_unlabeled_end_of_step2.h5ad'))

    print("Part II training done",flush=True)
    print("--- %s seconds ---" % (time.time() - start_time),flush=True)

    return adata_all, adata_unlabeled, predicted_labels_target, labeled_y, z_test, inverse_dict

def load_exist_data_exploration(args, filepath):
    adata_all = sc.read_h5ad(os.path.join(filepath, 'adata_all_end_of_labelspreading.h5ad'))
    adata_unlabeled = sc.read_h5ad(os.path.join(filepath, 'adata_unlabeled_final.h5ad'))
    return adata_all, adata_unlabeled

def vote(args):
    all_predictions = []
    all_labelspread = []
    for i in range(1, args.rounds+1):
        print("Loading seed", i)
        output_path = os.path.join(args.savedir, f"seed{i}")
        adata = sc.read_h5ad(os.path.join(output_path, "adata_unlabeled_end_of_step2.h5ad"))
        predictions = adata.obs['final_prediction'].rename(f"pred_round_{i}")
        label_spreading =  adata.obs['label_prop'].rename(f"label_round_{i}")
        all_predictions.append(predictions)
        all_labelspread.append(label_spreading)

    all_predictions_df = pd.concat(all_predictions, axis=1)
    all_labelspread_df = pd.concat(all_labelspread, axis=1)
    pattern = re.compile(r"novel\d+")
    def contains_novel(col):
        return col.str.contains(pattern).any()
    
    novel_cols = all_predictions_df.apply(contains_novel)
    novel_counts = all_predictions_df.loc[:, novel_cols].apply(lambda col: col.str.contains(pattern).sum())

    for col in all_predictions_df.columns[novel_cols]:
        if novel_counts[col] < args.vote_thresh:
            all_predictions_df[col] = all_labelspread_df[col]
    for i in range(1, args.rounds+1):
        print("Saving voted seed", i)
        colname = f"pred_round_{i}"
        row_array = all_predictions_df[colname].to_numpy()
        output_path = os.path.join(args.savedir, f"seed{i}")
        adata = sc.read_h5ad(os.path.join(output_path, "adata_unlabeled_end_of_step2.h5ad"))
        adata.obs['voted_final_prediction'] = row_array
        adata.write_h5ad(os.path.join(output_path, "adata_unlabeled_final.h5ad"))

