import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

def detect_spatial_info(adata):
    """
    Check whether an AnnData object contains spatial coordinates.
    Returns:
        has_spatial (bool)
        location (str or tuple)
            Possible values:
              - ("obsm", "spatial")
              - ("obsm", "X_spatial")
              - ("obs", ("x", "y"))
              - ("obs", ("x_coord", "y_coord"))
              - None  (if no spatial found)
    """
    if "spatial" in adata.obsm:
        return True, ("obsm", "spatial")

    if "X_spatial" in adata.obsm:
        return True, ("obsm", "X_spatial")

    if {"x", "y"}.issubset(adata.obs.columns):
        return True, ("obs", ("x", "y"))

    if {"x_coord", "y_coord"}.issubset(adata.obs.columns):
        return True, ("obs", ("x_coord", "y_coord"))
    return False, None

def get_spatial_coords(adata, pos_key=None):
    """
    Return an (n_cells × 2) array of spatial coordinates from `adata`, by checking in this order:
      1. If pos_key is given and exists in adata.obsm, use adata.obsm[pos_key].
      2. If "spatial" exists in adata.obsm, use adata.obsm["spatial"].
      3. If "X_spatial" exists in adata.obsm, use adata.obsm["X_spatial"].
      4. If both "x" and "y" appear in adata.obs.columns, return adata.obs[["x","y"]].values.
      5. If both "x_coord" and "y_coord" appear in adata.obs.columns, return adata.obs[["x_coord","y_coord"]].values.
    Otherwise, raise an error.
    """
    # 1) If user supplied a pos_key override, try that first
    if pos_key is not None and pos_key in adata.obsm:
        coords = adata.obsm[pos_key]
        if coords.shape[1] >= 2:
            return np.asarray(coords)[:, :2]
        else:
            raise ValueError(f"adata.obsm[{pos_key!r}] exists but has shape {coords.shape}, not at least 2 columns.")

    # 2) Check the most common obsm keys
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]
        if coords.shape[1] >= 2:
            return np.asarray(coords)[:, :2]
    if "X_spatial" in adata.obsm:
        coords = adata.obsm["X_spatial"]
        if coords.shape[1] >= 2:
            return np.asarray(coords)[:, :2]

    # 3) Fall back to obs‐columns "x" & "y"
    if {"x", "y"}.issubset(adata.obs.columns):
        return adata.obs[["x", "y"]].values

    # 4) Or obs‐columns "x_coord" & "y_coord"
    if {"x_coord", "y_coord"}.issubset(adata.obs.columns):
        return adata.obs[["x_coord", "y_coord"]].values

    # 5) Nothing found
    raise ValueError(
        "No spatial coordinates found in AnnData. "
        "Tried obsm keys: "
        f"{['spatial', 'X_spatial'] + ([pos_key] if pos_key else [])}. "
        "Also looked for obs columns ['x','y'] or ['x_coord','y_coord']."
    )

def get_edge_index_knn(pos, k):
    """
    Construct edge indices by connecting each point to its k nearest neighbors (excluding itself).
    Returns a list of [i, j] pairs, as well as the reverse [j, i] to mimic an undirected edge set.
    """
    n = pos.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(pos) # k+1 neighbors including itself
    distances, indices = nbrs.kneighbors(pos)  # distances.shape = (n, k+1), indices.shape = (n, k+1)
    
    sources = []
    targets = []
    for i in range(n):
        for nei in indices[i][1:]:   # skip the “self” neighbor
            sources.append(i)
            targets.append(nei)
            sources.append(nei)
            targets.append(i)

    return [sources, targets]


def get_edge_index_knn_region(pos, regions, k):
    """
    Construct edge indices by using k-NN within each region separately:
    - For each region, compute the k nearest neighbors among the points that belong to that region.
    - Map the regional neighbor indices back to the global index space.
    - Return a combined list of [i, j] edges (plus [j, i]) for all regions.
    """
    edge_list = []
    regions = np.asarray(regions)
    unique_regions = np.unique(regions)
    
    for reg in unique_regions:
        region_indices = np.where(regions == reg)[0]
        pos_region = pos[region_indices, :]
        
        if pos_region.shape[0] <= k: # if a region contains less than k points, make it fully conencted)
            for idx_i in range(pos_region.shape[0]):
                for idx_j in range(idx_i + 1, pos_region.shape[0]):
                    i_global = region_indices[idx_i]
                    j_global = region_indices[idx_j]
                    edge_list.append([i_global, j_global])
                    edge_list.append([j_global, i_global])
            continue
        
        # knn otherwise
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(pos_region)
        distances, indices = nbrs.kneighbors(pos_region)
        
        sources = []
        targets = []
        for i_local in range(pos_region.shape[0]):
            i_global = region_indices[i_local]
            for neighbor_local in indices[i_local][1:]:  # skip self
                j_global = region_indices[neighbor_local]
                sources.append(i_global); targets.append(j_global)
                sources.append(j_global); targets.append(i_global)

    return [sources, targets]

def calculate_optimal_accuracy_final(true_labels, cluster_labels, inverse_dict):
    """
    Compute optimal mapping between predicted clusters and true labels,
    using Hungarian assignment when possible and greedy matching otherwise.
    """
    true_set = set(true_labels)
    cluster_set = set(cluster_labels)

    # === Case 1: Perfect match, no mapping required ===
    if true_set == cluster_set:
        print("Perfect match: returning original labels.")
        return np.array(cluster_labels), {}

    clusters_to_map = {item for item in cluster_set if item not in set(inverse_dict.values())}
    true_labels_to_map = true_set - cluster_set

    if len(clusters_to_map) == 0:
        print("No cluster needs mapping — returning original labels.")
        return np.array(cluster_labels), {}

    if len(true_labels_to_map) == 0:
        print("No true novel to map — returning original labels.")
        return np.array(cluster_labels), {}
    
    all_labels = list(true_labels_to_map.union(clusters_to_map))
    confusion = confusion_matrix(true_labels, cluster_labels, labels=all_labels)
    
    if len(clusters_to_map) == len(true_labels_to_map):
        row_ind, col_ind = linear_sum_assignment(confusion, maximize=True)
        mapping = {all_labels[col]: all_labels[row] for row, col in zip(row_ind, col_ind)}
    else:
        mapping = {}
        assigned_clusters = set()
        for true_label in true_labels_to_map:
            true_label_index = all_labels.index(true_label)
            overlaps = confusion[true_label_index, :]
            for cluster_label in clusters_to_map:
                cluster_index = all_labels.index(cluster_label)
                if cluster_label not in assigned_clusters:
                    if overlaps[cluster_index] == overlaps.max():
                        mapping[cluster_label] = true_label
                        assigned_clusters.add(cluster_label)
                        break

    print("Mapping relationship:", mapping)
    mapped_cluster_labels = [mapping.get(label, label) for label in cluster_labels]
    return np.array(mapped_cluster_labels), mapping
    
def confidence_score(adata_all, args):
    """
    Compute exploratory confidence scores for unlabeled cells using:
      (1) kNN neighborhood heterogeneity
      (2) Distance to class centroids (Gaussian kernel)
    """
    # Extract UMAP coordinates and mask labeled / unlabeled cells
    umap_coords = adata_all.obsm['X_umap']
    labeled_mask = adata_all.obs['labeled_or_not'] == 'labeled'
    unlabeled_mask = adata_all.obs['labeled_or_not'] == 'unlabeled'
    ground_truth = adata_all.obs['ground_truth'][labeled_mask].values

    # k nearest neighbors
    labeled_coords = umap_coords[labeled_mask]
    nn_model = NearestNeighbors(n_neighbors=args.k).fit(labeled_coords)
    unlabeled_coords = umap_coords[unlabeled_mask]
    distances, indices = nn_model.kneighbors(unlabeled_coords)
    # heterogeneity scores 
    heterogeneity_scores = []
    for neighbor_indices in indices:
        neighbor_types = ground_truth[neighbor_indices]
        type_counts = pd.Series(neighbor_types).value_counts()
        heterogeneity = 1/len(type_counts[type_counts != 0].unique())
        heterogeneity_scores.append(heterogeneity)
    heterogeneity_scores = np.array(heterogeneity_scores)
    heterogeneity_scores = (heterogeneity_scores - heterogeneity_scores.min()) / (
        heterogeneity_scores.max() - heterogeneity_scores.min()
    )

    # centroid distance weighted by Gaussain kernel
    centroids = {cls: labeled_coords[ground_truth == cls].mean(axis=0) for cls in np.unique(ground_truth)}
    kernel_scores = []
    for cell_coord in unlabeled_coords:
        distances_to_centroids = [np.linalg.norm(cell_coord - centroid) for centroid in centroids.values()]
        kernel_weights = np.exp(-np.array(distances_to_centroids)**2 / (2 * args.sigma**2))
        kernel_scores.append(sum(kernel_weights))
    kernel_scores = np.array(kernel_scores)
    kernel_scores = (kernel_scores - kernel_scores.min()) / (kernel_scores.max() - kernel_scores.min())

    # combine score and plot
    combined_scores = 0.5 * heterogeneity_scores + 0.5 * kernel_scores
    adata_all.obs.loc[unlabeled_mask, 'combined_score'] = combined_scores
    unlabeled_data = adata_all.obs[unlabeled_mask]
    return adata_all, unlabeled_data, unlabeled_mask


def conf_GMM(unlabeled_data, args):
    # fit GMM at n_component of 8
    scores = unlabeled_data['combined_score'].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=args.gmm_ncomponent, random_state=42)
    gmm.fit(scores)
    clusters = gmm.predict(scores)
    cluster_means = gmm.means_.flatten()
    novel_cluster = np.argmin(cluster_means) 
    gmm_mean_min = np.min(cluster_means)
    unlabeled_data['predicted_novel'] = (clusters == novel_cluster)
    return unlabeled_data, gmm_mean_min

import numpy as np
import hnswlib
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import ConvergenceWarning
import warnings

class FastLabelSpreading(BaseEstimator, ClassifierMixin):
    def __init__(self, k=10, alpha=0.2, max_iter=30, tol=1e-3):
        self.k = k
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def _build_graph_hnsw(self, X):
        n_samples, n_features = X.shape
        p = hnswlib.Index(space='l2', dim=n_features)
        p.init_index(max_elements=n_samples, ef_construction=400, M=16)
        p.add_items(X)
        p.set_ef(100)

        labels, _ = p.knn_query(X, k=self.k)

        rows = np.repeat(np.arange(n_samples), self.k)
        cols = labels.flatten()
        data = np.ones(len(rows))

        adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples)).tocsr()
        return adj

    def fit(self, X, y):
        # X, y = validate_data(self, X, y, accept_sparse=False, ensure_2d=True, dtype=float)
        X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True, dtype=float)

        self.X_ = X
        check_classification_targets(y)

        self.classes_ = np.unique(y[y != -1])
        n_samples, n_classes = X.shape[0], len(self.classes_)

        self.label_distributions_ = np.zeros((n_samples, n_classes))
        for idx, label in enumerate(self.classes_):
            self.label_distributions_[y == label, idx] = 1

        y_static = self.label_distributions_ * (1 - self.alpha)

        unlabeled_mask = (y == -1)

        graph = self._build_graph_hnsw(X)

        l_previous = np.zeros_like(self.label_distributions_)

        for self.n_iter_ in range(self.max_iter):
            if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
                break

            l_previous = self.label_distributions_.copy()
            self.label_distributions_ = safe_sparse_dot(graph, self.label_distributions_)
            self.label_distributions_ = self.alpha * self.label_distributions_ + y_static

        else:
            warnings.warn(
                "max_iter=%d was reached without convergence." % self.max_iter,
                category=ConvergenceWarning,
            )
            self.n_iter_ += 1

        self.label_distributions_ /= np.sum(self.label_distributions_, axis=1, keepdims=True)
        self.transduction_ = self.classes_[np.argmax(self.label_distributions_, axis=1)]

        return self

    def predict(self, X):
        check_is_fitted(self)
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self)
        weight_matrix = self._build_graph_hnsw(X)
        probs = safe_sparse_dot(weight_matrix, self.label_distributions_)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs
    

def color_dict(combined_labels, colormap_dict=None):
    """
    Generate a color mapping for labels.
    - If colormap_dict is provided but missing colors for some labels,
      assign new colors from a predefined palette.
    - If the palette does not have enough colors, raise an error.
    """
    tab20 = sns.color_palette("tab20")
    set3  = sns.color_palette("Set3")
    dark2  = sns.color_palette("Dark2")
    paired  = sns.color_palette("Paired")
    combined_palette = tab20 + set3 + dark2 + paired
    unique_palette = list(dict.fromkeys(map(tuple, combined_palette)))

    if colormap_dict is None:
        if len(unique_palette) < len(combined_labels):
            raise ValueError("Not enough unique colors to assign for all labels.")
        return dict(zip(combined_labels, unique_palette[:len(combined_labels)]))

    final_cmap = dict(colormap_dict)

    # if colormap_dict provide, check if all the labels are assigned with colors, and if not, assign manually.
    missing_labels = [lb for lb in combined_labels if lb not in final_cmap]
    used_colors = set(map(tuple, final_cmap.values()))
    available_colors = [c for c in unique_palette if tuple(c) not in used_colors]
    if len(available_colors) < len(missing_labels):
        raise ValueError(
            f"Not enough remaining colors for missing labels. "
            f"Needed {len(missing_labels)}, available {len(available_colors)}."
        )
    for lb, col in zip(missing_labels, available_colors):
        final_cmap[lb] = col
    return final_cmap