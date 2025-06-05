import torch
import numpy as np
from tqdm.auto import tqdm
from .common_utils import logger
from .evaluation_utils import ( 
    to_tensor_eval as to_tensor, # Use the eval-specific to_tensor
    batch_pairwise_euclidean_distance,
    batch_pairwise_cosine_similarity,
    spearman_rank_correlation,
    pearson_correlation,
    _procrustes_alignment_core # The robust internal one
)

# --- Precomputation ---
def _compute_distance_matrix(X, distance_metric='cosine', device='cpu', batch_size=1024):
    X_on_device = to_tensor(X, device=device)
    if distance_metric == 'euclidean':
        return batch_pairwise_euclidean_distance(X_on_device, None, device=device, batch_size=batch_size)
    elif distance_metric == 'cosine':
        sim = batch_pairwise_cosine_similarity(X_on_device, None, device=device, batch_size=batch_size)
        return 1.0 - sim # Convert similarity to distance
    else: raise ValueError(f"Unsupported distance_metric: {distance_metric}")

def _neighbor_indices(dist_matrix, kmax):
    n = dist_matrix.shape[0]
    if kmax <= 0 or n <= 1: return torch.empty((n, 0), device=dist_matrix.device, dtype=torch.long)
    k_for_topk = min(kmax + 1, n)
    _, nn_indices = torch.topk(dist_matrix, k=k_for_topk, largest=False, sorted=True)
    return nn_indices[:, 1:] # Exclude self

def _get_unique_distances_flat(dist_matrix):
    n = dist_matrix.shape[0]
    if n <= 1: return torch.empty(0, device=dist_matrix.device, dtype=dist_matrix.dtype)
    indices = torch.triu_indices(n, n, offset=1, device=dist_matrix.device)
    return dist_matrix[indices[0], indices[1]]

def precompute_evaluation_data(
    X_orig_np, X_comp_np, distance_metric='cosine', device='cpu',
    batch_size=1024, compute_nn=True, k_max_for_nn=30,
    use_unique_distances_for_flat=True 
):
    logger.info(f"Precomputing evaluation data (N={X_orig_np.shape[0]}). Metric: {distance_metric}, k_max_nn: {k_max_for_nn}")
    X_orig_t = to_tensor(X_orig_np, device=device, dtype=torch.float32)
    X_comp_t = to_tensor(X_comp_np, device=device, dtype=torch.float32)

    if torch.isnan(X_orig_t).any() or torch.isinf(X_orig_t).any() or \
       torch.isnan(X_comp_t).any() or torch.isinf(X_comp_t).any():
        raise ValueError("Input embeddings (original or compressed) contain NaN/Inf values.")

    n_samples = X_orig_t.shape[0]
    precomp = {
        'X_tensor_orig': X_orig_t, 'X_tensor_comp': X_comp_t,
        'N': n_samples, 'kmax_computed_nn': 0, 'device': device
    }

    if n_samples == 0:
        empty_dist = torch.empty((0,0), device=device, dtype=X_orig_t.dtype)
        empty_flat = torch.empty(0, device=device, dtype=X_orig_t.dtype)
        precomp.update({'dist_orig': empty_dist, 'dist_comp': empty_dist,
                        'dist_orig_flat': empty_flat, 'dist_comp_flat': empty_flat,
                        'nn_orig': None, 'nn_comp': None})
        return precomp

    precomp['dist_orig'] = _compute_distance_matrix(X_orig_t, distance_metric, device, batch_size)
    precomp['dist_comp'] = _compute_distance_matrix(X_comp_t, distance_metric, device, batch_size)

    if use_unique_distances_for_flat:
        precomp['dist_orig_flat'] = _get_unique_distances_flat(precomp['dist_orig'])
        precomp['dist_comp_flat'] = _get_unique_distances_flat(precomp['dist_comp'])
    else: # If you need all N*N distances (e.g. for some definitions of Kruskal Stress)
        precomp['dist_orig_flat'] = precomp['dist_orig'].reshape(-1)
        precomp['dist_comp_flat'] = precomp['dist_comp'].reshape(-1)


    if compute_nn and n_samples > 1 and k_max_for_nn > 0:
        k_to_find = min(k_max_for_nn, n_samples - 1)
        if k_to_find > 0:
            precomp['nn_orig'] = _neighbor_indices(precomp['dist_orig'], k_to_find)
            precomp['nn_comp'] = _neighbor_indices(precomp['dist_comp'], k_to_find)
            precomp['kmax_computed_nn'] = k_to_find
    logger.info("Finished precomputing evaluation data.")
    return precomp

# --- Metric Functions (using precomputed data) ---

def trustworthiness_metric(precomp, k=10):
    dist_orig, nn_orig, nn_comp, n, kmax_c = (precomp['dist_orig'], precomp['nn_orig'],
                                             precomp['nn_comp'], precomp['N'], precomp['kmax_computed_nn'])
    if n == 0 or k <= 0: return 1.0
    if nn_orig is None or nn_comp is None : return float('nan') if not (n <= 1 or (kmax_c == 0 and k > 0)) else 1.0
    if k > kmax_c: logger.warning(f"Trustworthiness: k={k} > kmax_computed_nn={kmax_c}."); return float('nan')
    if n > 1 and k >= n : k = n -1 # k cannot exceed n-1 neighbors

    ranks_sum_for_intruders = 0.0
    # Precompute all ranks in the original space (0-indexed)
    # Adding small epsilon for stability in argsort if distances are identical
    ranks_in_orig_all = torch.argsort(torch.argsort(dist_orig + torch.randn_like(dist_orig) * 1e-9, dim=1, stable=True), dim=1, stable=True)

    for i in tqdm(range(n), desc="Trustworthiness", leave=False, disable=(n < 500)):
        orig_neighbors_set = set(nn_orig[i, :k].tolist())
        comp_neighbors_indices = nn_comp[i, :k]

        for neighbor_idx_in_comp_space in comp_neighbors_indices:
            if neighbor_idx_in_comp_space.item() not in orig_neighbors_set:
                original_rank_of_intruder = ranks_in_orig_all[i, neighbor_idx_in_comp_space.item()].item()
                one_indexed_rank = original_rank_of_intruder + 1
                if one_indexed_rank > k : 
                     ranks_sum_for_intruders += (one_indexed_rank - k)


    if n <= 1 or k==0 : return 1.0 
    normalization_factor = n * k * (2 * n - 3 * k - 1)
    if normalization_factor <= 0: 
        return 1.0 if ranks_sum_for_intruders == 0 else (float('nan') if ranks_sum_for_intruders > 0 else 1.0)

    penalty = (2.0 / normalization_factor) * ranks_sum_for_intruders
    return max(0.0, 1.0 - penalty)


def continuity_metric(precomp, k=10):
    # Similar structure to trustworthiness, but roles of nn_orig and nn_comp are swapped for finding lost neighbors
    dist_comp, nn_orig, nn_comp, n, kmax_c = (precomp['dist_comp'], precomp['nn_orig'],
                                             precomp['nn_comp'], precomp['N'], precomp['kmax_computed_nn'])
    if n == 0 or k <= 0: return 1.0
    if nn_orig is None or nn_comp is None: return float('nan') if not (n <= 1 or (kmax_c == 0 and k > 0)) else 1.0
    if k > kmax_c: logger.warning(f"Continuity: k={k} > kmax_computed_nn={kmax_c}."); return float('nan')
    if n > 1 and k >= n : k = n -1

    ranks_sum_for_lost = 0.0
    ranks_in_comp_all = torch.argsort(torch.argsort(dist_comp + torch.randn_like(dist_comp) * 1e-9, dim=1, stable=True), dim=1, stable=True)

    for i in tqdm(range(n), desc="Continuity", leave=False, disable=(n < 500)):
        comp_neighbors_set = set(nn_comp[i, :k].tolist())
        orig_neighbors_indices = nn_orig[i, :k]

        for neighbor_idx_in_orig_space in orig_neighbors_indices:
            if neighbor_idx_in_orig_space.item() not in comp_neighbors_set: # This neighbor is "lost"
                # Find rank of this lost neighbor in the compressed space's neighborhood of i
                compressed_rank_of_lost = ranks_in_comp_all[i, neighbor_idx_in_orig_space.item()].item()
                one_indexed_rank = compressed_rank_of_lost + 1
                if one_indexed_rank > k:
                    ranks_sum_for_lost += (one_indexed_rank - k)
    
    if n <= 1 or k == 0: return 1.0
    normalization_factor = n * k * (2 * n - 3 * k - 1)
    if normalization_factor <= 0:
        return 1.0 if ranks_sum_for_lost == 0 else (float('nan') if ranks_sum_for_lost > 0 else 1.0)
        
    penalty = (2.0 / normalization_factor) * ranks_sum_for_lost
    return max(0.0, 1.0 - penalty)

def mean_relative_rank_error_metric(precomp, k=10):
    dist_orig, dist_comp, n, kmax_c = (precomp['dist_orig'], precomp['dist_comp'],
                                     precomp['N'], precomp['kmax_computed_nn'])
    
    if n <= 1 or k <= 0: return 0.0
    if k > kmax_c : logger.warning(f"MRRE: k={k} > kmax_computed_nn={kmax_c}."); return float('nan')
    if k >= n : k = n-1
    if k == 0: return 0.0

    total_mrre_sum_for_all_points = 0.0
    
    # Get 0-indexed ranks for all points in compressed space
    all_ranks_in_comp_0idx = torch.argsort(torch.argsort(dist_comp + torch.randn_like(dist_comp) * 1e-9, dim=1, stable=True), dim=1, stable=True)

    for i in tqdm(range(n), desc="MRRE", leave=False, disable=(n < 500)):
        orig_k_neighbor_indices = torch.argsort(dist_orig[i,:] + torch.randn_like(dist_orig[i,:]) * 1e-9, stable=True)[1 : k + 1]
        
        if orig_k_neighbor_indices.numel() == 0: continue

        # Their ranks r(i,j) are simply 1, 2, ..., k
        actual_orig_ranks_1idx = torch.arange(1, orig_k_neighbor_indices.shape[0] + 1,
                                              device=dist_orig.device, dtype=torch.float32)
        
        # Ranks of these original neighbors in the *compressed* space's view of point i
        ranks_of_orig_neighbors_in_comp_0idx = all_ranks_in_comp_0idx[i, orig_k_neighbor_indices]
        ranks_of_orig_neighbors_in_comp_1idx = ranks_of_orig_neighbors_in_comp_0idx.float() + 1.0
        
        relative_errors = torch.abs(actual_orig_ranks_1idx - ranks_of_orig_neighbors_in_comp_1idx) / actual_orig_ranks_1idx
        if relative_errors.numel() > 0:
            total_mrre_sum_for_all_points += relative_errors.mean().item() # item-wise MRRE averaged

    return (total_mrre_sum_for_all_points / n) if n > 0 else 0.0


def neighbor_precision_metric(precomp, k=10):
    nn_orig, nn_comp, n, kmax_c = (precomp['nn_orig'], precomp['nn_comp'],
                                   precomp['N'], precomp['kmax_computed_nn'])
    if n == 0 or k <= 0: return 1.0
    if nn_orig is None or nn_comp is None: return float('nan') if not (n <= 1 or (kmax_c == 0 and k > 0)) else 1.0
    if k > kmax_c: logger.warning(f"NeighborPrecision: k={k} > kmax_computed_nn={kmax_c}."); return float('nan')
    
    total_precision = 0.0
    for i in range(n):
        orig_neighbors = set(nn_orig[i, :k].tolist())
        comp_neighbors = set(nn_comp[i, :k].tolist())
        intersection_size = len(orig_neighbors.intersection(comp_neighbors))
        total_precision += intersection_size / k if k > 0 else 1.0
        
    return (total_precision / n) if n > 0 else 1.0

def local_procrustes_metric(precomp, k=10, procrustes_batch_size=256):
    X_orig_t, X_comp_t, nn_orig, nn_comp, n, kmax_c, device = (
        precomp['X_tensor_orig'], precomp['X_tensor_comp'],
        precomp['nn_orig'], precomp['nn_comp'], precomp['N'],
        precomp['kmax_computed_nn'], precomp['device']
    )
    if n == 0: return 0.0
    if k <= 1: logger.warning("LocalProcrustes: k<=1 requested. Min k is 2 for alignment."); return float('nan')
    if nn_orig is None or nn_comp is None or kmax_c == 0:
        logger.warning(f"LocalProcrustes: NNs not available or kmax_c=0."); return float('nan')
    if k > kmax_c: logger.warning(f"LocalProcrustes: k={k} > kmax_c={kmax_c}."); return float('nan')

    all_disparities_list = []
    
    valid_local_X_orig_sets = []
    valid_local_X_comp_sets = []

    for i in range(n): # This loop is for collecting valid neighborhoods
        
        orig_indices = nn_orig[i, :k] # Indices of k-NN of point i in original space
        comp_indices = nn_comp[i, :k] # Indices of k-NN of point i in compressed space

        # Get the actual embeddings for these *sets of points*
        local_X_o_coords = X_orig_t[orig_indices]
        local_X_c_coords = X_comp_t[comp_indices]
        
        if local_X_o_coords.shape[0] == k and local_X_c_coords.shape[0] == k:
            if torch.linalg.matrix_rank(local_X_o_coords - local_X_o_coords.mean(dim=0)) < min(k, local_X_o_coords.shape[1]) or \
               torch.linalg.matrix_rank(local_X_c_coords - local_X_c_coords.mean(dim=0)) < min(k, local_X_c_coords.shape[1]):
                continue 

            valid_local_X_orig_sets.append(local_X_o_coords)
            valid_local_X_comp_sets.append(local_X_c_coords)

    if not valid_local_X_orig_sets:
        logger.warning("LocalProcrustes: No valid (non-degenerate) local neighborhoods found."); return float('nan')

    num_valid_points = len(valid_local_X_orig_sets)
    for idx in tqdm(range(0, num_valid_points, procrustes_batch_size), desc="Local Procrustes", leave=False, disable=num_valid_points < procrustes_batch_size*2):
        batch_X_orig = torch.stack(valid_local_X_orig_sets[idx : idx + procrustes_batch_size])
        batch_X_comp = torch.stack(valid_local_X_comp_sets[idx : idx + procrustes_batch_size])
        
        current_batch_disparities = []
        for j in range(batch_X_orig.shape[0]): # Iterate within the batch for _procrustes_alignment_core
            try:
                _, _, disp_j = _procrustes_alignment_core(
                    batch_X_orig[j], batch_X_comp[j],
                    device=device, compute_scaling=True
                )
                if not np.isnan(disp_j):
                    current_batch_disparities.append(disp_j)
            except Exception as e_item:
                logger.debug(f"Local Procrustes: Error on item {idx+j}: {e_item}")
        
        if current_batch_disparities:
            all_disparities_list.extend(current_batch_disparities)
            
    if not all_disparities_list: return float('nan')
    final_disparities_tensor = torch.tensor(all_disparities_list, device=device)
    return torch.nanmean(final_disparities_tensor).item()


def global_procrustes_metric(precomp):
    X_orig_t, X_comp_t, n, device = (
        precomp['X_tensor_orig'], precomp['X_tensor_comp'],
        precomp['N'], precomp['device']
    )
    if n == 0: return 0.0
    try:
        _, _, disparity = _procrustes_alignment_core(X_orig_t, X_comp_t, device=device, compute_scaling=True)
        return disparity
    except Exception as e:
        logger.error(f"GlobalProcrustes: Error during alignment: {e}", exc_info=True)
        return float('nan')

def kruskal_stress_metric(precomp):
    d_orig_flat, d_comp_flat, n = precomp['dist_orig_flat'], precomp['dist_comp_flat'], precomp['N']
    if n == 0 or d_orig_flat.numel() == 0: return 0.0
    d_orig_flat = d_orig_flat.to(d_comp_flat.device)
    
   
    sum_sq_diff = torch.sum((d_orig_flat - d_comp_flat).pow(2))
    sum_sq_orig = torch.sum(d_orig_flat.pow(2))

    if sum_sq_orig.item() < 1e-12: # Original distances are all zero
        return 0.0 if sum_sq_diff.item() < 1e-12 else float('inf') # Perfect if compressed also zero
    return torch.sqrt(sum_sq_diff / (sum_sq_orig + 1e-12)).item()


def spearman_distcorr_metric(precomp):
    # Pearson correlation on ranks of distances
    d_orig_flat, d_comp_flat = precomp['dist_orig_flat'], precomp['dist_comp_flat']
    if d_orig_flat.numel() < 2 or d_comp_flat.numel() < 2: return float('nan')
    return spearman_rank_correlation(d_orig_flat, d_comp_flat, device=precomp['device']).item()

def pearson_distcorr_metric(precomp):
    # Pearson correlation on actual distances
    d_orig_flat, d_comp_flat = precomp['dist_orig_flat'], precomp['dist_comp_flat']
    if d_orig_flat.numel() < 2 or d_comp_flat.numel() < 2: return float('nan')
    return pearson_correlation(d_orig_flat, d_comp_flat, device=precomp['device']).item()

def explained_variance_ratio_metric(precomp):
    # Ratio of total variance: sum(var(Z_j)) / sum(var(X_j))
    # Assumes X_tensor_orig and X_tensor_comp are (N, D)
    X_orig_t, X_comp_t, n, device = (
        precomp['X_tensor_orig'], precomp['X_tensor_comp'],
        precomp['N'], precomp['device']
    )
    if n < 2: return float('nan') # Variance undefined for < 2 samples
    
    var_orig = X_orig_t.var(dim=0, unbiased=True).sum() # Sum of variances of each original dimension
    var_comp = X_comp_t.var(dim=0, unbiased=True).sum() # Sum of variances of each compressed dimension
    
    if var_orig.item() < 1e-12:
        return 1.0 if var_comp.item() < 1e-12 else float('inf')
    return (var_comp / (var_orig + 1e-12)).item()


def pip_loss_metric(precomp, normalize_embeddings=False):
    # PIP Loss: ||XX^T - ZZ^T||_F^2 (squared Frobenius norm) or ||XX^T - ZZ^T||_F (just Frobenius norm)
    # Your PDF: ||XXT - ZZT||F (no square on norm)
    # Your evaluation.py: torch.norm( (X@X.T) - (Xr@Xr.T), p="fro" ).item() - also no square on norm result
    X_orig_t, X_comp_t, n, device = (
        precomp['X_tensor_orig'], precomp['X_tensor_comp'],
        precomp['N'], precomp['device']
    )
    if n == 0: return 0.0

    X_eval = X_orig_t
    Z_eval = X_comp_t

    if normalize_embeddings: # Typically not done for PIP unless specified
        X_eval = X_orig_t / (torch.norm(X_orig_t, p=2, dim=1, keepdim=True) + 1e-12)
        Z_eval = X_comp_t / (torch.norm(X_comp_t, p=2, dim=1, keepdim=True) + 1e-12)

    gram_orig = X_eval @ X_eval.T
    gram_comp = Z_eval @ Z_eval.T
    
    # The PDF equation for PIP loss seems to be ||XXT - ZZT||_F (Frobenius norm of the difference)
    # No square on the final result.
    return torch.linalg.norm(gram_orig - gram_comp, ord='fro').item()


def eigenspace_overlap_score_metric(precomp, eps=1e-9):

    X_orig_t, X_comp_t, n, device = (
        precomp['X_tensor_orig'], precomp['X_tensor_comp'],
        precomp['N'], precomp['device']
    )
    if n == 0: return 0.0
    d_orig, d_comp = X_orig_t.shape[1], X_comp_t.shape[1]
    if d_orig == 0 or d_comp == 0: return 1.0 if d_orig == 0 and d_comp == 0 else 0.0

    try:
        U_orig, S_orig, _ = torch.linalg.svd(X_orig_t, full_matrices=False)
        U_comp, S_comp, _ = torch.linalg.svd(X_comp_t, full_matrices=False)
    except torch.linalg.LinAlgError:
        logger.warning("SVD failed in Eigenspace Overlap Score.")
        return float('nan')

    rank_orig = (S_orig > eps).sum().item()
    rank_comp = (S_comp > eps).sum().item()

    if rank_orig == 0 or rank_comp == 0:
        return 1.0 if rank_orig == 0 and rank_comp == 0 else 0.0

    U_orig_r = U_orig[:, :rank_orig]
    U_comp_r = U_comp[:, :rank_comp]

    # Frobenius norm squared of the projection matrix U_orig_r^T @ U_comp_r
    overlap_sq_frob = (U_orig_r.T @ U_comp_r).pow(2).sum()
    
    norm_factor = float(max(d_orig, d_comp)) # As per your evaluation.py's EOS
    if norm_factor < eps : return 0.0

    return (overlap_sq_frob / norm_factor).item()


# --- Novel EOS_k (Residual Eigenspace Overlap Score) ---
def residual_eigenspace_overlap_score_metric(precomp, k_components_to_remove, n_subspace_for_overlap, eps=1e-9):
    """
    Implements EOS_k based on Algorithm 1 from the PDF, using Left Singular Vectors (U)
    for subspace comparison to handle differing feature dimensions in residuals.
    "Residual Eigenspace Overlap Score"

    Args:
        precomp: Dictionary with precomputed X_tensor_orig, X_tensor_comp, N, device.
        k_components_to_remove (int): Number of top principal components to remove from X and Z.
        n_subspace_for_overlap (int): Dimension of the subspace from residuals to compute overlap on.
        eps (float): Epsilon for SVD rank determination and numerical stability.

    Returns:
        float: EOS_k score.
    """
    X_orig, Z_comp, n_samples, device = (
        precomp['X_tensor_orig'], precomp['X_tensor_comp'],
        precomp['N'], precomp['device']
    )

    if n_samples == 0:
        logger.debug("EOS_k: n_samples is 0, returning 0.0")
        return 0.0
    if n_samples < 2: # SVD behaves differently with 1 sample, or rank issues.
        logger.warning("EOS_k: n_samples < 2, cannot reliably compute SVD for residuals. Returning NaN.")
        return float('nan')


    def get_residual_and_left_singular_vectors(M, k_remove, M_name="matrix"):
        """
        Helper to compute the residual M' = M - M V_k V_k^T
        and then return the top n_subspace_for_overlap left singular vectors (U_sub) of M'.
        """
        # Ensure M is on the correct device and float32 for SVD
        M_t = M.to(device=device, dtype=torch.float32)

        if M_t.shape[1] == 0: # No features in M
            logger.debug(f"EOS_k ({M_name}): Matrix has 0 features. Residual is 0-dim. Returning None for U_sub, 0 rank.")
            return torch.zeros_like(M_t), None, 0

        M_prime = M_t
        if k_remove > 0:
            if k_remove >= M_t.shape[1]: # Removing all or more components than exist
                logger.debug(f"EOS_k ({M_name}): k_remove={k_remove} >= num_features={M_t.shape[1]}. Residual will be zero matrix.")
                M_prime = torch.zeros_like(M_t)
            else:
                try:
                    # For M' = M - M V_k V_k^T, we need V_k (top k right singular vectors of M)
                    _, S_M, Vh_M = torch.linalg.svd(M_t, full_matrices=False)
                    
                    # Check rank to avoid issues if k_remove > rank(M)
                    rank_M = (S_M > eps).sum().item()
                    actual_k_to_remove = min(k_remove, rank_M)

                    if actual_k_to_remove > 0:
                        V_M_cols = Vh_M.T # Vh_M is (D, D), V_M_cols is (D, D) with singular vectors as columns
                        V_M_top_k = V_M_cols[:, :actual_k_to_remove] # First k columns of V
                        
                        # Projection onto top-k PCs: M @ V_M_top_k @ V_M_top_k.T
                        # Residual: M - (M @ V_M_top_k @ V_M_top_k.T)
                        M_prime = M_t - (M_t @ V_M_top_k @ V_M_top_k.T)
                    else: # k_remove > 0 but rank_M is 0 or k_remove leads to 0 actual components to remove
                        logger.debug(f"EOS_k ({M_name}): k_remove={k_remove} but actual_k_to_remove is 0 (rank_M={rank_M}). Using original M as M_prime.")
                        M_prime = M_t # Effectively no components removed
                        
                except torch.linalg.LinAlgError:
                    logger.warning(f"SVD failed for {M_name} during residual calculation. Using original M as M_prime.")
                    M_prime = M_t # Fallback: use original matrix if SVD for residual fails
        
        # Now, SVD of the residual M_prime
        if M_prime.shape[1] == 0: # M_prime ended up with zero features
             logger.debug(f"EOS_k ({M_name}): M_prime has 0 features. Returning None for U_sub, 0 rank.")
             return M_prime, None, 0 # M_prime, U_sub, rank_M_prime

        try:
            U_M_prime, S_M_prime, _ = torch.linalg.svd(M_prime, full_matrices=False)
            rank_M_prime = (S_M_prime > eps).sum().item()
            return M_prime, U_M_prime, rank_M_prime
        except torch.linalg.LinAlgError:
            logger.warning(f"SVD failed for residual M_prime ({M_name}). Returning None for U_sub.")
            return M_prime, None, 0 # M_prime, U_sub, rank_M_prime


    # Get residuals and their Left Singular Vectors (U) and ranks
    X_prime, U_X_prime, rank_X_prime = get_residual_and_left_singular_vectors(X_orig, k_components_to_remove, "X_orig")
    Z_prime, U_Z_prime, rank_Z_prime = get_residual_and_left_singular_vectors(Z_comp, k_components_to_remove, "Z_comp")

    if U_X_prime is None or U_Z_prime is None: # Indicates SVD failure on a residual or zero-dim residual
        logger.warning("EOS_k: Could not obtain U vectors for one or both residuals. Returning NaN.")
        return float('nan')
    
    # Determine the number of subspace dimensions to actually use for overlap
    # effective_n_sub should not exceed the rank of the residuals or the number of columns in U
    max_cols_U_X = U_X_prime.shape[1]
    max_cols_U_Z = U_Z_prime.shape[1]

    actual_n_sub_X = min(n_subspace_for_overlap, rank_X_prime, max_cols_U_X)
    actual_n_sub_Z = min(n_subspace_for_overlap, rank_Z_prime, max_cols_U_Z)
    
    effective_n_sub = min(actual_n_sub_X, actual_n_sub_Z)

    if effective_n_sub <= 0:
        logger.debug(f"EOS_k: effective_n_sub is {effective_n_sub} (ranks X':{rank_X_prime}, Z':{rank_Z_prime}).")
        # If both residual spaces are effectively zero-rank (e.g., k_components_to_remove >= original ranks)
        # this could be considered perfect alignment of "nothingness".
        if rank_X_prime == 0 and rank_Z_prime == 0:
            logger.debug("EOS_k: Both residual ranks are 0. Returning 1.0.")
            return 1.0
        logger.debug("EOS_k: One residual rank is 0 or effective_n_sub is 0. Returning 0.0.")
        return 0.0 # One subspace is empty/trivial, the other is not, or no common overlap dim

    # Get top 'effective_n_sub' LEFT singular vectors (columns of U)
    U_X_prime_sub = U_X_prime[:, :effective_n_sub] # Shape: (n_samples, effective_n_sub)
    U_Z_prime_sub = U_Z_prime[:, :effective_n_sub] # Shape: (n_samples, effective_n_sub)

    # Step 11 (Algorithm 1): M_overlap = (U_X_prime_sub)^T @ U_Z_prime_sub
    # This matrix M_overlap contains dot products of basis vectors from the two subspaces.
    # Its singular values (related to principal angles) measure the similarity.
    M_overlap = U_X_prime_sub.T @ U_Z_prime_sub # Shape: (effective_n_sub, effective_n_sub)

    # Step 12 (Algorithm 1): Singular values of M_overlap
    try:
        singular_values_M_overlap = torch.linalg.svdvals(M_overlap)
    except torch.linalg.LinAlgError:
        logger.warning("SVD failed for M_overlap matrix in EOS_k. Returning NaN.")
        return float('nan')
    except Exception as e: # Catch any other SVD issue
        logger.warning(f"An unexpected error occurred during SVD of M_overlap in EOS_k: {e}. Returning NaN.")
        return float('nan')


    # Step 13 (Algorithm 1): EOS_k = (1/N_sub) * sum(sigma_i^2)
    # sigma_i are the singular values of M_overlap.
    # These are cosines of principal angles if U_X_prime_sub and U_Z_prime_sub are orthonormal (which they are).
    score = (1.0 / effective_n_sub) * torch.sum(singular_values_M_overlap.pow(2))
    
    logger.debug(f"EOS_k calculated: {score.item()} (effective_n_sub: {effective_n_sub})")
    return score.item()


# --- Main Evaluation Function ---
def compute_all_metrics(precomputed_data, k_val_for_local_metrics=10, k_for_eos_k=5, n_sub_for_eos_k=10):
    """Computes all defined metrics using the precomputed data."""
    metrics = {}
    logger.info(f"Computing all metrics with k_local={k_val_for_local_metrics}, k_eos={k_for_eos_k}, n_sub_eos={n_sub_for_eos_k}")

    # Local Neighborhood Fidelity
    metrics['trustworthiness'] = trustworthiness_metric(precomputed_data, k=k_val_for_local_metrics)
    metrics['continuity'] = continuity_metric(precomputed_data, k=k_val_for_local_metrics)
    metrics['mean_relative_rank_error'] = mean_relative_rank_error_metric(precomputed_data, k=k_val_for_local_metrics)
    metrics['neighbor_precision'] = neighbor_precision_metric(precomputed_data, k=k_val_for_local_metrics)
    metrics['local_procrustes'] = local_procrustes_metric(precomputed_data, k=k_val_for_local_metrics)

    # Global Geometry Fidelity
    metrics['kruskal_stress'] = kruskal_stress_metric(precomputed_data)
    metrics['spearman_distcorr'] = spearman_distcorr_metric(precomputed_data)
    metrics['pearson_distcorr'] = pearson_distcorr_metric(precomputed_data)
    metrics['global_procrustes'] = global_procrustes_metric(precomputed_data)

    # Spectral Retention / Information Retention
    metrics['explained_variance_ratio'] = explained_variance_ratio_metric(precomputed_data)
    metrics['pip_loss'] = pip_loss_metric(precomputed_data)
    metrics['eigenspace_overlap_score'] = eigenspace_overlap_score_metric(precomputed_data)
    metrics['eos_k_residual_overlap'] = residual_eigenspace_overlap_score_metric(
        precomputed_data,
        k_components_to_remove=k_for_eos_k,
        n_subspace_for_overlap=n_sub_for_eos_k
    )
    
    # Log computed metrics
    for name, val in metrics.items():
        logger.info(f"Metric - {name}: {val:.4f}")
        
    return metrics