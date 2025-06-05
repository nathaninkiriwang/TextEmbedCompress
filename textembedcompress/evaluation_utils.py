import torch
import math
import numpy as np
from .common_utils import logger # Assuming to_tensor might be more general

def to_tensor_eval(x, device='cpu', dtype=torch.float32): # Renamed to avoid clash if common_utils has one
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=dtype)
    return x.to(device=device, dtype=dtype)

def batch_pairwise_euclidean_distance(x, y=None, device='cpu', batch_size=1024):
    x = to_tensor_eval(x, device=device)
    if y is None: y = x
    else: y = to_tensor_eval(y, device=device)
    # ... (rest of your batch_pairwise_euclidean_distance)
    # Simplified for brevity, use your full robust version
    if x.shape[0] == 0 or (y is not None and y.shape[0] == 0):
        return torch.empty((x.shape[0], y.shape[0] if y is not None else x.shape[0]), device=device)
    return torch.cdist(x, y, p=2.0)


def batch_pairwise_cosine_similarity(x, y=None, device='cpu', batch_size=1024, eps=1e-8):
    x = to_tensor_eval(x, device=device)
    if y is None: y = x
    else: y = to_tensor_eval(y, device=device)
    # ... (rest of your batch_pairwise_cosine_similarity)
    # Simplified for brevity, use your full robust version
    if x.shape[0] == 0 or (y is not None and y.shape[0] == 0):
        return torch.empty((x.shape[0], y.shape[0] if y is not None else x.shape[0]), device=device)
    
    x_norm = x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps)
    y_norm = y / (torch.norm(y, p=2, dim=1, keepdim=True) + eps)
    return torch.mm(x_norm, y_norm.transpose(0, 1))


def spearman_rank_correlation(a, b, device='cpu', eps=1e-9):
    a_t = to_tensor_eval(a, device=device).view(-1)
    b_t = to_tensor_eval(b, device=device).view(-1)
    # ... (rest of your spearman_rank_correlation)
    if a_t.numel() < 2: return torch.tensor(float('nan'), device=device)
    a_ranks = torch.argsort(torch.argsort(a_t, stable=True), stable=True).float()
    b_ranks = torch.argsort(torch.argsort(b_t, stable=True), stable=True).float()
    a_ranks_mean, b_ranks_mean = a_ranks.mean(), b_ranks.mean()
    a_centered, b_centered = a_ranks - a_ranks_mean, b_ranks - b_ranks_mean
    num = (a_centered * b_centered).sum()
    den_a_sq, den_b_sq = (a_centered**2).sum(), (b_centered**2).sum()
    den = torch.sqrt(den_a_sq * den_b_sq)
    return num / (den + eps) if den > eps else torch.tensor(float('nan'), device=device)


def pearson_correlation(a, b, device='cpu', eps=1e-9):
    a_t = to_tensor_eval(a, device=device, dtype=torch.float32).view(-1)
    b_t = to_tensor_eval(b, device=device, dtype=torch.float32).view(-1)
    # ... (rest of your pearson_correlation)
    if a_t.numel() < 2: return torch.tensor(float('nan'), device=device)
    a_mean, b_mean = a_t.mean(), b_t.mean()
    a_centered, b_centered = a_t - a_mean, b_t - b_mean
    num = (a_centered * b_centered).sum()
    den_a_sq, den_b_sq = (a_centered**2).sum(), (b_centered**2).sum()
    den = torch.sqrt(den_a_sq * den_b_sq)
    return num / (den + eps) if den > eps else torch.tensor(float('nan'), device=device)


def _procrustes_alignment_core(X_target, Y_source, device='cpu', compute_scaling=True, eps=1e-9):
    # This is the robust version from your evaluation.py
    X = to_tensor_eval(X_target, device=device, dtype=torch.float32)
    Y = to_tensor_eval(Y_source, device=device, dtype=torch.float32)
    # ... (exact code from your evaluation.py for _procrustes_alignment_core)
    # For brevity, assuming it's copied here. Ensure all internal logger calls use `from .common_utils import logger`
    if X.ndim != 2 or Y.ndim != 2: raise ValueError("Procrustes: Input X_target and Y_source must be 2D tensors.")
    if X.shape[0] != Y.shape[0]: raise ValueError("Procrustes: X_target and Y_source must have the same number of samples.")
    n_samples, d_target, d_source = X.shape[0], X.shape[1], Y.shape[1]
    if n_samples == 0:
        R_id = torch.eye(d_source, d_target, device=device) if d_source <= d_target else torch.eye(d_target, d_source, device=device).T
        return R_id, 1.0, 0.0
    mu_X, mu_Y = X.mean(dim=0), Y.mean(dim=0)
    X_c, Y_c = X - mu_X, Y - mu_Y
    M = Y_c.t() @ X_c
    try: U_svd, S_svd_diag, Vh_svd = torch.linalg.svd(M, full_matrices=False)
    except torch.linalg.LinAlgError as e_svd:
        logger.warning(f"Procrustes SVD failed: {e_svd}. Returning NaN disparity.")
        R_id_fallback = torch.eye(d_source, d_target, device=device) if d_source <= d_target else torch.eye(d_target, d_source, device=device).T
        return R_id_fallback, 1.0, float('nan')
    R = U_svd @ Vh_svd
    s = 1.0
    if compute_scaling:
        sum_Y_c_sq_fro = torch.sum(Y_c**2)
        if sum_Y_c_sq_fro > eps: s = S_svd_diag.sum() / sum_Y_c_sq_fro
        elif S_svd_diag.sum().abs() < eps : s = 1.0 
        else: s = float('inf'); logger.warning("Procrustes: Y_c is zero but SVD singular values are not, scale is inf.")
    Y_c_transformed = s * (Y_c @ R)
    sum_X_c_sq_fro = torch.sum(X_c**2)
    disparity_val = float('nan')
    if sum_X_c_sq_fro < eps: disparity_val = 0.0 if torch.sum((X_c - Y_c_transformed)**2) < eps else 1.0
    else: disparity_val = torch.sum((X_c - Y_c_transformed)**2) / sum_X_c_sq_fro
    return R, s, disparity_val.item()


