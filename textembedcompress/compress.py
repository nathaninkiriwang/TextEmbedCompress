import numpy as np
from sklearn.decomposition import IncrementalPCA, FactorAnalysis, FastICA
from sklearn.random_projection import GaussianRandomProjection
import warnings
from .common_utils import logger

try:
    from umap import UMAP as UMAP_learn 
except ImportError:
    UMAP_learn = None
    logger.warning("umap-learn not installed. CPU UMAP will not be available.")

try:
    import pacmap
except ImportError:
    pacmap = None
    logger.warning("pacmap not installed. PaCMAP will not be available.")

# Placeholder for cuml UMAP if you decide to support it optionally
# try:
#     from cuml.manifold import UMAP as UMAP_cuml
# except ImportError:
#     UMAP_cuml = None

class _IdentityReducer:
    def fit_transform(self, X): return X
    def transform(self, X): return X


def reduce_dimensionality(embeddings_np, method, target_dim, batch_size_for_ipca=1024, random_state=42):
    """
    Applies dimensionality reduction to the embeddings.
    Returns reduced embeddings and the fitted reducer model.
    """
    n_samples, original_dim = embeddings_np.shape
    logger.info(f"Reducing dimensionality from {original_dim} to {target_dim} using {method.upper()}. Samples: {n_samples}")

    if target_dim >= original_dim:
        logger.warning(f"Target dimension ({target_dim}) is >= original dimension ({original_dim}). Returning original embeddings.")
        return embeddings_np, _IdentityReducer()
    if target_dim <=0:
        logger.error(f"Target dimension ({target_dim}) must be positive.")
        raise ValueError("Target dimension must be positive.")
    if n_samples < target_dim and method.lower() in ['pca', 'fa', 'ica']: # Some methods require n_samples >= n_components
        logger.warning(f"Number of samples ({n_samples}) is less than target dimension ({target_dim}) for {method}. Adjusting target_dim to {n_samples-1 if n_samples >1 else 1}.")
        target_dim = max(1, n_samples -1) if n_samples > 1 else 1
        if target_dim == 0: # Edge case: single sample cannot be reduced by these
             logger.warning(f"Cannot apply {method} with target_dim=0 (single sample). Returning original.")
             return embeddings_np, _IdentityReducer()


    method = method.lower()
    reducer = None

    if method == "none":
        return embeddings_np, _IdentityReducer()
    elif method == "pca":
        # IncrementalPCA is good for large datasets that might not fit in memory for standard PCA
        reducer = IncrementalPCA(n_components=target_dim, batch_size=min(batch_size_for_ipca, n_samples))
        # IPCA needs to be fit in batches
        transformed_embeddings = np.zeros((n_samples, target_dim), dtype=embeddings_np.dtype)
        for i in range(0, n_samples, batch_size_for_ipca):
            reducer.partial_fit(embeddings_np[i:i + batch_size_for_ipca])
        # After fitting all parts, then transform
        for i in range(0, n_samples, batch_size_for_ipca):
            transformed_embeddings[i:i + batch_size_for_ipca] = reducer.transform(embeddings_np[i:i+batch_size_for_ipca])
        return transformed_embeddings, reducer
    elif method == "rp":
        reducer = GaussianRandomProjection(n_components=target_dim, random_state=random_state)
    elif method == "fa":
        reducer = FactorAnalysis(n_components=target_dim, random_state=random_state, max_iter=1000) # Default iter might be low
    elif method == "ica":
        reducer = FastICA(n_components=target_dim, random_state=random_state, max_iter=1000, tol=1e-3) # Increased max_iter
    elif method == "umap":
        if UMAP_learn:
            reducer = UMAP_learn(n_components=target_dim, n_neighbors=15, min_dist=0.1, random_state=random_state, transform_seed=random_state)
        else:
            raise ImportError("UMAP (umap-learn) is not installed. Please install it to use this method.")
    elif method == "pacmap":
        if pacmap:
            reducer = pacmap.PaCMAP(n_components=target_dim, n_neighbors=15, MN_ratio=0.5, FP_ratio=2.0, random_state=random_state)
        else:
            raise ImportError("PaCMAP is not installed. Please install it to use this method.")
    else:
        warnings.warn(f"Unknown DR method '{method}'. Returning original embeddings.")
        return embeddings_np, _IdentityReducer()

    if reducer:
        logger.info(f"Fitting {method.upper()} reducer...")
        # For non-IPCA methods, fit_transform directly
        if method != "pca": # PCA (IPCA) is handled differently above
            transformed_embeddings = reducer.fit_transform(embeddings_np)
        return transformed_embeddings.astype(np.float32), reducer
    
    return embeddings_np, _IdentityReducer()


def quantize_embeddings(embeddings_np, mode='int8'):
    """Applies quantization to the embeddings."""
    if not isinstance(embeddings_np, np.ndarray):
        embeddings_np = np.array(embeddings_np)

    logger.info(f"Quantizing embeddings to {mode}.")
    if mode == "int8":
        # Symmetric quantization per-tensor
        abs_max = np.max(np.abs(embeddings_np))
        if abs_max == 0: # Avoid division by zero if all embeddings are zero
            scale = 1.0
        else:
            scale = 127.0 / abs_max
        quantized = np.round(embeddings_np * scale).astype(np.int8)
        # Store scale for dequantization (though not strictly needed by your PDF's downstream tasks)
        # For this package, we primarily care about the compressed representation for storage/eval
        return quantized, {"quantization_mode": "int8", "scale": scale if abs_max != 0 else 0.0}
    elif mode == "none":
        return embeddings_np, {"quantization_mode": "none"}
    else:
        warnings.warn(f"Unknown quantization mode '{mode}'. Returning original embeddings.")
        return embeddings_np, {"quantization_mode": "none"}