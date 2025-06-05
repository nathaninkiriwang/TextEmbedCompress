import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from .common_utils import logger, to_tensor
import numpy as np
import re
import gzip

class StaticEmbeddingModel:
    """Loads static embeddings like GloVe, Word2Vec, FastText."""
    def __init__(self, path: str, dim: int = 300):
        self.dim = dim
        self.vecs = {}
        self.unk = np.zeros(dim, dtype=np.float32)
        self.tok = re.compile(r"[A-Za-z']+")
        
        logger.info(f"Loading static embeddings from {path}...")
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                parts = line.rstrip().split()
                if len(parts) == 2 and i == 0: # Skip header line e.g. fastText
                    logger.info(f"Skipping header: {line.strip()}")
                    continue
                if len(parts) != dim + 1:
                    if i < 5: # Log only a few malformed lines
                        logger.warning(f"Malformed line (expected {dim+1} parts, got {len(parts)}): '{line[:50]}...'")
                    continue
                word, *vals = parts
                try:
                    self.vecs[word] = np.asarray(vals, dtype=np.float32)
                except ValueError:
                    if i < 5:
                         logger.warning(f"Could not parse values for word '{word}'")
        if not self.vecs:
            logger.error(f"No vectors loaded from {path}. Check path and format.")
            raise ValueError(f"No vectors loaded from {path}")
        logger.info(f"Loaded {len(self.vecs)} static vectors.")

    def encode(self, sentences, batch_size=32, show_progress_bar=False, **kwargs):
        out = []
        iterable = tqdm(sentences, desc="Encoding static") if show_progress_bar else sentences
        for s in iterable:
            words = self.tok.findall(s.lower())
            vecs = [self.vecs.get(w, self.unk) for w in words] or [self.unk]
            out.append(np.mean(vecs, axis=0))
        return np.vstack(out).astype(np.float32)

def load_embedding_model(model_name_or_path, device='cpu', trust_remote_code=True):
    """Loads a sentence-transformer model or a static model."""
    if model_name_or_path.endswith(('.txt', '.vec', '.txt.gz', '.vec.gz')):
        try:
            # Attempt to guess dimension for common static files if not provided
            dim_match = re.search(r'(\d+)d', model_name_or_path.lower())
            dim = int(dim_match.group(1)) if dim_match else 300 # Default to 300
            logger.info(f"Detected static embedding file. Using dimension: {dim}")
            return StaticEmbeddingModel(model_name_or_path, dim=dim)
        except Exception as e:
            logger.error(f"Failed to load static model {model_name_or_path}: {e}")
            raise
    else:
        logger.info(f"Loading sentence-transformer model: {model_name_or_path} on device: {device}")
        try:
            model = SentenceTransformer(
                model_name_or_path,
                device=device,
                trust_remote_code=trust_remote_code
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load sentence-transformer model {model_name_or_path}: {e}")
            raise

def generate_embeddings(model, texts, batch_size=32, device='cpu', show_progress_bar=True):
    """Generates embeddings for a list of texts using the provided model."""
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    
    # For SentenceTransformer models
    if isinstance(model, SentenceTransformer):
        # Ensure model is on the correct device
        original_device = model.device
        model.to(device)
        
        all_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=False, # Get numpy arrays first
            normalize_embeddings=False # Do not normalize here, preserve original
        )
        model.to(original_device) # Move model back if it was moved
        return all_embeddings.astype(np.float32) # Ensure float32
    # For StaticEmbeddingModel or other custom models
    elif hasattr(model, 'encode'):
        return model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
    else:
        raise TypeError("Model type not supported for embedding generation.")