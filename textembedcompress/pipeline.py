import os
import json
import numpy as np
import torch
import gc

from .common_utils import logger
from .models import load_embedding_model, generate_embeddings
from .datasets import load_and_extract_texts
from .compress import reduce_dimensionality, quantize_embeddings
from .evaluate import precompute_evaluation_data, compute_all_metrics

class EmbeddingPipeline:
    def __init__(self, model_name_or_path, device=None, trust_remote_code=True):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        logger.info(f"Initializing pipeline with device: {self.device}")
        self.model = load_embedding_model(model_name_or_path, self.device, trust_remote_code)
        self.original_embeddings = None
        self.compressed_embeddings = None
        self.compression_info = {}
        self.evaluation_metrics = {}

    def embed(self, dataset_identifier, text_column_names=None, dataset_split=None,
              batch_size=32, show_progress_bar=True):
        """Loads dataset and generates original embeddings."""
        texts = load_and_extract_texts(dataset_identifier, text_column_names, dataset_split)
        if not texts:
            raise ValueError("No texts were extracted from the dataset.")
        
        self.original_embeddings = generate_embeddings(
            self.model, texts, batch_size, self.device, show_progress_bar
        )
        logger.info(f"Generated original embeddings of shape: {self.original_embeddings.shape}")
        # Free up GPU memory if model was on GPU for encoding
        if self.device == 'cuda' and hasattr(self.model, 'to'):
             self.model.to('cpu') # Move model to CPU after embedding if it's large
             torch.cuda.empty_cache()
        return self

    def compress(self, dr_method='pca', target_dim=128, quantization_method='int8',
                 batch_size_for_ipca=1024, random_state_for_dr=42):
        """Applies DR and quantization to the original embeddings."""
        if self.original_embeddings is None:
            raise ValueError("Original embeddings not generated. Call embed() first.")

        current_embeddings = self.original_embeddings
        
        # Dimensionality Reduction
        if dr_method and dr_method.lower() != 'none':
            reduced_embeddings, reducer_info = reduce_dimensionality(
                current_embeddings, dr_method, target_dim, batch_size_for_ipca, random_state_for_dr
            )
            self.compression_info['dr_method'] = dr_method
            self.compression_info['target_dim'] = target_dim # Actual target dim used
            self.compression_info['dr_reducer_info'] = str(reducer_info) # Basic info
            current_embeddings = reduced_embeddings
            logger.info(f"Applied DR. New shape: {current_embeddings.shape}")
        else:
            self.compression_info['dr_method'] = 'none'

        # Quantization
        if quantization_method and quantization_method.lower() != 'none':
            quantized_embeddings, quant_info = quantize_embeddings(current_embeddings, quantization_method)
            self.compression_info.update(quant_info)
            self.compressed_embeddings = quantized_embeddings # This might be int8
            logger.info(f"Applied quantization. Final type: {self.compressed_embeddings.dtype}")
        else:
            self.compressed_embeddings = current_embeddings # No quantization, still float32
            self.compression_info['quantization_mode'] = 'none'
        
        # If compressed embeddings are int8, evaluation needs float32 for distance calcs
        # The current evaluation framework expects float inputs.
        # For storage, int8 is fine. For evaluation, we might need to dequantize or adapt metrics.
        # For now, assume evaluation metrics will handle numpy arrays and convert to torch tensors.
        # If quantization was applied and resulted in int8, we'll use a dequantized version for eval
        # or ensure eval functions can handle int8 by scaling.
        # The simplest is to evaluate the float32 version *before* true int8 storage,
        # or dequantize for evaluation if only int8 is stored.
        # Your paper's Figure 1 caption mentions "int8-quantized embeddings with DR".
        # This implies evaluation happens on the (implicitly dequantized for math ops) int8 values.
        # Let's assume `compute_all_metrics` will get the potentially int8 `self.compressed_embeddings`
        # and handle it (e.g. by converting to float for calculations).
        # The `precompute_evaluation_data` already converts to float32 tensors.

        return self

    def evaluate(self, distance_metric_for_eval='cosine', k_val_for_local_metrics=10,
                 k_for_eos_k=5, n_sub_for_eos_k=10, batch_size_for_eval_precompute=512):
        """Evaluates compressed embeddings against original ones."""
        if self.original_embeddings is None or self.compressed_embeddings is None:
            raise ValueError("Original or compressed embeddings not available. Call embed() and compress() first.")

        # Compressed embeddings might be int8. Evaluation functions expect float.
        # precompute_evaluation_data will convert to float32 tensors.
        eval_device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU for eval if possible
        
        logger.info(f"Starting evaluation on device: {eval_device}")
        precomputed_data = precompute_evaluation_data(
            self.original_embeddings,
            self.compressed_embeddings, # This might be int8, precompute will handle conversion
            distance_metric=distance_metric_for_eval,
            device=eval_device,
            batch_size=batch_size_for_eval_precompute,
            k_max_for_nn=max(k_val_for_local_metrics, 15) # Ensure enough neighbors for all metrics
        )
        
        self.evaluation_metrics = compute_all_metrics(
            precomputed_data,
            k_val_for_local_metrics=k_val_for_local_metrics,
            k_for_eos_k=k_for_eos_k,
            n_sub_for_eos_k=n_sub_for_eos_k
        )
        
        # Clean up GPU memory after evaluation
        del precomputed_data
        gc.collect()
        if eval_device == 'cuda':
            torch.cuda.empty_cache()
            
        return self

    def save_results(self, output_dir):
        """Saves compressed embeddings and metrics."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.compressed_embeddings is not None:
            emb_path = os.path.join(output_dir, "compressed_embeddings.npy")
            np.save(emb_path, self.compressed_embeddings)
            logger.info(f"Saved compressed embeddings to {emb_path}")

        info_path = os.path.join(output_dir, "compression_info.json")
        with open(info_path, 'w') as f:
            # Convert numpy types in info to standard types for JSON
            serializable_info = {}
            for k, v in self.compression_info.items():
                if isinstance(v, np.generic): serializable_info[k] = v.item()
                else: serializable_info[k] = v
            json.dump(serializable_info, f, indent=4)
        logger.info(f"Saved compression info to {info_path}")

        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            serializable_metrics = {}
            for k, v in self.evaluation_metrics.items():
                if isinstance(v, np.generic): serializable_metrics[k] = v.item()
                elif isinstance(v, torch.Tensor): serializable_metrics[k] = v.item()
                else: serializable_metrics[k] = v
            json.dump(serializable_metrics, f, indent=4)
        logger.info(f"Saved evaluation metrics to {metrics_path}")

    def run(self, dataset_identifier, dr_method, target_dim, quantization_method, output_dir,
            text_column_names=None, dataset_split=None, embed_batch_size=32,
            k_val_for_local_metrics=10, k_for_eos_k=5, n_sub_for_eos_k=10,
            distance_metric_for_eval='cosine', random_state_for_dr=42):
        """Runs the full pipeline: embed, compress, evaluate, save."""
        self.embed(dataset_identifier, text_column_names, dataset_split, embed_batch_size)
        self.compress(dr_method, target_dim, quantization_method, random_state_for_dr=random_state_for_dr)
        self.evaluate(distance_metric_for_eval, k_val_for_local_metrics, k_for_eos_k, n_sub_for_eos_k)
        if output_dir:
            self.save_results(output_dir)
        
        logger.info("Pipeline finished.")
        return {
            "compression_info": self.compression_info,
            "evaluation_metrics": self.evaluation_metrics,
            "output_location": output_dir
        }