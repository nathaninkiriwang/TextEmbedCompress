# TextEmbedCompress

**TextEmbedCompress** is a Python toolkit designed for compressing text embeddings and rigorously evaluating their quality using a comprehensive suite of intrinsic, task-agnostic metrics. It features several dimensionality reduction techniques, int8 quantization, and a novel spectral fidelity measure, $EOS_k$, to assess the preservation of semantic structure beyond dominant variance components.

This framework allows researchers and practitioners to make informed decisions about embedding compression strategies, balancing computational efficiency with the preservation of meaningful information.

## Key Features

* **Embedding Model Support:**
    * Easily load and use models from the [Sentence Transformers](https://www.sbert.net/) library.
    * Support for static word embedding models like GloVe, Word2Vec, and FastText (sentence embeddings derived via averaging).
* **Compression Techniques:**
    * **Dimensionality Reduction (DR):**
        * Principal Component Analysis (PCA)
        * Independent Component Analysis (ICA)
        * Gaussian Random Projections (RP)
        * Factor Analysis (FA)
        * Uniform Manifold Approximation and Projection (UMAP)
        * Pairwise Controlled Manifold Approximation (PaCMAP)
    * **Quantization:**
        * Symmetric int8 per-tensor quantization.
* **Comprehensive Intrinsic Evaluation Framework:**
    * **Local Neighborhood Fidelity:**
        * [cite_start]Trustworthiness ($T_k$) 
        * [cite_start]Continuity ($C_k$) 
        * [cite_start]Mean Relative Rank Error ($MRRE_k$) 
        * [cite_start]Neighborhood Precision at k ($NP_k$) 
        * [cite_start]Local Average Procrustes Measure (LPro) 
    * **Global Geometry Fidelity:**
        * [cite_start]Kruskal's Stress (KS) 
        * [cite_start]Spearman Distance Correlation (SDC) 
        * [cite_start]Pearson Distance Correlation (PDC) 
        * [cite_start]Global Procrustes Measure (GPro) 
    * **Spectral Retention / Information Fidelity:**
        * [cite_start]Explained Variance Ratio (EVR) 
        * [cite_start]Pairwise Inner-Product (PIP) Loss 
        * [cite_start]Eigenspace Overlap Score (EOS) 
        * [cite_start]**Novel $EOS_k$ (Residual Eigenspace Overlap Score):** Our proposed metric to evaluate semantic preservation after removing top-k principal components, offering a more nuanced view of information retention.
* **Easy-to-Use Pipeline:** A streamlined `EmbeddingPipeline` class to manage the workflow from embedding generation to compression and evaluation.
* **Reproducibility:** Control over random states for stochastic DR methods.

## Installation

You can install `TextEmbedCompress` directly from PyPI:

```bash
pip install TextEmbedCompress
```

## Quick-Start

Here's a simple example of how to use **TextEmbedCompress**:

```python
from textembedcompress import EmbeddingPipeline 
import json

# 1. Initialize the pipeline with a sentence-transformer model
# This will automatically use CUDA if available, otherwise CPU.
pipeline = EmbeddingPipeline(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")

# 2. Define your dataset (can be Hugging Face dataset ID, path to local file, or list of texts)
# For this example, we'll use a dummy list of texts.
sample_texts = [
    "This is the first sentence for testing.",
    "Here is another sentence, quite different from the first.",
    "Embeddings are numerical representations of text.",
    "Compressing embeddings can save space and speed up computations.",
    "Evaluation helps to understand the quality of compressed embeddings."
] * 20 # Multiply to get enough samples for DR/evaluation

# 3. Run the full compression and evaluation pipeline
results = pipeline.run(
    dataset_identifier=sample_texts,
    dr_method="pca",
    target_dim=32, # Target dimension after DR
    quantization_method="int8", # Apply int8 quantization
    output_dir="output/all-MiniLM-L6-v2_pca32_int8", # Directory to save results
    k_val_for_local_metrics=5,  # k for T_k, C_k, NP_k, MRRE_k, LPro
    k_for_eos_k=2,              # Number of top components to remove for EOS_k
    n_sub_for_eos_k=10,         # Subspace dimension for EOS_k overlap calculation
    distance_metric_for_eval='cosine', # Distance metric for nn-based evaluations
    random_state_for_dr=42
)

# 4. Access results
print("\n--- Compression Info ---")
# Ensure results["compression_info"] is serializable for json.dumps
serializable_compression_info = {k: str(v) for k, v in results["compression_info"].items()}
print(json.dumps(serializable_compression_info, indent=2))

print("\n--- Evaluation Metrics ---")
serializable_evaluation_metrics = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in results["evaluation_metrics"].items()}
print(json.dumps(serializable_evaluation_metrics, indent=2))

print(f"\nCompressed embeddings and metrics saved to: {results['output_location']}")

```



## Usage Details

The core of the package is the `EmbeddingPipeline` class.

### `EmbeddingPipeline(model_name_or_path, device=None, trust_remote_code=True)`
* `model_name_or_path`: Name of a Sentence Transformers model (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`) or path to a static embedding file (e.g., `"path/to/glove.840B.300d.txt"`).
* `device`: Optional; "cpu" or "cuda". Defaults to "cuda" if available, else "cpu".
* `trust_remote_code`: For loading Hugging Face models that require custom code.

### Main Methods

* **`.embed(dataset_identifier, text_column_names=None, dataset_split=None, batch_size=32, show_progress_bar=True)`**:
    * Loads the specified dataset and generates original embeddings.
    * `dataset_identifier`: Hugging Face dataset name (e.g., `"imdb"`), a tuple for HF datasets with configs (e.g., `("glue", "mrpc")`), path to a local file, or a list of Python strings.
    * `text_column_names`: List of column names containing text in an HF dataset.
    * `dataset_split`: E.g., "train", "test", "train[:10%]".
* **`.compress(dr_method='pca', target_dim=128, quantization_method='int8', ...)`**:
    * Applies dimensionality reduction and/or quantization.
    * `dr_method`: One of `'pca'`, `'ica'`, `'rp'`, `'fa'`, `'umap'`, `'pacmap'`, or `'none'`.
    * `target_dim`: Desired output dimension after DR.
    * `quantization_method`: `'int8'` or `'none'`.
* **`.evaluate(distance_metric_for_eval='cosine', k_val_for_local_metrics=10, k_for_eos_k=5, n_sub_for_eos_k=10, ...)`**:
    * Computes all evaluation metrics comparing original and compressed embeddings.
    * `distance_metric_for_eval`: Distance used for neighbor-based metrics ('cosine' or 'euclidean').
    * `k_val_for_local_metrics`: Neighborhood size for local metrics like $T_k, C_k, NP_k, LPro$.
    * `k_for_eos_k`: Number of dominant components to remove before $EOS_k$ calculation.
    * `n_sub_for_eos_k`: Subspace dimension for overlap calculation in $EOS_k$.
* **`.run(...)`**: A convenience method that calls `embed`, `compress`, `evaluate`, and `save_results` sequentially. Takes parameters for all these steps.
* **`.save_results(output_dir)`**:
    * Saves compressed embeddings (as `.npy`), compression info (as `.json`), and evaluation metrics (as `.json`) to the specified directory.

### The $EOS_k$ Metric

The $EOS_k$ (Residual Eigenspace Overlap Score) is a novel metric designed to assess how well semantic structure is preserved in compressed embeddings *after* accounting for and removing the influence of the top-$k$ most dominant principal components (directions of highest variance).

The rationale is that these top components often capture broad, sometimes task-agnostic or even noisy, variance that can overshadow more subtle, task-relevant semantic features. $EOS_k$ works by:
1.  Calculating residual embeddings for both original ($\mathbf{X}'$) and compressed ($\mathbf{Z}'$) spaces by subtracting the projections onto their respective top-$k$ right singular vectors.
2.  Performing SVD on these residuals $\mathbf{X}'$ and $\mathbf{Z}'$.
3.  Comparing the alignment of the top $N_{sub}$ principal directions (derived from left singular vectors) of these "cleaned" residual subspaces.
4.  A high $EOS_k$ score indicates that complex, task-relevant structure beyond the dominant components is well-preserved.


## Contributing

Contributions are welcome! Whether it's bug reports, feature suggestions, or pull requests, please feel free to engage.

1.  **Reporting Issues:** Please use the [GitHub issue tracker](https://github.com/yourusername/TextEmbedCompress/issues).
2.  **Feature Requests:** Submit an issue detailing the feature and its use case.
3.  **Pull Requests:**
    * Fork the repository.
    * Create a new branch for your feature or bug fix.
    * Ensure your code adheres to quality standards (e.g., run linters, add tests).
    * Submit a pull request with a clear description of your changes.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
