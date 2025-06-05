import sys
import os
import argparse
import json
current_script_directory = os.path.dirname(os.path.abspath(__file__))
project_root_directory = os.path.dirname(current_script_directory)
if project_root_directory not in sys.path:
    sys.path.insert(0, project_root_directory)

from textembedcompress import EmbeddingPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress and evaluate text embeddings.")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence transformer model name or path.")
    # Example for HF dataset: parser.add_argument("--dataset_id", type=str, default="imdb", help="Hugging Face dataset identifier.")
    # Example for list of texts:
    parser.add_argument("--dataset_id", type=str, default="hf_dataset_dummy", help="Dataset identifier (HF name, path, or 'list_dummy' for example).")
    parser.add_argument("--text_columns", type=str, default=None, help="Comma-separated text column names for HF dataset (e.g., 'text').")
    parser.add_argument("--dataset_split", type=str, default="test[:1%]", help="Dataset split (e.g., 'train', 'test[:100]').") # Small for example
    
    parser.add_argument("--dr_method", type=str, default="pca", choices=['pca', 'ica', 'rp', 'fa', 'umap', 'pacmap', 'none'], help="Dimensionality reduction method.")
    parser.add_argument("--target_dim", type=int, default=64, help="Target dimension for DR.")
    parser.add_argument("--quant_method", type=str, default="int8", choices=['int8', 'none'], help="Quantization method.")
    
    parser.add_argument("--output_dir", type=str, default="output_results", help="Directory to save results.")
    
    parser.add_argument("--k_local", type=int, default=10, help="k for local neighborhood metrics.")
    parser.add_argument("--k_eos_k", type=int, default=5, help="k components to remove for EOS_k.")
    parser.add_argument("--n_sub_eos_k", type=int, default=10, help="Subspace dimension for EOS_k overlap.")

    args = parser.parse_args()

    # Handle dummy dataset option
    if args.dataset_id == "list_dummy":
        dataset_input = [
            "This is the first sentence for testing.",
            "Here is another sentence, quite different from the first.",
            "Embeddings are numerical representations of text.",
            "Compressing embeddings can save space and speed up computations.",
            "Evaluation helps to understand the quality of compressed embeddings.",
            "The quick brown fox jumps over the lazy dog.",
            "Natural Language Processing is a fascinating field of study.",
            "Machine learning models are becoming increasingly complex.",
            "This framework aims to provide robust tools for NLP practitioners.",
            "Let's test the pipeline with these sample sentences."
        ] * 10 # Make it a bit larger for DR
        text_cols = None
        dataset_splt = None
    elif args.dataset_id == "hf_dataset_dummy":
        dataset_name_for_loader = "glue"
        dataset_config_for_loader = "mrpc" # Or any other valid GLUE config like 'sst2', 'cola', etc.
        
        dataset_input = (dataset_name_for_loader, dataset_config_for_loader) # Pass as a tuple

        # Determine text columns based on the chosen config
        if dataset_config_for_loader == "mrpc":
            text_cols = ["sentence1", "sentence2"]
        elif dataset_config_for_loader == "sst2":
            text_cols = ["sentence"]
        elif dataset_config_for_loader == "cola":
            text_cols = ["sentence"]
        # Add more elif blocks for other GLUE configs if needed
        else:
            # Fallback or raise error if text_cols are unknown for the config
            text_cols = None 
            print(f"Warning: Text columns for GLUE config '{dataset_config_for_loader}' not explicitly set.")

        dataset_splt = "train[:1%]" # Or "validation[:1%]" for mrpc
    else:
        dataset_input = args.dataset_id
        text_cols = args.text_columns.split(',') if args.text_columns else None
        dataset_splt = args.dataset_split


    pipeline = EmbeddingPipeline(model_name_or_path=args.model_name)
    
    results = pipeline.run(
        dataset_identifier=dataset_input,
        text_column_names=text_cols,
        dataset_split=dataset_splt,
        dr_method=args.dr_method,
        target_dim=args.target_dim,
        quantization_method=args.quant_method,
        output_dir=args.output_dir,
        k_val_for_local_metrics=args.k_local,
        k_for_eos_k=args.k_eos_k,
        n_sub_for_eos_k=args.n_sub_eos_k
    )

    print("\n--- Compression Info ---")
    print(json.dumps(results["compression_info"], indent=2))
    print("\n--- Evaluation Metrics ---")
    print(json.dumps(results["evaluation_metrics"], indent=2))
    print(f"\nResults saved to: {results['output_location']}")