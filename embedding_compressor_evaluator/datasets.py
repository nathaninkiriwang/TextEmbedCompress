from datasets import load_dataset as hf_load_dataset
from .common_utils import logger
from collections.abc import Mapping, Sequence

def _grab_texts_from_row(item, text_column_names):
    """Extracts text from a single row/item based on specified or guessed columns."""
    if isinstance(item, str):
        yield item
    elif isinstance(item, Mapping):
        if text_column_names:
            for col_name in text_column_names:
                if col_name in item and isinstance(item[col_name], str):
                    yield item[col_name]
                elif col_name in item and isinstance(item[col_name], Sequence) and not isinstance(item[col_name], (str,bytes)):
                    for sub_item in item[col_name]:
                        if isinstance(sub_item, str): yield sub_item
        else: # Try to guess text columns
            for value in item.values():
                if isinstance(value, str):
                    yield value
                elif isinstance(value, list): # e.g. list of sentences
                    for sub_val in value:
                        if isinstance(sub_val, str):
                            yield sub_val
    elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)): # e.g. list of strings
        for sub_item in item:
            if isinstance(sub_item, str):
                yield sub_item


def load_and_extract_texts(dataset_identifier, text_column_names=None, split=None, local_file_type=None):
    """
    Loads a dataset and extracts all text fields.
    dataset_identifier: Hugging Face dataset name, path to local file, or list of texts.
    text_column_names: List of column names to extract text from (for HF datasets).
    split: Dataset split to use (e.g., 'train', 'test').
    local_file_type: 'csv', 'json', 'text' if dataset_identifier is a local path.
    """
    if isinstance(dataset_identifier, list) and all(isinstance(text, str) for text in dataset_identifier):
        logger.info(f"Using provided list of {len(dataset_identifier)} texts directly.")
        return dataset_identifier

    texts = []
    logger.info(f"Attempting to load dataset: {dataset_identifier}")
    try:
        if isinstance(dataset_identifier, tuple) and len(dataset_identifier) == 2:
            # This handles cases like ('glue', 'mrpc')
            dataset_name, dataset_config = dataset_identifier
            logger.info(f"Loading Hugging Face dataset '{dataset_name}' with config '{dataset_config}'")
            ds = hf_load_dataset(dataset_name, dataset_config, split=split)
        elif local_file_type:
            logger.info(f"Loading local file {dataset_identifier} as {local_file_type}")
            ds = hf_load_dataset(local_file_type, data_files=dataset_identifier, split=split if split else "train")
        else:
            logger.info(f"Loading Hugging Face dataset or local path: {dataset_identifier}")
            # This will try to load by path or single name (if no config needed)
            ds = hf_load_dataset(dataset_identifier, split=split)
        
        if not text_column_names and ds.column_names:
            # Basic guess for text columns if not specified
            potential_cols = [col for col in ['text', 'sentence', 'query', 'document', 'content'] if col in ds.column_names]
            if potential_cols:
                text_column_names = potential_cols
                logger.info(f"Guessed text columns: {text_column_names}")
            else: # If no standard names, try all string columns from the first example
                logger.warning("No text_column_names specified and could not guess common ones. Will try to extract all string fields.")


        for row in ds:
            texts.extend(list(_grab_texts_from_row(row, text_column_names)))

    except Exception as e:
        logger.error(f"Failed to load or process dataset {dataset_identifier}: {e}")
        # Re-raise the error or handle it as appropriate for your package
        raise e # It's often good to re-raise so the user sees the original datasets lib error


    if not texts:
        logger.warning(f"No texts extracted from dataset: {dataset_identifier}. Check dataset format and text_column_names.")
    else:
        logger.info(f"Extracted {len(texts)} text entries from dataset.")
    return list(set(texts)) # Return unique texts