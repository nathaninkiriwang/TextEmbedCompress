import torch
import logging
import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def to_tensor(data, device='cpu', dtype=torch.float32):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=dtype)
    return data.to(device=device)