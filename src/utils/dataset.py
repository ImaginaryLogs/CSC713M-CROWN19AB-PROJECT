import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, IterableDataset, DataLoader
from streaming import StreamingDataset, StreamingDataLoader
class LazyStreamingDataset(StreamingDataset):
    """Custom wrapper to ensure tensors are correctly typed for Lightning."""
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        obj = super().__getitem__(index)
        # MDS stores as ndarray; Lightning needs torch.Tensor
        x = torch.from_numpy(obj['features'].copy()).float()
        y = torch.tensor(obj['label'], dtype=torch.long)
        return x, y