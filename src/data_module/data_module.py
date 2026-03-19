from pathlib import Path
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import dataset
from src.utils import logging_module
from streaming import StreamingDataset, StreamingDataLoader

logger = logging_module.get_logging(__name__)

class DataModule(LightningDataModule):
    def __init__(self, train_dir: str, val_dir: str, test_dir: str, batch_size: int = 1024, num_workers: int = 4):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Placeholders for datasets
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        
    def setup(self, stage: str = "fit"):
        """
        Official Logic: Initialize datasets pointing to MDS directories.
        Note: We don't load the data here, just the index.json metadata.
        """
        
        if stage == "fit" or stage is None:
            self.train_ds = dataset.LazyStreamingDataset(
                local=self.train_dir, 
                shuffle=True, 
                batch_size=self.batch_size
            )
            self.val_ds = dataset.LazyStreamingDataset(
                local=self.val_dir, 
                shuffle=False, 
                batch_size=self.batch_size
            )
            logger.info(f"Fit Stage: {len(self.train_ds)} train, {len(self.val_ds)} val samples.")
            
        elif stage == "test":
            self.test_ds = dataset.LazyStreamingDataset(
                local=self.val_dir,
                shuffle=False,
                batch_size=self.batch_size
            )
            logger.info(f"Test Stage: {len(self.test_ds)} test samples.")
            
            logger.info(f"Streaming DataModule initialized with {len(self.test_ds)} training samples.")
    
    def train_dataloader(self):
        # StreamingDataLoader is a drop-in replacement optimized for MDS
        return StreamingDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True, # Recommended for GPU training
            drop_last=True   # Prevents 'remainder' batches from skewing gradients
        )

    def val_dataloader(self):
        return StreamingDataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self):
        return StreamingDataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_full_arrays(self, stage="train")-> tuple[np.ndarray, np.ndarray]:
        """Bridge for Classical ML: Converts stream to Eager Numpy Arrays."""
        ds = self.train_ds if stage == "train" else self.val_ds
        if ds is None: raise ValueError("Dataset not initialized.")
            
        X_list, y_list = [], []
        first_shape = None

        for i, (x, y) in enumerate(ds):
            curr_shape = x.shape
            if first_shape is None:
                first_shape = curr_shape
            
            if curr_shape != first_shape:
                # This is your culprit!
                logger.error(f"Shape Mismatch at index {i}! Expected {first_shape}, got {curr_shape}")
                continue # Skip it for now to let the script finish
                
            X_list.append(x.numpy())
            y_list.append(y.item())
        X = np.stack(X_list)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array(y_list)
        return X, y 
    
    @property
    def train_dataset(self) -> StreamingDataset:
        if self.train_ds is None:
            raise RuntimeError("DataModule.setup() has not been called yet.")
        return self.train_ds

    @property
    def val_dataset(self) -> StreamingDataset:
        if self.val_ds is None:
            raise RuntimeError("DataModule.setup() has not been called yet.")
        return self.val_ds