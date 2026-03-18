from pathlib import Path
import pandas as pd
import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Subset,
    WeightedRandomSampler,
    random_split,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import dataset


class DataModule(LightningDataModule):
    def __init__(self, feature_files: list[str], label_file: str, batch_size=32, use_synthetic=False):
        super().__init__()
        self.feature_files = feature_files
        self.label_file = label_file
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.use_synthetic = use_synthetic

    def setup(self, stage=None):
        # 1. Load Labels
        master_df = pd.read_csv(self.label_file)

        # 2. Join all selected feature CSVs
        for f_file in self.feature_files:
            f_df = pd.read_csv(f_file)
            master_df = pd.merge(master_df, f_df, on='name', how='inner')

        # 3. Identify target and drop metadata
        # Ensure 'neutralizes' exists; adjusted to handle potential missing cols
        target_col = 'neutralizes' 
        metadata_cols = ['name', 'binds', target_col]
        
        X = master_df.drop(columns=[c for c in metadata_cols if c in master_df.columns]).values
        y = master_df[target_col].to_numpy()
        
        # 4. Initial Split (Real Data Only)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # 5. Scaling (Fit on Train, Transform others)
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        # 6. Synthetic Augmentation (Train Set Only)
        if self.use_synthetic and (stage == "fit" or stage is None):
            # For now, we "pass through" but keep the structure for SMOTE/Jittering
            # Note: You can add SMOTE here later.
            X_train_final, y_train_final = X_train, y_train 
            print(f"Training set: {np.bincount(y_train_final.astype(int))}")
        else:
            X_train_final, y_train_final = X_train, y_train

        # 7. Final Dataset Assignment
        if stage == "fit" or stage is None:
            self.train_ds = dataset.CDRDataset(X_train_final, y_train_final)
            self.val_ds = dataset.CDRDataset(X_val, y_val)
        
        if stage == "test" or stage is None:
            self.test_ds = dataset.CDRDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)