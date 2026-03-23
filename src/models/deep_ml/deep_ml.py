from sklearn.base import ClassifierMixin, BaseEstimator 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression as ScikitLR
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from typing import Any, Callable, Type, Union, cast
from sklearn.decomposition import PCA
import numpy as np
import types
import joblib
from etc import constants_training
from pathlib import Path
from src.utils.logging_module import get_logging
import lightning as L
import torch
import torch.nn as nn
from etc import constants_training
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
import wandb
from lightning.pytorch.loggers import WandbLogger
logger = get_logging(__name__)

class DeepMlClassifier(L.LightningModule):
    def __init__(self, 
                 input_dim, 
                 depth=constants_training.DEFAUTL_DEEPML_DEPTH, 
                 width=constants_training.DEFAUTL_DEEPML_WIDTH, 
                 dropout=constants_training.DEFAUTL_DEEPML_DROPOUT,
                 lr: float = constants_training.DEFAULT_LR,
                 class_weight=None,
                 weight_decay: float = constants_training.DEFAULT_WEIGHT_DECAY
    ):
        super().__init__()
        layers = []
        curr_dim = input_dim
        
        for i in range(depth):
            # We "funnel" the width down as we go deeper
            next_dim = width // (2**i) 
            layers.extend([
                nn.Linear(curr_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            curr_dim = next_dim
            
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(curr_dim, 1) # Final Classification
        if class_weight:
            w0 = class_weight.get(0, 1.0)
            w1 = class_weight.get(1, 1.0)
            pos_weight_val = torch.tensor([w1 / w0])
        else:
            pos_weight_val = torch.tensor([1.0])
        self.register_buffer("pos_weight", pos_weight_val)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.width = width
        self.depth = depth
        self.save_hyperparameters()
        self.train_metrics = MetricCollection(
            {"acc": Accuracy(task="binary", num_classes=2)},
            prefix="train_",
        )
        self.val_metrics = MetricCollection(
            {
                "acc": Accuracy(task="binary", num_classes=2),
                "f1": F1Score(task="binary", num_classes=2, average="macro"),
                "precision": Precision(task="binary", num_classes=2, average="macro"),
                "recall": Recall(task="binary", num_classes=2, average="macro"),
            },
            prefix="val_",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test_")
        self.test_preds: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []
        self.save_hyperparameters()
        
    
    def extra_repr(self) -> str:
        return (
            f"width={self.width}, "
            f"depth={self.depth}, "
            f"lr={self.lr}, "
            f"dropout={self.dropout}, "
            f"weight_decay={self.weight_decay}, "
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x.squeeze(1)

    def training_step(self, 
                      batch ,
                      batch_idx: int):
        x, y_true, _ = batch
        y_logits = self(x)
        
        loss = self.loss_fn(y_logits, y_true.float())
        self.log("train_loss", loss, prog_bar=True)
        y_prob = torch.sigmoid(y_logits)
        self.train_metrics(y_prob, y_true)
        self.log_dict(self.train_metrics, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(
        self, batch, batch_idx: int
    ) -> torch.Tensor:
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y.float())

        self.log("val_loss", loss, prog_bar=True)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def test_step(
        self, batch, batch_idx: int
    ) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.float())

        preds = (torch.sigmoid(logits) > 0.5).long()
        
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())
        
        self.test_metrics(preds, y)
        self.log_dict(self.test_metrics)
        
        return loss
    
    def on_test_epoch_start(self) -> None:
        self.test_preds.clear()
        self.test_targets.clear()
    
    def on_test_epoch_end(self) -> None:
        preds = torch.cat(self.test_preds).numpy().tolist()
        actuals = torch.cat(self.test_targets).numpy().tolist()

        if isinstance(self.logger, WandbLogger):
            # 2. Create the plot object
            cm_plot = wandb.plot.confusion_matrix(
                probs=None,
                y_true=actuals,
                preds=preds,
                class_names=["Non-Neutralizing", "Neutralizing"]
            )
            
            # 3. Log it using the 'experiment' attribute
            self.logger.experiment.log({"test/conf_matrix": cm_plot})
        self.test_preds.clear()
        self.test_targets.clear()
        
    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        return self(x)    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=3),
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }
        