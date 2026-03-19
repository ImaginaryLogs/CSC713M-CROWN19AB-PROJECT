import torch, numpy as np
from typing import Optional
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall, MatthewsCorrCoef, AUROC
from etc import constants_training
from sklearn.metrics import classification_report
TRAIN_METRICS = MetricCollection(
    {"acc": Accuracy(task="multiclass", num_classes=constants_training.NUM_CLASSES)}
)
# Optimized for your Binary Neutralization Task
# Note: Use task="binary" if you only have 2 classes (Bind vs Not)
EVAL_METRICS = MetricCollection({
    "acc": Accuracy(task="binary"),
    "f1": F1Score(task="binary", average="macro"),
    "mcc": MatthewsCorrCoef(task="binary"),
    "auroc": AUROC(task="binary"),
    "precision": Precision(task="binary"),
    "recall": Recall(task="binary")
})

def compute_and_format_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: Optional[np.ndarray] = None, prefix: str = "val/"):
    """Bridge between Scikit-Learn NumPy and TorchMetrics."""
    # Convert NumPy to Torch Tensors
    t_true = torch.from_numpy(y_true).long()
    t_pred = torch.from_numpy(y_pred).long()
    
    # Update metrics
    results = EVAL_METRICS(t_pred, t_true)
    
    if y_probs is not None:
        # If it's already 1D (slicing happened outside), use it directly
        # If it's 2D (full predict_proba output), take the second column
        if y_probs.ndim == 2:
            probs_tensor = torch.from_numpy(y_probs[:, 1]).float()
        else:
            probs_tensor = torch.from_numpy(y_probs).float()
        if "auroc" in EVAL_METRICS:
            results["auroc"] = EVAL_METRICS["auroc"](probs_tensor, t_true)
    
    out = {f"{prefix}{k}": v.item() for k, v in results.items()}
    # Format for WandB: { 'val/acc': 0.85, ... }
    out["class_report"] = classification_report(t_true, t_pred, output_dict=True)
    return out

