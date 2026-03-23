import torch, numpy as np
from typing import Optional
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall, MatthewsCorrCoef, AUROC
from etc import constants_training
import pandas as pd
from sklearn.metrics import classification_report
import wandb
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


def log_confusion_cases(model_key: str, names: np.ndarray, y_actual: np.ndarray, y_probas: np.ndarray, n_top=10):
    if y_probas.ndim == 2:
        y_probas = y_probas[:, 1]
        
    df = pd.DataFrame({
        'name': names,
        'y_actual': y_actual,
        'y_prob': y_probas,
        'pred': (y_probas >= 0.5).astype(int)
    })
    
    df['arrogance'] = np.abs(df['y_actual'] - df['y_prob'])
    df['uncertain'] = np.abs(df['y_prob'] - 0.5)

    audit_rows = []

    # Define the 4 categories
    categories = {
        "TP": (df['y_actual'] == 1) & (df['pred'] == 1),
        "TN": (df['y_actual'] == 0) & (df['pred'] == 0),
        "FP": (df['y_actual'] == 0) & (df['pred'] == 1),
        "FN": (df['y_actual'] == 1) & (df['pred'] == 0)
    }

    for cat_name, mask in categories.items():
        cat_df = df[mask]
        if cat_df.empty:
            continue

        # Get the most "Confident" in this category 
        # (For TP/TN these are 'Good' samples; for FP/FN these are 'Arrogant' errors)
        arrogant = cat_df.nlargest(min(len(cat_df), n_top), 'arrogance')
        for _, row in arrogant.iterrows():
            audit_rows.append([cat_name, "Confident", row['name'], row['y_prob'], row['y_actual']])

        # Get the most "Uncertain" (Closest to the 0.5 decision boundary)
        uncertain = cat_df.nsmallest(min(len(cat_df), n_top), 'uncertain')
        for _, row in uncertain.iterrows():
            audit_rows.append([cat_name, "Uncertain", row['name'], row['y_prob'], row['y_actual']])

    columns = ["Category", "Type", "Antibody_Name", "Confidence", "Actual"]
    table = wandb.Table(data=audit_rows, columns=columns)
    
    wandb.log({f"audit_confusion_cases_{model_key}": table})
