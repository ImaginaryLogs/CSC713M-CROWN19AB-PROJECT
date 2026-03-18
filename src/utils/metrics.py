import torch
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

from etc import constants_training

TRAIN_METRICS = MetricCollection(
    {"acc": Accuracy(task="multiclass", num_classes=constants_training.NUM_CLASSES)}
)

EVAL_METRICS = MetricCollection(
    {
        "acc": Accuracy(task="multiclass", num_classes=constants_training.NUM_CLASSES),
        "f1": F1Score(task="multiclass", num_classes=constants_training.NUM_CLASSES, average="macro"),
        "precision": Precision(
            task="multiclass", num_classes=constants_training.NUM_CLASSES, average="macro"
        ),
        "recall": Recall(task="multiclass", num_classes=constants_training.NUM_CLASSES, average="macro"),
    }
)


def format_metrics(
    metrics_dict: dict[str, torch.Tensor], prefix: str = ""
) -> dict[str, float]:
    return {f"{prefix}{k}": float(v) for k, v in metrics_dict.items()}
