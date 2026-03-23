import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch
import itertools
from src.models.deep_ml.deep_ml import DeepMlClassifier
from src.data_module.data_module import DataModule
from src.features.feature_factory import FeatureFactory, FeatureType

from src.utils import logging_module, serialize, wandb_setter, name_generator, set_seed, metrics
from etc import constants_training
import wandb
import numpy as np

logger = logging_module.get_logging(__name__)

def run_deep_training(
    task="neutralization", 
    oversampling_ratio: int = 1, 
    has_synthetic_cdr: bool = False, 
    has_pca: bool = False, 
    selected_features: list[FeatureType] = [FeatureType.NAIVE, FeatureType.MOTIF_CONJOINT, FeatureType.BIOCHEMICAL], # Added for parity
    class_weight = {0: 1.0, 1: 1.0},
    lr=constants_training.DEFAULT_LR,
    dropout=0.2,
    weight_decay=1e-4,
    depth=3, 
    width=1024, 
    **kwargs
):
    set_seed.set_seed()
    
    PROCESSED_DIR = constants_training.ART_FEATURES_DIR
    feature_slug = "_".join([t.name.lower() for t in selected_features])
    data_config_name = f"{task}_ds{feature_slug}_ov{oversampling_ratio}_syn{has_synthetic_cdr}"
    TASK_DIR = PROCESSED_DIR / data_config_name 
    
    if not ((TASK_DIR / "train" / "index.json").exists()):
        logger.info(f"Generating NEW shards for {data_config_name}...")
        serialize.serialize_etl_data(
            constants_training.RAWDATA_FILE, 
            TASK_DIR, 
            task,
            feature_types=selected_features, # Now explicit
            oversample_factor=oversampling_ratio, 
            use_synthetic_data=has_synthetic_cdr
        )
    logger.info("Test1")
    dm = DataModule(
        train_dir=str(TASK_DIR / "train"),
        val_dir=str(TASK_DIR / "val"),
        test_dir=str(TASK_DIR / "test") 
    )
    logger.info("Test2")
    dm.setup(stage="fit")       
    logger.info("Test3")

    sample_batch = next(iter(dm.train_dataloader()))
    logger.info("Test3.5")
    input_dim = sample_batch[0].shape[1] 
    logger.info(f"Input Dimension: {input_dim}")
    logger.info("Test4")
    model = DeepMlClassifier(
        input_dim=input_dim, 
        depth=depth, 
        width=width,
        class_weight=class_weight, 
        lr=lr,                        
        dropout=dropout,
        weight_decay=weight_decay
    )
    logger.info("Test4")
    flags = name_generator.get_config_bitmask(
        has_oversamp=oversampling_ratio > 1, 
        has_synthetic=has_synthetic_cdr,
        has_weight_imbalance=((class_weight[0] > 1) and (class_weight[1] > 1))
    )
    run_name = name_generator.get_run_name("deepNn", task=task, flags=flags)
    
    wandb_logger = WandbLogger(
        project="CSC713M_MSINTSY",
        name=run_name,
        group="deep_learning",
        config={
            "model": "deep_nn",
            "task": task,
            "depth": depth,
            "width": width,
            "oversampling_ratio": oversampling_ratio,
            "has_quantum_layer": False,
            "lr" : lr,
            "dropout" : dropout,
            "weight_decay" : weight_decay
        }
    )
    
    logger.info(f"Training Deep ML")
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            ModelCheckpoint(dirpath=constants_training.ART_MODELS_DIR, filename=run_name)
        ],
        log_every_n_steps=10
    )
    logger.info(f"Fitting Deep ML")
    trainer.fit(model, datamodule=dm)
    logger.info(f"Training Deep ML")
    trainer.test(model, datamodule=dm)
    
    X_test, y_actual, names_test = dm.get_full_arrays(stage='test')

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float().to(model.device)
        y_probas = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
        y_probas_full = np.stack([1 - y_probas, y_probas], axis=1) 

    metrics.log_confusion_cases(
        model_key="deepNn",
        names=names_test, 
        y_actual=y_actual,
        y_probas=y_probas_full,
        n_top=10
    )
    
    
    wandb.finish()

types = [
    FeatureType.NAIVE,
    FeatureType.MOTIF_CONJOINT,
    FeatureType.BIOCHEMICAL
]

weight_scenarios = [
    {0: 1.0, 1: 1.0},  # Baseline
    {0: 2.0, 1: 1.0},  # Punish negatives (High Precision focus)
    {0: 1.0, 1: 2.0},  # Punish positives (High Recall focus)
]

# Hyperparameter Grid
lrs = [1e-3, 1e-4]
dropouts = [0.2, 0.5]
weight_decays = [1e-4, 1e-2]
pca = False

if __name__ == "__main__":
    for scenario in weight_scenarios:
        for d, w, oversampling_ratio in itertools.product([2, 3, 4], [512, 1024], [1, 2]):
            for lr, dropout, weight_decay in itertools.product(lrs, dropouts, weight_decays):
                for task in ["neutralization", "binding"]:
                    run_deep_training(
                        task=task,
                        depth=d,
                        width=w,
                        oversampling_ratio=oversampling_ratio,
                        selected_features=types,
                        class_weight=scenario,
                        lr=lr,
                        dropout=dropout,
                        weight_decay=weight_decay,
                    )