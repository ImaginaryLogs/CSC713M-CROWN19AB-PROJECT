import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import torch

from src.models.deep_ml.deep_ml import DeepMlClassifier
from src.data_module.data_module import DataModule
from src.utils import logging_module, serialize, wandb_setter, name_generator, set_seed
from etc import constants_training
import wandb

logger = logging_module.get_logging(__name__)

def run_deep_training(
    task="neutralization", 
    oversampling_ratio=1, 
    has_synthetic_cdr=False, 
    depth=3, 
    width=1024, 
    **kwargs
):
    set_seed.set_seed()
    
    # 1. Directory & Data Setup (Reusing your logic)
    data_config_name = f"{task}_ov{oversampling_ratio}_syn{has_synthetic_cdr}"
    TASK_DIR = constants_training.ART_FEATURES_DIR / data_config_name 
    
    if not ((TASK_DIR / "train" / "index.json").exists()):
        logger.info(f"Generating NEW shards for {data_config_name}...")
        serialize.serialize_etl_data(
            constants_training.RAWDATA_FILE, 
            TASK_DIR, 
            task,
            oversample_factor=oversampling_ratio, 
            use_synthetic_data=has_synthetic_cdr
        )

    # 2. DataModule Setup (Streaming)
    dm = DataModule(
        train_dir=str(TASK_DIR / "train"),
        val_dir=str(TASK_DIR / "val"),
        test_dir=str(TASK_DIR / "test") 
    )
    dm.setup(stage="fit")

    # 3. Model Initialization
    # We get input_dim from a single sample to stay dynamic
    sample_batch = next(iter(dm.train_dataloader()))
    input_dim = sample_batch[0].shape[1] 
    logger.info(f"Input Dimension: {input_dim}")
    model = DeepMlClassifier(
        input_dim=input_dim, 
        depth=depth, 
        width=width,
        **kwargs
    )

    flags = name_generator.get_config_bitmask(
        has_oversamp=oversampling_ratio > 1, 
        has_synthetic=has_synthetic_cdr,
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
            "has_quantum_layer": False
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
    
    wandb.finish()

if __name__ == "__main__":
    for d in [2, 3, 4]:
        for w in [512, 1024]:
            for oversampling_ratio in [1, 2]:
                run_deep_training(depth=d, width=w, oversampling_ratio=oversampling_ratio)