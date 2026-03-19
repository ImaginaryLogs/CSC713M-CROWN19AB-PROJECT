import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score
import torch
# Import your custom modules
from src.models.classical_ml import classical_ml
from src.features.feature_factory import FeatureFactory, FeatureType
from src.data_module.data_module import DataModule
from src.utils import logging_module, serialize, wandb_setter, name_generator, metrics, set_seed
from etc import constants_training, constants_labels
from typing import Any



logger = logging_module.get_logging(__name__)

def run_classical_training(model_key="rf", task="neutralization", oversampling_ratio:int = 1, has_synthetic_cdr: bool = False, **kwargs: Any):
    set_seed.set_seed()
    PROCESSED_DIR = constants_training.ART_FEATURES_DIR
    TASK_DIR = PROCESSED_DIR / task # 'binding' or 'neutralization'
    
    # Check if this specific task's MDS exists
    if not (TASK_DIR / "train" / "index.json").exists():
        # feature_files = [
        #     PROCESSED_DIR / "motif_3kmer_features.csv",
        #     PROCESSED_DIR / "biochemical_features.csv"
        # ]
        # labels_path = PROCESSED_DIR / "labels.csv"
        
        serialize.serialize_etl_data(constants_training.RAWDATA_FILE, PROCESSED_DIR)
        
        # serialize.serialize_all_data(
        #     feature_csvs=feature_files, # K-mer first (it's the biggest)
        #     label_csv=labels_path,
        #     output_base=PROCESSED_DIR,
        #     tasks_config=MY_TASKS,
        #     chunk_size=constants_training.CHUNK_SIZE
        # )
    
    # Point DataModule to the task-specific subfolders
    dm = DataModule(
        train_dir=str(TASK_DIR / "train"),
        val_dir=str(TASK_DIR / "val"),
        test_dir=str(TASK_DIR / "val") 
    )
    
    logger.info("Setting up Streaming DataModule...")
    dm.setup(stage="fit")

    assert dm.train_ds is not None, "Train dataset failed to initialize"    
    assert dm.val_ds is not None, "Validation dataset failed to initialize"
    
    # 3. Extract Numpy Arrays (The "Lazy-to-Eager" Bridge)
    # Since scikit-learn needs the full matrix, we collect it from the stream

    logger.info(f"Collecting {len(dm.train_dataset)} samples...")
    X_train, y_train = dm.get_full_arrays(stage="train")
    logger.info("WandB Initializiation...")
    # 5. Initialize WandB for Experiment Tracking
    wandb_setter.setup_wandb()
    flags = name_generator.get_config_bitmask(has_pca=has_pca, has_oversamp=oversampling_ratio>1, has_synthetic=has_synthetic_cdr)
    name = name_generator.get_run_name(model_key, task=task, flags=flags)
    run = wandb.init(
        project="CSC713M_MSINTSY",
        name=name,
        config={"model": model_key, "group": "classical_ml", "task": task}
    )

    # 6. Initialize and Train the Scikit-Learn Model
    logger.info(f"Training Classical Model: {model_key}")
    model_class = classical_ml.CLASSICAL_ML_CLASSIFIER[model_key]
    clf = model_class(random_state=42, **kwargs)
    clf.fit(X_train, y_train)
    X_val, y_val = dm.get_full_arrays(stage='val')
    
    assert isinstance(X_val, np.ndarray)
    assert isinstance(y_val, np.ndarray)
    # 7. Evaluation
    y_pred = clf.predict(X_val) 
    y_probas = clf.predict_proba(X_val)
    y_probas_pos = y_probas[:, 1]
    
    met = metrics.compute_and_format_metrics(
        y_true=y_val, 
        y_pred=y_pred, #type: ignore
        y_probs=y_probas_pos
    )
    logger.info(met)
    logger.info(f"{kwargs}")

    # 8. Log Results to WandB
    wandb.log(met)
    
    # Save the physical model file on your Arch system
    clf.save(directory=constants_training.ART_MODELS_DIR, name=name)
    run.finish()

if __name__ == "__main__":
    # You can iterate through your registry to compare models
    for model in ["rf", "knn","lr", "nb", "svm"]:
        for task in ["neutralization", "binding"]:
            for has_pca in [False, True]:
                for oversampling_ratio in [1, 2]:
                    for has_synthetic_cdr in [False, True]:
                        run_classical_training(model_key=model, has_pca=has_pca, task=task, oversampling_ratio=oversampling_ratio, has_synthetic_cdr=has_synthetic_cdr)
        