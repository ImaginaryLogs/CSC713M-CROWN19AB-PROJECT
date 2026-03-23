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
import inspect

BASELINE_CLASSWEIGHT = {0: 1.0, 1: 1.0}

logger = logging_module.get_logging(__name__)

def run_classical_training_orchestrator(
    model_key="rf", 
    task="neutralization", 
    oversampling_ratio:int = 1, 
    has_synthetic_cdr: bool = False, 
    has_pca: bool = False, 
    class_weight: dict[int, float] = BASELINE_CLASSWEIGHT,
    selected_features: list[FeatureType] = [FeatureType.NAIVE],
    **kwargs: Any):
    
    set_seed.set_seed()
    PROCESSED_DIR = constants_training.ART_FEATURES_DIR
    feature_slug = "_".join([t.name.lower() for t in selected_features])
    data_config_name = f"{task}_ds{feature_slug}_ov{oversampling_ratio}_syn{has_synthetic_cdr}"
    TASK_DIR = PROCESSED_DIR / data_config_name 
    k = not ((TASK_DIR / "train" / "index.json").exists())
    logger.info(f"{k}")
    if not ((TASK_DIR / "train" / "index.json").exists()):
        logger.info(f"Generating NEW shards for {data_config_name}...")
        serialize.serialize_etl_data(
            constants_training.RAWDATA_FILE, 
            TASK_DIR,
            task,
            feature_types=selected_features,
            oversample_factor=oversampling_ratio, 
            use_synthetic_data=has_synthetic_cdr
        )
    else:
        logger.info(f"Using EXISTING shards for {data_config_name}")

    has_punished_weight = (class_weight[0] > class_weight[1])
    logger.info(f"Has weights: {class_weight}, punished weights: {has_punished_weight}")
    if oversampling_ratio > 1 and has_punished_weight:
        logger.warning("Caution: Combining oversampling and class weights. Results will be highly biased.")
    # Point DataModule to the unique folder
    dm = DataModule(
        train_dir=str(TASK_DIR / "train"),
        val_dir=str(TASK_DIR / "val"),
        test_dir=str(TASK_DIR / "test") 
    )
    
    logger.info("Setting up Streaming DataModule...")
    dm.setup(stage="fit")

    assert dm.train_ds is not None, "Train dataset failed to initialize"    
    assert dm.val_ds is not None, "Validation dataset failed to initialize"
    
    # 3. Extract Numpy Arrays (The "Lazy-to-Eager" Bridge)
    # Since scikit-learn needs the full matrix, we collect it from the stream

    logger.info(f"Collecting {len(dm.train_dataset)} samples...")
    X_train, y_train, _ = dm.get_full_arrays(stage="train")
    logger.info("WandB Initializiation...")
    # 5. Initialize WandB for Experiment Tracking
    wandb_setter.setup_wandb()
    flags = name_generator.get_config_bitmask(has_pca=has_pca, has_oversamp=oversampling_ratio>1, has_synthetic=has_synthetic_cdr, has_weight_imbalance=has_punished_weight)
    name = name_generator.get_run_name(model_key, task=task, flags=flags)
    run = wandb.init(
        project="CSC713M_MSINTSY",
        name=name,
        group="classical_ml",
        config={
            "model": model_key, 
            "task": task,
            "has_pca": has_pca,
            "has_synthetic_cdr": has_synthetic_cdr,
            "oversampling_ratio": oversampling_ratio,
            "class_weight": str(class_weight)
        }
    )

    # 6. Initialize and Train the Scikit-Learn Model
    logger.info(f"Training Classical Model: {model_key}")
    model_class = classical_ml.CLASSICAL_ML_CLASSIFIER[model_key]
    clf = model_class(random_state=42, has_pca=has_pca, class_weight=class_weight,**kwargs)
    clf.fit(X_train, y_train)
    X_val, y_val, names_val = dm.get_full_arrays(stage='val')
    
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
    
    logger.info(f"Generating detailed error audit for {model_key}...")
    metrics.log_confusion_cases(
        model_key=model_key,
        names=names_val, 
        y_actual=y_val,
        y_probas=y_probas,
        n_top=10
    )

    # 8. Log Results to WandB
    wandb.log(met)
    
    # Save the physical model file on your Arch system
    clf.save(directory=constants_training.ART_MODELS_DIR, name=name)
    run.finish()

types = [
    FeatureType.NAIVE,
    FeatureType.MOTIF_CONJOINT,
    FeatureType.BIOCHEMICAL
]



if __name__ == "__main__":    
    for task in ["neutralization", "binding"]:
        for has_pca in [False, True]:
            for oversampling_ratio in [1, 2]:
                for has_synthetic_cdr in [False, True]:
                    for model in ["rf", "xgb", "knn", "lr", "nb", "svm"]:
                        model_cls = classical_ml.CLASSICAL_ML_CLASSIFIER[model]
                        sig = inspect.signature(model_cls)
                        supports_weights = 'class_weight' in sig.parameters or model == "xgb"
                        
                        weight_scenarios: list[dict[int, float]] = [
                            {0: 2.0, 1: 1.0},  # Punish negatives more
                            {0: 1.5, 1: 1.0},
                            {0: 1.0, 1: 1.0}, # Baseline
                        ]
                        
                        if not supports_weights:
                            weight_scenarios = [{0: 1.0, 1: 1.0}] # Force only one run
                        for class_weight in weight_scenarios:
                            run_classical_training_orchestrator(model_key=model, has_pca=has_pca, task=task, oversampling_ratio=oversampling_ratio, has_synthetic_cdr=has_synthetic_cdr, class_weight=class_weight, selected_features=types)
    