import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score

# Import your custom modules
from src.models.classical_ml import classical_ml
from src.features.feature_factory import FeatureFactory, FeatureType
from src.data_module.data_module import DataModule

def run_classical_training(model_key="rf"):
    # 1. Setup Paths
    RAW_DATA = Path("data/raw/cov_abdab_search_results.csv")
    PROCESSED_DIR = Path("data/processed")
    
    # 2. Ensure Features/Labels are Generated
    factory = FeatureFactory()
    # For Classical ML, we typically use Motif and Biochemical features
    feature_types = [FeatureType.MOTIF_3KMER, FeatureType.BIOCHEMICAL]
    
    factory.run_multi_feature_pipeline(
        input_path=RAW_DATA, 
        output_dir=PROCESSED_DIR, 
        types=feature_types,
        generate_labels=True
    )

    # 3. Use DataModule to Load and Scale Data
    feature_files = [
        str(PROCESSED_DIR / "motif_3kmer_features.csv"),
        str(PROCESSED_DIR / "biochemical_features.csv")
    ]
    dm = DataModule(
        feature_files=feature_files,
        label_file=str(PROCESSED_DIR / "labels.csv"),
        batch_size=1024, # Large batch size as we're just extracting arrays
        use_synthetic=True # SMOTE will run inside setup()
    )
    
    # Trigger the processing logic (Scaling, Splitting, Synthetic Data)
    dm.setup(stage="fit")

    # 4. Extract Numpy Arrays from DataModule
    # We access the internal datasets created by dm.setup()
    X_train = dm.train_ds.features.numpy()
    y_train = dm.train_ds.labels.numpy()
    
    X_val = dm.val_ds.features.numpy()
    y_val = dm.val_ds.labels.numpy()

    # 5. Initialize WandB for Experiment Tracking
    wandb.init(project="protein-classical-baselines", name=f"run-{model_key}")

    # 6. Initialize and Train the Scikit-Learn Model
    print(f"--- Training Classical Model: {model_key} ---")
    model_class = classical_ml.CLASSICAL_ML_CLASSIFIER[model_key]
    clf = model_class(random_state=42)
    
    clf.fit(X_train, y_train)

    # 7. Evaluation
    y_pred = clf.predict(X_val)
    y_probas = clf.predict_proba(X_val)[:, 1]

    # Calculate Research-Critical Metrics
    mcc = matthews_corrcoef(y_val, y_pred)
    auc = roc_auc_score(y_val, y_probas)
    
    print(f"Validation MCC: {mcc:.4f}")
    print(f"Validation AUC: {auc:.4f}")
    print(classification_report(y_val, y_pred))

    # 8. Log Results to WandB
    wandb.log({
        "val/mcc": mcc,
        "val/auc": auc,
        "model_type": model_key
    })
    
    # Save the physical model file on your Arch system
    clf.save(directory="checkpoints/classical", name=f"{model_key}_final.joblib")
    
    wandb.finish()

if __name__ == "__main__":
    # You can iterate through your registry to compare models
    for model in ["rf", "svm", "lr"]:
        run_classical_training(model_key=model)