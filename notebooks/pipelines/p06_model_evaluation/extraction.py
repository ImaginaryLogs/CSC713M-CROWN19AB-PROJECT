from pathlib import Path
from src.features.feature_factory import FeatureType, FeatureFactory
from src.data_module.data_module import DataModule

ROOT_DIR = Path(__file__).resolve(strict=True).parents[3]
PROCESSED_DIR =  ROOT_DIR / "artifacts" / "features"
types = [
    FeatureType.NAIVE,
    FeatureType.MOTIF_CONJOINT,
    FeatureType.BIOCHEMICAL
]

feature_slug = "_".join([t.name.lower() for t in types])

def search(task: str, oversampling_ratio: int, has_synthetic_cdr: bool):    
    data_config_name = f"{task}_ds{feature_slug}_ov{oversampling_ratio}_syn{has_synthetic_cdr}"
    TASK_DIR = PROCESSED_DIR / data_config_name 
    dm = DataModule(
        train_dir=str(TASK_DIR / "train"),
        val_dir=str(TASK_DIR / "val"),
        test_dir=str(TASK_DIR / "test") 
    )
    dm.setup(stage='fit')
    feature_names = dm.get_feature_names(types)
    X_val, y_val, names_val = dm.get_full_arrays(stage='val')
    feature_lookup = {name: X_val[i] for i, name in enumerate(names_val)}
    return feature_lookup

k = FeatureFactory