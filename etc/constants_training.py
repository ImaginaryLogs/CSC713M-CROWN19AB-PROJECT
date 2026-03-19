from pathlib import Path

# Path Config
ROOT_DIR = Path(__file__).resolve(strict=True).parents[1]
ART_MODELS_DIR = ROOT_DIR / "artifacts" / "models"
ART_FEATURES_DIR = ROOT_DIR / "artifacts" / "features"
RAWDATA_FILE = ROOT_DIR / "data" / "CoV-AbDab_080224.csv"


SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

DEFAULT_SEED = 42
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.01
CHUNK_SIZE = 2048
NUM_CLASSES = 2
# WANDB
WANDB_ENTITY="logarithmicpresence-de-la-salle-university"
WANDB_PROJECT="tumaini-model-training"

# MODEL SPECIFIC
SVM_MAX_ITER = 5000
PCA_N_COMPONENTS = 100