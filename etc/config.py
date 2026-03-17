from pathlib import Path

# Raw Label Col.
BINDING_YES = "Binds to"
BINDING_NOT = "Doesn't Bind to"
NEUTRAL_YES = "Neutralising Vs"
NEUTRAL_NOT = "Not Neutralising Vs"

# Strain Config
TARGET = 'SARS-CoV2_WT'
NON_TARGETS = ['MERS-CoV','SARS-CoV1','HKU1','OC43','SARS-CoV1','229E']

# Raw features
EXTRACTABLE_BIOSEQUENCE_FEATURES = ['CDRH3','CDRL3','VL','VHorVHH'] # 'Heavy V Gene', 'Heavy J Gene', 'Light V Gene', 'Light J Gene'
IGNORED_FEATURES = [
    'Sources',
    'Date Added',
    'Last Updated',
    'Update Description',
    'Notes/Following Up?',
    'ABB Homology Model (if no structure)', 
    'Structures'
]

# Amino Acid
AMINO_ACID_ALPHABETS = 'ACDEFGHIKLMNPQRSTVWY'

# Target Labels
BINDING_TARGET = f'is_binding_{TARGET}'
NEUTRAL_TARGET = f'is_neutral_{TARGET}'
IS_NANOBODY_COL = f'is_nanobody'

BINDING_YES_LABEL = f'binding_yes'
BINDING_NOT_LABEL = f'binding_not'
BINDING_UNK_LABEL = f'binding_unk'
NEUTRAL_YES_LABEL = f'neutral_yes'
NEUTRAL_NOT_LABEL = f'neutral_not'
NEUTRAL_UNK_LABEL = f'neutral_unk'

# Path Config
ROOT_DIR = Path(__file__).resolve(strict=True).parents[1]
MODELS_DIR = ROOT_DIR / "artifacts" / "models"
FEATURES_DIR = ROOT_DIR / "artifacts" / "features"
RAWDATA_FILE = ROOT_DIR / "data" / "CoV-AbDab_080224.csv"

SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

DEFAULT_SEED = 42
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.01
CHUNK_SIZE = 100

#WANDB

WANDB_ENTITY="logarithmicpresence-de-la-salle-university"
WANDB_PROJECT="tumaini-model-training"