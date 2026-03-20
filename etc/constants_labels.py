from pathlib import Path

# Raw Label Col.
BINDING_YES_RAW = "Binds to"
BINDING_NOT_RAW = "Doesn't Bind to"
NEUTRAL_YES_RAW = "Neutralising Vs"
NEUTRAL_NOT_RAW = "Not Neutralising Vs"

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

GROUPED_LABELS = [BINDING_YES_LABEL, BINDING_NOT_LABEL, BINDING_UNK_LABEL, NEUTRAL_YES_LABEL, NEUTRAL_NOT_LABEL, NEUTRAL_UNK_LABEL]
GROUPED_RAW = [BINDING_YES_RAW, BINDING_NOT_RAW, NEUTRAL_YES_RAW, NEUTRAL_NOT_RAW]

LABEL_MAPPING = {
    "binding": {
            "yes": BINDING_YES_LABEL, 
            "not": BINDING_NOT_LABEL 
    }, 
    "neutralization": {    
            "yes": NEUTRAL_YES_LABEL, 
            "not": NEUTRAL_NOT_LABEL
    }
}

