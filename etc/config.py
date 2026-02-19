# Raw Label Col.
BINDING_YES = "Binds to"
NOT_BINDING = "Doesn't Bind to"
NEUTRAL_YES = "Neutralising Vs"
NOT_NEUTRAL = "Not Neutralising Vs"

# Strain Config
TARGET = 'SARS-CoV2_WT'
NON_TARGETS = ['MERS-CoV','SARS-CoV1','HKU1','OC43','SARS-CoV1','229E']

# Raw features
EXTRACTABLE_BIOSEQUENCE_FEATURES = ['CDRH3','CDRL3','VL','VHorVHH'] # 'Heavy V Gene', 'Heavy J Gene', 'Light V Gene', 'Light J Gene'
IGNORED_FEATURES = ['Sources','Date Added','Last Updated','Update Description','Notes/Following Up?','ABB Homology Model (if no structure)', 'Structures']

# Amino Acid
AMINO_ACID_ALPHABETS = 'ACDEFGHIKLMNPQRSTVWY'

# Target Labels
BINDING_TARGET = f'is_binding_{TARGET}'
NEUTRAL_TARGET = f'is_neutral_{TARGET}'
IS_NANOBODY_COL = f'is_nanobody'
# Transformed Label List
