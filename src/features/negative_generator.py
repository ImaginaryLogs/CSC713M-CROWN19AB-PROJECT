from sklearn.model_selection import train_test_split
import random
from etc import constants_labels, constants_training
import pandas as pd
from src.utils import logging_module 

logger = logging_module.get_logging(__name__)

def safe_shuffle(seq: str, local_rng: random.Random):
    # Return as is if it's ND, NaN, or too short to shuffle meaningfully
    if pd.isna(seq) or str(seq).upper() == 'ND' or len(str(seq)) < 2:
        return seq
    
    # Convert to list, shuffle, and rejoin
    inner = list(str(seq)[1:-1])
    local_rng.shuffle(inner)
    return seq[0] + "".join(inner) + seq[-1]

def generate_hard_negatives(df, target_count, binding_col=constants_labels.BINDING_YES_RAW, neutral_col=constants_labels.NEUTRAL_YES_RAW):
    """
    Creates decoy negatives by shuffling CDR loops. 
    Maintains 'ND' and NaN values to preserve Nanobody vs Antibody logic.
    """
    
    # Sample from existing binders (Class 1) to ensure realistic scaffolds
    positives = df[df[binding_col] == constants_labels.TARGET].sample(n=target_count, replace=True if target_count > len(df) else False)
    negatives = positives.copy()


    # Apply shuffling only to the CDR loops
    # Shuffling the 'fingers' but keeping the Framework intact
    negatives['CDRH3'] = negatives['CDRH3'].apply(safe_shuffle)
    
    if 'CDRL3' in negatives.columns:
        negatives['CDRL3'] = negatives['CDRL3'].apply(safe_shuffle)

    # Mark as non-binding and non-neutralizing
    negatives[constants_labels.BINDING_NOT_RAW] = constants_labels.TARGET
    negatives[constants_labels.NEUTRAL_NOT_RAW] = constants_labels.TARGET
    negatives[binding_col] = None
    negatives[neutral_col] = None
    
    # Optional: Tag them to keep track them in EDA
    negatives['data_source'] = 'synthetic_negative'
    
    return negatives

def prepare_datasets(df, has_synthetic: bool = False):
    # 1. Initial Split (Test is locked away)
    train_val, test_df = train_test_split(
        df, test_size=constants_training.SPLITS["test"], stratify=df[constants_labels.TARGET], random_state=42
    )
    
    # 2. Secondary Split (Validation is locked away)
    train_df, val_df = train_test_split(
        train_val, test_size=constants_training.SPLITS["val"], stratify=train_val[constants_labels.TARGET], random_state=42
    )
    
    if has_synthetic:
        # 3. Augment ONLY the training set
        logger.info(f"Original Train Size: {len(train_df)}")
        
        # Generate hard negatives based on the training binders
        hard_negs = generate_hard_negatives(train_df, target_count=len(train_df)//2)
        train_df = pd.concat([train_df, hard_negs], ignore_index=True)
        
        logger.info(f"Augmented Train Size: {len(train_df)}")
    else:
        logger.info(f"Original Train Size used")
    return train_df, val_df, test_df