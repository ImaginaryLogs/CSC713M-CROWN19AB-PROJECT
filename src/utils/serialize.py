from streaming.base.format.mds import MDSWriter
import pandas as pd
from etc import constants_training, constants_labels
from pathlib import Path
from src.utils import logging_module
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from streaming import MDSWriter
import random
import torch # For memory-efficient tensor handling if needed
from src.features import feature_factory, negative_generator
from typing import Any, Union, Set, TypedDict, Dict
from src.features import p01_label_preprocessor
import traceback

logger = logging_module.get_logging(__name__)

def convert_to_mds(feature_csv: Path, label_csv: Path, output_dir: Path, target_yes_col: str, target_not_col: str):
    """
    Converts CSV to MDS format. 
    Labels are mapped to 1 (YES) and 0 (NOT). UNK is dropped.
    All other columns (excluding 'name' and labels) are treated as features.
    """
    logger.info(f"Indexing labels from {target_yes_col} and {target_not_col}...")
    df_labels = pd.read_csv(label_csv, usecols=['name', target_yes_col, target_not_col])
    
    # Logic: 
    # If YES column is 1 -> Label is 1
    # If NOT column is 1 -> Label is 0
    # If both are 0 (or it's UNK) -> Drop the row
    label_lookup = {}
    for _, row in df_labels.iterrows():
        if row[target_yes_col] == 1:
            label_lookup[row['name']] = 1
        elif row[target_not_col] == 1:
            label_lookup[row['name']] = 0
            
    logger.info(f"Found {len(label_lookup)} valid binary samples (YES/NOT).")

    # 2. Agnostic Feature Detection
    sample_feat = pd.read_csv(feature_csv, nrows=1)
    feature_cols = [c for c in sample_feat.columns if c != 'name']
    
    columns = {'features': 'ndarray:float32', 'label': 'int'}

    # 3. Write to MDS
    with MDSWriter(out=str(output_dir), columns=columns, compression='zstd') as out:
        for chunk in pd.read_csv(feature_csv, chunksize=constants_training.CHUNK_SIZE):
            for _, row in chunk.iterrows():
                row_name = str(row['name']).strip()
                
                if row_name in label_lookup:
                    feats = row[feature_cols].values.astype('float32')
                    out.write({
                        'features': feats,
                        'label': label_lookup[row_name]
                    })
    logger.info(f"Finished indexing labels {target_yes_col} and {target_not_col}")

class TaskWriters(TypedDict):
    train: MDSWriter
    val: MDSWriter
    test: MDSWriter

class TaskMetadata(TypedDict):
    train: Set[str]
    val: Set[str]
    test: Set[str]
    lookup: Dict[str, int]
    writers: TaskWriters
    
def _lightweight_split(
    df_meta_raw: pd.DataFrame, # Just the raw dataframe with labels
    output_base: Path,
    task: str,
    ):
    task_splits = {}
    
    cols = constants_labels.LABEL_MAPPING[task]
    # Identify valid samples for this task
    valid_mask = (df_meta_raw[cols["yes"]] == 1) | (df_meta_raw[cols["not"]] == 1)
    valid_meta: pd.DataFrame = df_meta_raw[valid_mask].copy()
    
    # Binary labels for stratification
    strat_labels = np.where(valid_meta[cols["yes"]] == 1, 1, 0)
    
    # 3-Way Split of NAMES only
    train_val_names, test_names = train_test_split(
        valid_meta['name'].values, 
        test_size=constants_training.SPLITS["test"], 
        stratify=strat_labels, 
        random_state=constants_training.DEFAULT_SEED
    )
    
    # Split train/val from the remaining 85%
    # 1. Get the mask for the 85% (train + val)
    train_val_df = valid_meta[valid_meta['name'].isin(train_val_names)]

    # 2. CRITICAL: Re-extract names and labels from the SAME filtered dataframe
    # This ensures lengths are identical: [10378, 10378]
    current_names = train_val_df['name'].values
    current_labels = np.where(train_val_df[cols["yes"]] == 1, 1, 0)

    train_names, val_names = train_test_split(
        current_names, 
        test_size=constants_training.SPLITS["val"], 
        stratify=current_labels, 
        random_state=constants_training.DEFAULT_SEED
    )

    task_out = output_base 
    
    writers = {
            "train": MDSWriter(out=str(task_out / "train"), columns={'features': 'ndarray:float32', 'label': 'int'}, compression='zstd'),
            "val": MDSWriter(out=str(task_out / "val"), columns={'features': 'ndarray:float32', 'label': 'int'}, compression='zstd'),
            "test": MDSWriter(out=str(task_out / "test"), columns={'features': 'ndarray:float32', 'label': 'int'}, compression='zstd')
    }
    
    lookup = dict(zip(valid_meta['name'], strat_labels))
    
    
    locs = {
        "train": set(train_names),
        "val": set(val_names),
        "test": set(test_names),
        "lookup": lookup,
        "writers": writers
    }
    task_splits[task] = locs
    return task_splits

def _mds_writer(
    chunk: pd.DataFrame, 
    row: pd.Series, 
    name: str, 
    meta: TaskMetadata, 
    local_rng: random.Random,
    factory_feature_extraction: feature_factory.FeatureFactory,
    features: list[feature_factory.FeatureType],
    oversample_factor: int = 1,
    use_synthetic: bool = True
):

    if name not in meta["lookup"]: return
    
    label = int(meta["lookup"][name])
    
    # Identify which split this row belongs to
    assigned_split = None
    if name in meta["train"]: assigned_split = "train"
    elif name in meta["val"]: assigned_split = "val"
    elif name in meta["test"]: assigned_split = "test"
    if not assigned_split: return

    # 1. Standard Extraction (for all splits)
    # We extract features only when we know we need them for a writer
    feats = factory_feature_extraction.extract_features_from_row(row, features)
    logger.info(f"{name} {feats.__str__()}")
    meta["writers"][assigned_split].write({'features': feats, 'label': label})

    # 2. Augmentation (TRAIN ONLY)
    if assigned_split == "train" and label == 1:
        # A. Oversampling
        for _ in range(oversample_factor - 1):
            meta["writers"]["train"].write({'features': feats, 'label': label})
        
        # B. Synthetic Hard Negatives
        if use_synthetic:
            shuffled_row = row.copy()
            shuffled_row['CDRH3'] = negative_generator.safe_shuffle(row['CDRH3'], local_rng)
            shuffled_row['CDRL3'] = negative_generator.safe_shuffle(row['CDRL3'], local_rng)
            shuffled_feats = factory_feature_extraction.extract_features_from_row(shuffled_row, features)
            meta["writers"]["train"].write({'features': shuffled_feats, 'label': 0})

def is_valid_sequence(seq) -> bool:
    """Checks if a sequence is present and biologically processable."""
    if pd.isna(seq) or str(seq).strip().upper() in ['ND', '', 'NONE', 'NAN']:
        return False
    return True

def _task_delegation(
    raw_csv_path: Path, 
    task_splits, 
    feature_factory: feature_factory.FeatureFactory, 
    feauters: list[feature_factory.FeatureType], 
    oversample_factor: int = 1, 
    use_synthetic: bool = False
):
    logger.info(f"Streaming and Extracting Features (Oversample: {oversample_factor}, Synthetic: {use_synthetic})")
    local_rng = random.Random(constants_training.DEFAULT_SEED)
    # Use chunksize to ensure we only have a few hundred rows of heavy data in RAM at once
    for chunk in tqdm(pd.read_csv(raw_csv_path, chunksize=constants_training.CHUNK_SIZE)):
        clean_chunk = chunk.dropna(subset=constants_labels.EXTRACTABLE_BIOSEQUENCE_FEATURES)
        clean_chunk = clean_chunk[~clean_chunk['CDRH3'].str.contains('ND', na=False)]
        
        for _, row in clean_chunk.iterrows():
            name = row['Name']
            
            if not is_valid_sequence(row.get('CDRH3')) or not is_valid_sequence(row.get('CDRL3')):
                logger.warning(f"Antibody {name} either is missing CDRH3 {not is_valid_sequence(row.get('CDRH3'))} or CDRL3 {not is_valid_sequence(row.get('CDRL3'))}")
                continue
            
            for task_name, meta in task_splits.items():
                _mds_writer(chunk, row, name, meta, local_rng, feature_factory, feauters, oversample_factor=oversample_factor, use_synthetic=use_synthetic)

def serialize_etl_data(
    raw_csv: Path,
    output_base: Path,
    task: str,
    oversample_factor: int = 1,
    use_synthetic_data = False
):
    # 1. Load Metadata
    # We load slightly more columns to satisfy the Label_Preprocess requirements
    meta_cols = ['Name', 'Ab or Nb', 'CDRH3', 'CDRL3'] + constants_labels.GROUPED_RAW
    df_raw = pd.read_csv(raw_csv, usecols=meta_cols)
    
    # 2. RUN LABEL PREPROCESSOR FIRST
    # This converts "Wuhan;Alpha" strings into 1s and 0s
    label_worker = p01_label_preprocessor.Label_Preprocess()
    processed_labels_df = label_worker.transform(df_raw)
    # Define Registry/Factory inside or import it
    factory = feature_factory.FeatureFactory()
    types = [
        feature_factory.FeatureType.NAIVE,
        feature_factory.FeatureType.MOTIF_CONJOINT,
        feature_factory.FeatureType.BIOCHEMICAL
    ]
    
    meta_tasks: Dict[str, TaskMetadata] = _lightweight_split(processed_labels_df, output_base, task)
    try:
        _task_delegation(raw_csv, meta_tasks, factory, types, oversample_factor, use_synthetic_data)
    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        # Guaranteed cleanup: Close writers even if extraction fails
        for meta in meta_tasks.values():
            k = meta["writers"]
            for w in k.values():
                w.finish() #type: ignore
                
    logger.info("Serialization complete. All shards flushed to disk.")
    

def serialize_all_data(
    feature_csvs: list[Path], 
    label_csv: Path, 
    output_base: Path, 
    tasks_config: dict, 
    use_synthetic: bool = False,
    oversample_factor: int = 1, # 1 means no extra copies, 2 means double binders
    chunk_size: int = constants_training.CHUNK_SIZE
):
    # 1. Load Labels & Secondary Features (Framework/Biochemical)
    labels_df = pd.read_csv(label_csv)
    secondary_lookups = {}
    for f_path in feature_csvs[1:]:
        df = pd.read_csv(f_path)
        cols = [c for c in df.columns if c != 'name']
        secondary_lookups[f_path.stem] = {
            row['name']: row[cols].values.astype('float32') for _, row in df.iterrows()
        }

    # 2. Setup Task Metadata & Splits
    task_metadata = {}
    # 2. Setup Task Metadata (Splits and Writers)
    for task_name, cols in tasks_config.items():
        valid = labels_df[(labels_df[cols["yes"]] == 1) | (labels_df[cols["not"]] == 1)].copy()
        
        # FIRST SPLIT: Separate 15% for the FINAL TEST (Holdout)
        train_val_names, test_names = train_test_split(
            valid['name'].tolist(), test_size=0.15, random_state=42, 
            stratify=np.where(valid[cols["yes"]] == 1, 1, 0)
        )
        
        # SECOND SPLIT: Split the remainder into Train (80%) and Val (20%)
        train_names, val_names = train_test_split(
            train_val_names, test_size=0.20, random_state=42,
            stratify=[labels_df.set_index('name').loc[n, cols["yes"]] for n in train_val_names]
        )
        
        task_out = output_base / task_name
        task_metadata[task_name] = {
            "train_set": set(train_names),
            "val_set": set(val_names),
            "test_set": set(test_names), # New: The 'Blind' Set
            "writers": {
                "train": MDSWriter(out=str(task_out / "train"), columns={'features': 'ndarray:float32', 'label': 'int'}, compression='zstd'),
                "val": MDSWriter(out=str(task_out / "val"),  columns={'features': 'ndarray:float32', 'label': 'int'}, compression='zstd'),
                "test": MDSWriter(out=str(task_out / "test"), columns={'features': 'ndarray:float32', 'label': 'int'}, compression='zstd')
            }
        }

    # 3. Stream and Intercept for Augmentation
    primary_csv = feature_csvs[0]
    local_rng = random.Random(constants_training.DEFAULT_SEED) # For reproducible shuffling

    try:
        for chunk in pd.read_csv(primary_csv, chunksize=chunk_size):
            primary_cols = [c for c in chunk.columns if c != 'name']
            
            for _, row in chunk.iterrows():
                name = row['name']
                feature_parts = [row[primary_cols].values.astype('float32')]
                
                # Fusion logic
                skip_row = False
                for lookup in secondary_lookups.values():
                    if name in lookup: feature_parts.append(lookup[name])
                    else: skip_row = True; break
                if skip_row: continue
                
                fused_vector = np.concatenate(feature_parts)
                
                for task_name, meta in task_metadata.items():
                    if name in meta["lookup"]:
                        label = meta["lookup"][name]
                        
                        # --- ROUTING & AUGMENTATION ---
                        if name in meta["val_set"]:
                            # RAW ONLY for Validation
                            meta["writers"]["val"].write({'features': fused_vector, 'label': label})
                        
                        elif name in meta["test_set"]:
                            meta["writers"]["test"].write({'features': fused_vector, 'label': label})
    
                        elif name in meta["train_set"]:
                            # 1. Write the original sample
                            meta["writers"]["train"].write({'features': fused_vector, 'label': label})
                            
                            # 2. If it's a BINDER (Label 1), apply Augmentations
                            if label == 1:
                                # A. Oversampling: Write extra copies
                                for _ in range(oversample_factor - 1):
                                    meta["writers"]["train"].write({'features': fused_vector, 'label': label})
                                
                                # B. Synthetic Hard Negatives: Shuffle and write as Label 0
                                if use_synthetic:
                                    # We shuffle the primary features (K-mers) 
                                    # Note: Shuffling K-mer counts is abstract, but it breaks the sequence motif order
                                    shuffled_primary = fused_vector.copy()
                                    
                                    meta["writers"]["train"].write({
                                        'features': shuffled_primary,
                                        'label': 0 # Shuffled binder becomes a 'Hard Negative'
                                    })
                            
    finally:
        for meta in task_metadata.values():
            for w in meta["writers"].values(): w.finish()

if __name__ == "__main__":
    PROCESSED_DIR = constants_training.ART_FEATURES_DIR
    convert_to_mds(
        feature_csv=PROCESSED_DIR / "motif_3kmer_features.csv",
        label_csv=PROCESSED_DIR / "labels.csv",
        output_dir=PROCESSED_DIR / "neutralization_mds",
        target_yes_col='neutral_yes',
        target_not_col='neutral_not'
    )