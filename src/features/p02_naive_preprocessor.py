import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from itertools import product
from dotenv import load_dotenv
from src.utils.worker import preprocessor_worker

import os, sys
from etc import constants_labels
from src.utils import logging_module
logger = logging_module.get_logging(__name__)


class Naive_Genetics_Preprocess(preprocessor_worker):
    def __init__(self, is_testing_data: bool = True) -> None:
        self.is_testing_data = is_testing_data
        super().__init__()

    
    def get_naive_biosequence_information(self, chunk: pd.DataFrame, df_processed: pd.DataFrame, col: str):
        temp_series = chunk[col].fillna("")
        lengths = temp_series.str.len()
        
        genetic_length_marker = f'{col}_length'
        
        new_cols = {
            genetic_length_marker : lengths.replace(0, 1) 
        }

        for aa in constants_labels.AMINO_ACID_ALPHABETS:
            counts = temp_series.str.count(aa)
            new_cols[f'{col}_amino_acid_percentage_{aa}'] = counts / new_cols[genetic_length_marker]
            
        new_cols[genetic_length_marker] = lengths
        
        new_df = pd.concat([df_processed, pd.DataFrame(new_cols)], axis=1)
        return new_df
    
    def get_one_hot_epitopes(self, chunk: pd.DataFrame):
        location_marker = 'Protein + Epitope'
        clean_col = chunk[location_marker].astype(str).str.upper()
        
        # Note: Order matters here to avoid 'Other_Spike' overlapping
        mappings = {
            'S_RBD': clean_col.str.contains('RBD'),
            'S_NTD': clean_col.str.contains('NTD'),
            'S_S2':  clean_col.str.contains('S2'),
            'S_S1':  clean_col.str.contains('S1'),
            'N_Protein': clean_col.str.contains('N') & ~clean_col.str.contains('S')
        }
        encoding_df = pd.DataFrame(mappings).astype(int)
        is_spike = clean_col.str.contains('S')
        hit_any_specific = encoding_df.any(axis=1)
        encoding_df['Other_Spike'] = (is_spike & ~hit_any_specific).astype(int)
        encoding_df['Unknown'] = (chunk[location_marker].isna() | 
                                clean_col.isin(['UNKNOWN', 'TBC', 'NAN']) | 
                                ~encoding_df.any(axis=1)).astype(int)
        
        return encoding_df
    
    def get_existing_data(self, chunk: pd.DataFrame, col: str) -> pd.DataFrame:
        return chunk[chunk[col].notna() & (chunk[col]) != 'ND']
    
    def data_cleaning(self, chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.drop(columns=constants_labels.IGNORED_FEATURES)
        for genetic_seq in constants_labels.EXTRACTABLE_BIOSEQUENCE_FEATURES:
            chunk = self.get_existing_data(chunk, genetic_seq)
        return chunk
    
    
    def transform(self, chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = self.data_cleaning(chunk)
        final_chunk = pd.DataFrame()
        for genetic_sequence in constants_labels.EXTRACTABLE_BIOSEQUENCE_FEATURES:
            final_chunk = self.get_naive_biosequence_information(chunk, final_chunk, genetic_sequence)
        location_chunk = self.get_one_hot_epitopes(chunk)
        final_chunk = pd.concat([final_chunk, location_chunk], axis = 1)
        final_chunk = final_chunk.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        final_chunk['name'] = chunk['Name'].values
        if final_chunk.isna().any().any():
            nan_cols = final_chunk.columns[final_chunk.isna().any()].tolist()
            logger.warning(f"Found NaNs in columns: {nan_cols[:10]}... (Total: {len(nan_cols)})")
        return final_chunk