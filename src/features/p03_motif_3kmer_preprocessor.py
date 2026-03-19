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

from etc import constants_biology, constants_labels
from src.utils import logging_module
logger = logging_module.get_logging(__name__)



class Motif_3kmer_Preprocess(preprocessor_worker):
    def __init__(self, is_testing_data: bool = True) -> None:
        self.is_testing_data = is_testing_data
        super().__init__()
        # Pre-generate the list of all 343 possible triads
    
    def get_3kmer_motif_triads_fast(self, chunk: pd.DataFrame, col: str):
        # We remove non-standard AAs by ignoring anything not in our 7 groups
        encoded_series = chunk[col].fillna("")
        
        # Prepare the CountVectorizer for 3-character "words"
        # analyzer='char' and ngram_range=(3,3) mimics the triad sliding window
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3), vocabulary=constants_biology.KMER3_NAMES, lowercase=False)
        triad_matrix = vectorizer.transform(encoded_series)
        lengths = encoded_series.str.len() - 2
        lengths = lengths.clip(lower=1) # Prevent div by zero
        
        triad_df = pd.DataFrame(
            triad_matrix.toarray() / lengths.values[:, None], # type: ignore
            columns=[f"{col}_3kmer_{t}" for t in constants_biology.KMER3_NAMES],
            index=chunk.index
        )
        
        return triad_df
    
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

        feature_conjoint_triads_dfs = []

        for seq in constants_labels.EXTRACTABLE_BIOSEQUENCE_FEATURES:
            ctd_df = self.get_3kmer_motif_triads_fast(chunk, seq)
            feature_conjoint_triads_dfs.append(ctd_df)
        
        final_chunk = pd.concat([final_chunk] + feature_conjoint_triads_dfs, axis=1)
        final_chunk = final_chunk.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        final_chunk['name'] = chunk['Name'].values
        if final_chunk.isna().any().any():
            nan_cols = final_chunk.columns[final_chunk.isna().any()].tolist()
            logger.warning(f"Found NaNs in columns: {nan_cols[:10]}... (Total: {len(nan_cols)})")
        return final_chunk