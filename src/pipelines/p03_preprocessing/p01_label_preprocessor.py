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
from etc import config



class Label_Preprocess(preprocessor_worker):
    def __init__(self, is_testing_data: bool = True) -> None:
        self.is_testing_data = is_testing_data
        self.encoder = self.composite_encoder()
        self.translation_array = [
              (config.BINDING_YES_LABEL,    self.encoder.encode_binding_yes_status)
            , (config.BINDING_NOT_LABEL,    self.encoder.encode_binding_not_status)
            , (config.BINDING_UNK_LABEL,    self.encoder.encode_binding_unk_status)
            , (config.NEUTRAL_YES_LABEL,    self.encoder.encode_neutral_yes_status)
            , (config.NEUTRAL_NOT_LABEL,    self.encoder.encode_neutral_not_status)
            , (config.NEUTRAL_UNK_LABEL,    self.encoder.encode_neutral_unk_status)
            , ('is_nanobody',               self.encoder.encode_antibody_status)
        ]
            
    
        super().__init__()

    class composite_encoder: 
        def __init__(self):
            pass
        
        def get_list_from_cell(self, cell: list[str]):
            """Safely converts semicolon-delimited strings to a set of strains."""
            if pd.isna(cell) or cell == '':
                return set()
            return set(str(cell).split(';'))
        
        def one_hot_encode(self, row: pd.Series, col: str, name: str):
            return False if pd.isna(row[col]) or row[col] == '' else name in str(row[col])
    
        def encode_antibody_status(self, row: pd.Series, name: str):
            return 1 if self.one_hot_encode(row, 'Ab or Nb', name) else 0
    
        def encode_neutral_yes_status(self, row: pd.Series, name: str) -> int:
            return 1 if self.one_hot_encode(row, config.NEUTRAL_YES, name) else 0
        
        def encode_neutral_not_status(self, row: pd.Series, name: str) -> int:
            return 1 if self.one_hot_encode(row, config.NEUTRAL_NOT, name) else 0
        
        def encode_neutral_unk_status(self, row: pd.Series, name: str) -> int:
            return 1 if not (self.encode_neutral_yes_status(row, name) or self.encode_neutral_not_status(row, name)) else 0

        def encode_binding_yes_status(self, row: pd.Series, name: str) -> int:
            return 1 if self.one_hot_encode(row, config.BINDING_YES, name) else 0
        
        def encode_binding_not_status(self, row: pd.Series, name: str) -> int:
            # If [Doesn't bind] == 'Wuhan', yes
            if self.one_hot_encode(row, config.BINDING_NOT, name):
                return 1
            # If [Binds] == 'Wuhan', No
            if self.one_hot_encode(row, config.BINDING_YES, name):
                return 0
            # If ([Binds] != Wuhan) AND ([Binds] == [Confirmed Non_Targets]), Yes
            if self.get_list_from_cell(row[config.BINDING_YES])\
                    .intersection(config.NON_TARGETS):
                return 1
            # Else, not enough information to confirm
            return 0
            
        def encode_binding_unk_status(self, row: pd.Series, name: str) -> int:
            return 1 if not (self.encode_binding_yes_status(row, name) or self.encode_binding_not_status(row, name)) else 0
    
    def get_existing_data(self, chunk: pd.DataFrame, col: str) -> pd.DataFrame:
        return chunk[chunk[col].notna() & (chunk[col]) != 'ND']
    
    def data_cleaning(self, chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.drop(columns=config.IGNORED_FEATURES)
        for genetic_seq in config.EXTRACTABLE_BIOSEQUENCE_FEATURES:
            chunk = self.get_existing_data(chunk, genetic_seq)
        return chunk
    
    def transform(self, chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = self.data_cleaning(chunk)
        final_chunk = pd.DataFrame()
        
        # For each feature, apply the necessary function
        # So here, for each label, apply the encoder
        for (ith, (outlabel, label_encoder)) in enumerate(self.translation_array):
            final_chunk[outlabel] = chunk.apply(lambda row: label_encoder(row, config.TARGET), axis=1)
        final_chunk['name'] = chunk['Name'] 
        
        return final_chunk