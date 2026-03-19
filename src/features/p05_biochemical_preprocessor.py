import pandas as pd
from peptides import Peptide
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re
import os, sys
sys.path.append(os.path.abspath("../../etc/"))
from etc import constants_labels
from src.utils.worker import preprocessor_worker
from typing import Any, Dict, Optional, Union
from src.utils import logging_module
import numpy as np
logger = logging_module.get_logging(__name__)
class Biochemical_Preprocess(preprocessor_worker):
    def __init__(self, is_testing_data: bool = True) -> None:
        self.is_testing_data = is_testing_data
        super().__init__()
        # Pre-generate the list of all 343 possible triads

    def extract_biochemical_features(self, sequence: Union[str, float], col: str) -> dict[str, Any]:
        # Handle NaNs or empty strings
        if pd.isna(sequence) or  not str(sequence).strip():
            return {
                f"{col}_hydrophobicity": 0,
                f"{col}_isoelectric_point": 0,
                f"{col}_charge_ph7": 0,
                f"{col}_molecular_weight": 0,
                
                # 2. Stability & Structure
                f"{col}_aliphatic_index": 0,
                f"{col}_instability_index": 0,
                f"{col}_aromaticity": 0,
                
                # 3. Binding Potential
                f"{col}_boman_index": 0,
                f"{col}_flexibility": 0
            }
        
        # SANITIZATION: Remove spaces and non-AA characters
        # This keeps only A-Z and removes anything else (like spaces or '*')
        clean_seq = re.sub(r'[^A-Z]', '', str(sequence).upper())
        
        # Biopython's instability_index requires at least 2 AAs
        if len(clean_seq) < 2:
            return {
                f"{col}_hydrophobicity": 0,
                f"{col}_isoelectric_point": 0,
                f"{col}_charge_ph7": 0,
                f"{col}_molecular_weight": 0,
                
                # 2. Stability & Structure
                f"{col}_aliphatic_index": 0,
                f"{col}_instability_index": 0,
                f"{col}_aromaticity": 0,
                
                # 3. Binding Potential
                f"{col}_boman_index": 0,
                f"{col}_flexibility": 0
            }

        try:
            p = Peptide(clean_seq)
            analysed_seq = ProteinAnalysis(clean_seq)
            
            features = {
                # 1. Physicochemical
                f"{col}_hydrophobicity": p.hydrophobicity(),
                f"{col}_isoelectric_point": p.isoelectric_point(),
                f"{col}_charge_ph7": p.charge(pH=7.4),
                f"{col}_molecular_weight": p.molecular_weight(),
                
                # 2. Stability & Structure
                f"{col}_aliphatic_index": p.aliphatic_index(),
                f"{col}_instability_index": analysed_seq.instability_index(),
                f"{col}_aromaticity": analysed_seq.aromaticity(),
                
                # 3. Binding Potential
                f"{col}_boman_index": p.boman(),
                f"{col}_flexibility": sum(analysed_seq.flexibility()) / len(clean_seq)
            }
            return features
        except KeyError as e:
            logger.error(f"Skipping sequence {clean_seq} due to invalid character: {e}")
            return {
                f"{col}_hydrophobicity": 0,
                f"{col}_isoelectric_point": 0,
                f"{col}_charge_ph7": 0,
                f"{col}_molecular_weight": 0,
                
                # 2. Stability & Structure
                f"{col}_aliphatic_index": 0,
                f"{col}_instability_index": 0,
                f"{col}_aromaticity": 0,
                
                # 3. Binding Potential
                f"{col}_boman_index": 0,
                f"{col}_flexibility": 0
            }
            
    def extract_cdr_features(self,  sequence: Union[str, float], col: str) -> dict[str, Any]:
        if pd.isna(sequence) or  not str(sequence).strip():
            return {
                f"{col}_aliphatic_index": 0, # Stability
                f"{col}_boman_index": 0, # Potential
                f"{col}_isoelectric_point": 0, # Charge
                f"{col}_molecular_weight": 0,
                f"{col}_aromaticity": 0,
            }
        
        # SANITIZATION: Remove spaces and non-AA characters
        # This keeps only A-Z and removes anything else (like spaces or '*')
        clean_seq = re.sub(r'[^A-Z]', '', str(sequence).upper())

        try:
            p = Peptide(clean_seq)
            analysed_seq = ProteinAnalysis(clean_seq)
            
            features = {
                f"{col}_aliphatic_index": p.aliphatic_index(),
                f"{col}_boman_index": p.boman(),
                f"{col}_isoelectric_point": p.isoelectric_point(),
                f"{col}_molecular_weight": p.molecular_weight(),
                f"{col}_aromaticity": analysed_seq.aromaticity()
            }
            return features
        except KeyError as e:
            logger.error(f"Skipping sequence {clean_seq} due to invalid character: {e}")
            return {
                f"{col}_aliphatic_index": 0, # Stability
                f"{col}_boman_index": 0, # Potential
                f"{col}_isoelectric_point": 0, # Charge
                f"{col}_molecular_weight": 0,
                f"{col}_aromaticity": 0,
            }
    def extract_vh_vl_features(self, sequence: Union[str, float], col: str) -> dict[str, Any]:
        if pd.isna(sequence) or  not str(sequence).strip():
            return {
                f"{col}_isoelectric_point": 0,
                f"{col}_molecular_weight": 0,
            }
        
        # SANITIZATION: Remove spaces and non-AA characters
        # This keeps only A-Z and removes anything else (like spaces or '*')
        clean_seq = re.sub(r'[^A-Z]', '', str(sequence).upper())

        try:
            p = Peptide(clean_seq)
            analysed_seq = ProteinAnalysis(clean_seq)
            
            features = {
                f"{col}_isoelectric_point": p.isoelectric_point(),
                f"{col}_molecular_weight": p.molecular_weight(),
            }
            return features
        except KeyError as e:
            logger.error(f"Skipping sequence {clean_seq} due to invalid character: {e}")
            return {
                f"{col}_isoelectric_point": 0,
                f"{col}_molecular_weight": 0,
            }
    
    def get_existing_data(self, chunk: pd.DataFrame, col: str) -> pd.DataFrame:
        return chunk[chunk[col].notna() & (chunk[col] != 'ND')]
    
    def data_cleaning(self, chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.drop(columns=constants_labels.IGNORED_FEATURES)
        for genetic_seq in constants_labels.EXTRACTABLE_BIOSEQUENCE_FEATURES:
            chunk = self.get_existing_data(chunk, genetic_seq)
        return chunk
    
    
    def transform(self, chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = self.data_cleaning(chunk)
    
        feature_dfs = []
        
        for seq_col in ['CDRH3','CDRL3']:
            extracted = chunk[seq_col].apply(lambda val: self.extract_cdr_features(val, seq_col))
            df_feats = pd.DataFrame(list(extracted), index=chunk.index)
            feature_dfs.append(df_feats)
            
        for seq_col in ['VL','VHorVHH']:
            extracted = chunk[seq_col].apply(lambda val: self.extract_cdr_features(val, seq_col))
            df_feats = pd.DataFrame(list(extracted), index=chunk.index)
            feature_dfs.append(df_feats)
                
        final_chunk = pd.concat(feature_dfs, axis=1)
        final_chunk = final_chunk.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        final_chunk['name'] = chunk['Name'].values
        if final_chunk.isna().any().any():
            nan_cols = final_chunk.columns[final_chunk.isna().any()].tolist()
            logger.warning(f"Found NaNs in columns: {nan_cols[:10]}... (Total: {len(nan_cols)})")
        return final_chunk