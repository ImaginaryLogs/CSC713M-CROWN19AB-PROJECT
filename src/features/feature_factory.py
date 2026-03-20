from enum import Enum, auto
from typing import Dict, Type, Union, List
import pandas as pd
from pathlib import Path
from src.features import p01_label_preprocessor, p02_naive_preprocessor, p03_motif_3kmer_preprocessor, p04_motif_conjoint_preprocessor, p05_biochemical_preprocessor
from src.utils import logging_module
from src.utils.worker import preprocessor_worker
import numpy as np
logger = logging_module.get_logging(__name__)

class FeatureType(Enum):
    LABELS = auto()
    NAIVE = auto()
    MOTIF_3KMER = auto()
    MOTIF_CONJOINT = auto()
    BIOCHEMICAL = auto()
    EMBEDDING = auto()
    
class FeatureFactory:
    def __init__(self) -> None:
        
        # Registry
        self._registry: Dict[FeatureType, Type[preprocessor_worker]] = {
            FeatureType.LABELS : p01_label_preprocessor.Label_Preprocess
            , FeatureType.NAIVE : p02_naive_preprocessor.Naive_Genetics_Preprocess
            , FeatureType.MOTIF_3KMER : p03_motif_3kmer_preprocessor.Motif_3kmer_Preprocess
            , FeatureType.MOTIF_CONJOINT : p04_motif_conjoint_preprocessor.Motif_Conjoint_Preprocess
            , FeatureType.BIOCHEMICAL : p05_biochemical_preprocessor.Biochemical_Preprocess
        }
        
    def get_worker(self, feature_type: FeatureType, **kwargs):
        worker_class = self._registry.get(feature_type)
        if not worker_class: 
            raise ValueError(f"Feature type {feature_type} not registered.")
        
        return worker_class(**kwargs)
    
    def extract_features_from_row(self, row: Union[pd.DataFrame, pd.Series], types: list[FeatureType]):
        final: List[pd.DataFrame] = []

        for worker in types:
            worker = self.get_worker(worker)
            final.append(worker.transform(row))
        final_result = pd.concat(final, axis=1)
        final_result = final_result.drop(labels=['name', 'Name'], errors='ignore')
        final_result = final_result.select_dtypes(include=[np.number])
        final_result = final_result.fillna(0.0)
        # Ensure the final object is a flat NumPy array
        if isinstance(final_result, pd.DataFrame):
            return final_result.to_numpy().flatten().astype(np.float32)
        elif isinstance(final_result, pd.Series):
            return final_result.to_numpy().astype(np.float32)
        
        return np.array(final_result).astype(np.float32) 
    
    def run_multi_feature_pipeline(self, 
                                   input_path: Path, 
                                   output_dir: Path, 
                                   types: list[FeatureType],
                                   generate_labels: bool = True):
        if generate_labels and FeatureType.LABELS not in types:
            types.insert(0, FeatureType.LABELS)
            
        for f_type in types:
            worker = self.get_worker(f_type)
            
            # Determine filename: labels.csv or {type}_features.csv
            if f_type == FeatureType.LABELS:
                out_name = "labels.csv"
            else:
                out_name = f"{f_type.name.lower()}_features.csv"
                
            out_path = output_dir / out_name
            
            logger.info(f"--- Running {f_type.name} Pipeline ---")
            worker.run_pipeline(input_path, out_path)