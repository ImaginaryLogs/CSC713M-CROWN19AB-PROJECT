from enum import Enum, auto
from typing import Dict, Type
import pandas as pd
from pathlib import Path
from src.features import p01_label_preprocessor, p02_naive_preprocessor, p03_motif_3kmer_preprocessor, p04_motif_conjoint_preprocessor, p05_biochemical_preprocessor

from src.utils import logging_module

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
        self._registry: Dict[FeatureType, Type] = {
            FeatureType.LABELS : p01_label_preprocessor
            , FeatureType.NAIVE : p02_naive_preprocessor
            , FeatureType.MOTIF_3KMER : p03_motif_3kmer_preprocessor
            , FeatureType.MOTIF_CONJOINT : p04_motif_conjoint_preprocessor
            , FeatureType.BIOCHEMICAL : p05_biochemical_preprocessor
        }
        
    def get_worker(self, feature_type: FeatureType, **kwargs):
        worker_class = self._registry.get(feature_type)
        if not worker_class: 
            raise ValueError(f"Feature type {feature_type} not registered.")
        
        return worker_class(**kwargs)
    
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