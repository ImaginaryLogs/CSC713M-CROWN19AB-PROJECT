import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from src.utils.logging_module import get_logging
from etc import constants_training, constants_labels
import traceback
from typing import Union

logger = get_logging(__name__)

class preprocessor_worker(ABC):
    def __init__(self, is_testing_data: bool = True):
        self.is_testing_data = is_testing_data
        self.chunk_size = constants_training.CHUNK_SIZE
    
    
    @abstractmethod
    def transform(self, chunk: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        pass
    
    
    def data_cleaning(self, chunk: Union[pd.DataFrame, pd.Series]):
        # Ensure chunk is always a DataFrame
        if isinstance(chunk, pd.Series):
            chunk = pd.DataFrame([chunk])
        return chunk
    
    def run_pipeline(self, input_path: Path, output_path: Path) -> bool:
        ith: int = 0
        worker_name: str = self.__class__.__name__
        
        try:
            reader = pd.read_csv(input_path, chunksize=self.chunk_size)
            
            for (i, chunk) in enumerate(reader):
                ith = i
                processed_chunk = self.transform(chunk)
                if i == 0:
                    processed_chunk.to_csv(output_path, mode='w', index=False, header=True)
                    logger.info(f"[{worker_name}] started. Created: {output_path.name}")
                else:
                    processed_chunk.to_csv(output_path, mode='a', index=False, header=False)
        
        except Exception as e: 
            logger.error(f"[{worker_name}] failed at iteration {ith} of chunk size {self.chunk_size}.\nERROR: {e}\n{traceback.print_exc(limit=None, file=None, chain=True)}")
            return False
        
        logger.info(f"[{worker_name}] finished succesfully.")
        return True