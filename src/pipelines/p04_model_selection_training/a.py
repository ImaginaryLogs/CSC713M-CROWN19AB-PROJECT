from typing import Any
from pathlib import Path
from lightning import LightningDataModule

class DataModule(LightningDataModule):
    def __init__(self, data_dir: Path, seed: int) -> None:
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
    
    
    
    
    
    