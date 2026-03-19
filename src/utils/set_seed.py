from lightning.pytorch import seed_everything 
from src.utils import logging_module
from etc import constants_training

logger = logging_module.get_logging(__name__)

def set_seed(seed: int = constants_training.DEFAULT_SEED, workers: bool = True):
    seed_everything(seed, workers=workers)
    logger.info(f"Random seet set to: {seed}")