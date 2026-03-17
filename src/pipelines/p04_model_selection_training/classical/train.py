import random, wandb
from etc import config

def train_ml_classical(
  classifier: str,
  feature_type: str,
  balanced: bool,
  seed: int,
) -> None:
  run = wandb.init(
      entity=config.WANDB_ENTITY
    , project=config.WANDB_PROJECT
    , config={
        "classifier": classifier
      , "feature_type": feature_type
      , "balanced": balanced
      , "seed": seed
    }
  )
  
  
  
  pass