import random, wandb
from etc import constants_labels

def train_ml_classical(
  classifier: str,
  feature_type: str,
  balanced: bool,
  seed: int,
) -> None:
  run = wandb.init(
      entity=constants_labels.WANDB_ENTITY
    , project=constants_labels.WANDB_PROJECT
    , config={
        "classifier": classifier
      , "feature_type": feature_type
      , "balanced": balanced
      , "seed": seed
    }
  )
  
  
  
  pass