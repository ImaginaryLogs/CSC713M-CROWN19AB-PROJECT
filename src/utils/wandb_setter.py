import logging, os, sys, wandb
from logging import Logger
from dotenv import load_dotenv


def setup_wandb() -> None:
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if api_key: 
        wandb.login(key=api_key)
    else:
        wandb.login()