import wandb
import warnings
import numpy as np
import torch


def init_wandb(name, configs={}):
    wandb.login()
    wandb.init(project=name, config=configs)


def ignore_warnings():
    warnings.filterwarnings("ignore")


def fix_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
