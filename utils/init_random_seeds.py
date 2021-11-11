import logging
import random
import torch
import numpy as np


def set_random_seed(seed):
    """
    Set the random seeds.
    """
    logging.info("Setting random seeds.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
