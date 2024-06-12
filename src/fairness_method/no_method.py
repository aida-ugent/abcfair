import torch.nn.functional as F
from torch import sigmoid, from_numpy
import numpy as np
import pandas as pd
import wandb

from fairness_method.base import FairnessMethod


class NoMethod(FairnessMethod):
    name = "no_method"
