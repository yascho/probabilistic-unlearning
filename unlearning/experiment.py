import numpy as np
import logging
import torch
import time
import random
import os
import json
from tqdm.auto import tqdm

from model import *
from dataloader import *
from unlearning import *

log = logging.getLogger(__name__)


class Experiment():

    def run(self, hparams, output_dir):
        datasets = load_datasets(hparams)
        tokenizer, model = create_model(hparams)
        set_random_seed(hparams['seed'])
        unlearn(hparams, tokenizer, model, datasets, output_dir)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
