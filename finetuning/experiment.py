import numpy as np
import logging
import torch
import random
import transformers
import os
import json

from model import *
from datasets import load_dataset
from preprocessing import preprocess_dataset
from training import training

log = logging.getLogger(__name__)


class Experiment():

    def run(self, hparams, output_dir):
        result = {}

        tokenizer, model = create_model(hparams)
        dataset = load_dataset("locuslab/TOFU", hparams['dataset'])['train']

        dataset = preprocess_dataset(dataset, tokenizer, hparams)

        training(tokenizer, model, dataset, hparams['training'], output_dir)

        return result
