import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from experiment import Experiment
import os
import json
import torch
import wandb
import numpy as np
from hydra.core.hydra_config import HydraConfig
from huggingface_hub import login
from dotenv import load_dotenv

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./")
def main(cfg):

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir

    exp = Experiment()

    if 'hparams' not in cfg:
        print(f"No config found: {cfg}")
        return

    wandb.init(project=os.environ['WANDB_PROJECT'],
               name=output_dir.replace("/", "_"))
    log.info(f"GPU: {torch.cuda.is_available()}")
    exp.run(cfg['hparams'], output_dir)
    wandb.finish()


if __name__ == "__main__":
    load_dotenv("environment.env")
    main()
