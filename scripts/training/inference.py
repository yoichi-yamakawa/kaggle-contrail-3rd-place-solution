import gc
import glob
import os
import re
from typing import Any, Dict, Optional, Sequence

import albumentations as A
import fire
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import scripts.training.comp_datasets as comp_datasets
import scripts.training.comp_transform as comp_transform
import scripts.training.models as models
import scripts.training.pl_module as pl_module
import scripts.training.utils as comp_utils
from laputamon.io_util.directory_in_zip import DirectoryInZip
from laputamon.io_util.file_io import load_yaml_as_dataclass
from laputamon.pytorch_util.executor import LightningExecutor
from laputamon.pytorch_util.lightning_module import BasePLModule
from scripts.training.config import TrainingConfig
from scripts.training.logger import get_wandb_names, get_wandb_resume_id
from scripts.training.pl_callbacks import CustomWriter
from scripts.training.pl_module import BasePLM
from scripts.training.utils import make_gathered_result_file, timer
from scripts.training.validation import get_kfolds, get_stratified_group_kolds


class PLModuleWrapper(nn.Module):
    def __init__(self, model):
        self.model = model

    def forward(self, inputs):
        return self.model.inputs


def prepare_config(
    config_path,
    *overrides,
):

    config: TrainingConfig = load_yaml_as_dataclass(TrainingConfig, config_path, overrides)

    suf = "_debug" if config.debug else ""
    exp_dir = f"{config.model_dir}/{config.exp_name}{suf}/{config.fold_index}"

    config.sampler_config = None
    return config


def load_model(config, ckpt_path, in_kaggle: bool = True):
    model_params = config.model_config.params

    if in_kaggle:
        model_params["model_params"]["from_pretrained"] = False
        submission_out_dir = "./"
    else:
        submission_out_dir = exp_dir

    if model_params:
        model = getattr(models, config.model_config.name)(model_params)
    else:
        model = getattr(models, config.model_config.name)()

    model = PLModuleWrapper(model)
    model.load_state_dict(ckpt_path)

    return model


def inference_pipeline(
    config_path: str,
    *overrides: Sequence[str],
    ckpt_path: str = None,
    output_name: str = "submission.pkl",
    in_kaggle: bool = False,
):
    config = prepare_config(config_path, *overrides)
    test_df = get_test_df()

    test_dataloader = prepare_test_dataset(config)

    model = load_model(config, ckpt_path=ckpt_path, in_kaggle=in_kaggle)
    all_preds = inference(model, test_dataloader, device="cuda:0")

    pd.to_pickle(all_preds, output_name)


if __name__ == "__main__":
    fire.Fire(
        {
            "inference": inference_pipeline,
        }
    )
