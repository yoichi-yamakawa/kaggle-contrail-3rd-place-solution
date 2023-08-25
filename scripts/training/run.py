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
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from torch import nn
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


# dataset preparation
def prepare_train_valid_dataset(config: TrainingConfig):
    train_df, valid_df = prepare_train_valid_df(
        config.train_dataset_dir,
        "train.csv",
        config,
    )

    train_transform_config = config.transform_config.train_transform
    valid_transform_config = config.transform_config.valid_transform
    train_transform = getattr(comp_transform, train_transform_config["name"])(
        **train_transform_config["transform_config"]
    )
    valid_transform = getattr(comp_transform, valid_transform_config["name"])(
        **valid_transform_config["transform_config"]
    )

    train_dataset_cls = getattr(comp_datasets, config.dataset_config.name)
    train_dataset = train_dataset_cls(
        train_df,
        config,
        "train",
        train_transform,
    )
    valid_dataset = train_dataset_cls(
        valid_df,
        config,
        "valid",
        valid_transform,
    )

    return train_dataset, valid_dataset, train_df, valid_df


def prepare_train_valid_df(
    dataset_dir: str,
    filename: str = "train.csv",
    config: TrainingConfig = None,
):
    train_dataset_dir = DirectoryInZip(dataset_dir)

    with timer("Loading data"):
        train_df = pd.read_pickle(f"{dataset_dir}/train.pkl")
        train_df["image_dir"] = config.train_image_dir
        train_df["from_train"] = True

        valid_df = pd.read_pickle(f"{dataset_dir}/valid.pkl")
        valid_df["image_dir"] = config.valid_image_dir
        valid_df["from_train"] = False

        if config.train_on_all_data:
            print("TRAIN ON ALL DATA!")
            train_df = pd.concat([train_df, valid_df]).reset_index(drop=True)

        if config.train_on_train_n_folds:
            print(f"TRAIN ON {config.n_folds}! fold:{config.fold_index}.")
            additional_train_df = valid_df.copy()

            train_idx, valid_idx = get_kfolds(
                train_df,
                n_folds=config.n_folds,
                is_stratified=False,
                shuffle=True,
                seed=773,
            )[config.fold_index]

            train_df, _valid_df = train_df.loc[train_idx].reset_index(drop=True), train_df.loc[valid_idx].reset_index(
                drop=True
            )

    with timer("Loading metadata"):
        train_meta_df = pd.read_json(f"{dataset_dir}/train_metadata.json")
        train_meta_df["record_id"] = train_df["record_id"].astype(str)
        valid_meta_df = pd.read_json(f"{dataset_dir}/validation_metadata.json")
        valid_meta_df["record_id"] = valid_df["record_id"].astype(str)

    if config.add_meta_data:
        train_df = train_df.merge(train_meta_df, on="record_id")
        valid_df = valid_df.merge(valid_meta_df, on="record_id")

    if config.debug:
        train_df = train_df.sample(30, random_state=773).reset_index(drop=True)
        valid_df = valid_df.iloc[:20]

    if config.pred_on_all_frame:
        train_df = pd.DataFrame(
            {"record_id": np.repeat(train_df["record_id"].values, 6), "frame": list(range(2, 8)) * len(train_df)}
        )
        train_df["image_dir"] = train_df.frame.apply(lambda x: config.train_image_dir)
        train_df["from_train"] = True

    if config.train_on_multi_frames:
        train_df = pd.DataFrame(
            {"record_id": np.repeat(train_df["record_id"].values, 6), "frame": list(range(2, 8)) * len(train_df)}
        )
        train_df["image_dir"] = train_df.frame.apply(lambda x: config.pseudo_label_dir)
        train_df["from_train"] = True

    if config.valid_on_multi_frames:
        valid_df = pd.DataFrame(
            {"record_id": np.repeat(valid_df["record_id"].values, 8), "frame": list(range(8)) * len(valid_df)}
        )
        valid_df["image_dir"] = valid_df.frame.apply(
            lambda x: config.valid_image_dir if x == 4 else config.pseudo_label_dir
        )

    print(f"base train_dataset_size: {len(train_df)}")
    print(f"base valid_dataset_size: {len(valid_df)}")

    total_train_size = len(train_df)

    if config.lr_scheduler_config.name == "cosine_schedule_with_warmup":
        config.lr_scheduler_config.lr_scheduler_params["train_dataset_size"] = total_train_size

    return train_df, valid_df


def prepare_test_df(
    config: TrainingConfig,
    in_kaggle: bool = False,
):
    if config.debug:
        test_df = test_df.loc[:300].reset_index(drop=True)

    if in_kaggle:
        image_dir = "/kaggle/input/google-research-identify-contrails-reduce-global-warming/test/"
    else:
        image_dir = config.test_image_dir

    record_ids = sorted([img_path for img_path in os.listdir(image_dir)])
    test_df = pd.DataFrame(
        {
            "record_id": record_ids,
        }
    )
    test_df["image_dir"] = image_dir
    test_df["from_train"] = False
    return test_df


def prepare_test_dataset(config: TrainingConfig, in_kaggle: bool = False):
    test_df = prepare_test_df(config, in_kaggle)

    test_transform_config = config.transform_config.test_transform
    test_transform = getattr(comp_transform, test_transform_config["name"])(**test_transform_config["transform_config"])
    test_dataset_cls = getattr(comp_datasets, config.dataset_config.name)
    test_dataset = test_dataset_cls(test_df, config, "test", test_transform)

    return test_dataset, test_df


def seed_everything(seed=773):
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run(
    config_path: str,
    *overrides: Optional[Sequence[str]],
    phase: str = "train",
    resume: bool = False,
):
    config: TrainingConfig = load_yaml_as_dataclass(TrainingConfig, config_path, overrides)

    seed_everything()

    # Set exp directory for callbacks
    config.exp_name = config_path.split("/")[-1].split(".")[0]
    if len(config.exp_name) > 50:
        raise ValueError()

    suf = "_debug" if config.debug else ""
    exp_dir = f"{config.model_dir}/{config.exp_name}{suf}/{config.fold_index}"
    os.makedirs(exp_dir, exist_ok=True)

    # Set logger
    if config.logger == "wandb":
        run_id = get_wandb_resume_id(exp_dir, resume)
        exp_name_dt = get_wandb_names(config)
        wandb_init_kwargs = {
            "resume": run_id,
            "project": f"p20-contrail-{config.exp_type}",
            "group": f"fold{config.fold_index}{suf}",
        }

        logger = WandbLogger(
            save_dir=exp_dir,
            name=exp_name_dt,
            log_model=False,
            config=config,
            **wandb_init_kwargs,
        )
    else:
        logger = None

    if phase in ["train", "valid"]:
        (
            train_dataset,
            valid_dataset,
            train_df,
            valid_df,
        ) = prepare_train_valid_dataset(config)

        if config.train_collate_fn is not None:
            train_collate_fn = getattr(comp_datasets, config.train_collate_fn)
            train_collate_fn = train_collate_fn(train_dataset.tokenizer)
        else:
            train_collate_fn = None

        if config.valid_collate_fn is not None:
            valid_collate_fn = getattr(comp_datasets, config.valid_collate_fn)
            valid_collate_fn = valid_collate_fn(valid_dataset.tokenizer)
        else:
            valid_collate_fn = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=train_collate_fn,
            pin_memory=False,
            drop_last=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=valid_collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        test_loader = None

    else:
        train_loader, valid_loader = None, None
        test_dataset = prepare_test_dataset(config)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    valid_loaders = [valid_loader]

    # define callbacks
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        monitor=config.monitor,
        verbose=True,
        dirpath=exp_dir,
        filename="best-{val_kaggle_score:.4f}-{epoch}",
        save_top_k=config.save_top_k,
        save_last=True,
        # every_n_epochs=config.every_n_epochs,
        mode="max",
    )
    callbacks.append(checkpoint_callback)

    if config.stochastic_weight_averaging:
        print("Apply SWA...")
        swa_callback = StochasticWeightAveraging(
            swa_epoch_start=config.swa_config.swa_epoch_start,
        )
        callbacks.append(swa_callback)

    use_distributed_sampler = True  # only False if you use custom Distributed Sampler

    if config.gpus > 1:
        sync_batchnorm = config.sync_batchnorm
        strategy = config.training_strategy
    else:
        sync_batchnorm = False
        strategy = "auto"

    trainer = pl.Trainer(
        sync_batchnorm=sync_batchnorm,
        strategy=strategy,
        accelerator="gpu",
        devices=config.gpus,
        precision=config.precision,
        max_epochs=config.max_epochs,
        reload_dataloaders_every_n_epochs=config.reload_dataloaders_every_n_epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        # val_check_interval=config.val_check_interval, # each n_steps(default: 1.0)
        # log_every_n_steps=config.log_every_n_steps,  # each n_steps(default: 50)
        num_sanity_val_steps=0,
        callbacks=callbacks,
        use_distributed_sampler=use_distributed_sampler,
        profiler="simple",
        gradient_clip_val=config.model_config.gradient_clip_val,
        # deterministic=True,
        logger=logger,
    )

    model_params = config.model_config.params

    if model_params:
        model = getattr(models, config.model_config.name)(model_params, phase=phase)
    else:
        model = getattr(models, config.model_config.name)()

    # use gradient ckpt
    # if phase == "train":
    #     model.model.gradient_checkpointing_enable()

    pl_module_cls = getattr(pl_module, config.pl_module)
    executor = LightningExecutor(trainer, pl_module_cls, model, train_loader, valid_loaders, test_loader, config)

    # use pretrained weights for training
    if "pretrained_weight_dir" in model_params["model_params"].keys():
        fold = 0
        pretraiend_weight_dir = model_params["model_params"]["pretrained_weight_dir"] + f"/{fold}"
        pretrained_weight_path = executor.get_best_ckpt_path(pretraiend_weight_dir)
        print(f"Pretrained weight path: {pretrained_weight_path}")

        executor.lazy_load_state_dict(pretrained_weight_path)
    else:
        pretrained_weight_path = None

    if phase == "train":
        if config.resume_from_checkpoint is not None:
            ckpt_path = config.resume_from_checkpoint
        else:
            last_ckpt_path = f"{exp_dir}/last.ckpt"
            # ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else None
            ckpt_path = None

        executor.fit(ckpt_path=ckpt_path)
        executor.slim_down_ckpt(exp_dir)


def predict_test(
    config_path: str,
    *overrides: Sequence[str],
    ckpt_path: str = None,
    output_name: str = "submission",
    in_kaggle: bool = False,
    tta_transform=None,
    pred_bbox_path=None,
):
    config: TrainingConfig = load_yaml_as_dataclass(TrainingConfig, config_path, overrides)

    suf = "_debug" if config.debug else ""
    config.exp_name = config_path.split("/")[-1].split(".")[0]
    exp_dir = f"{config.model_dir}/{config.exp_name}{suf}/{config.fold_index}"

    config.sampler_config = None
    test_dataset, test_df = prepare_test_dataset(config, in_kaggle)

    if pred_bbox_path is not None:
        print("--------")
        print("use_pred_masks")
        pred_box = np.load(pred_bbox_path)
        print(pred_bbox.shape)
        print("--------")
        test_dataset.pred_bbox = pred_bbox

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    if config.gpus > 1:
        strategy = "auto"
    else:
        strategy = "auto"

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

    if not in_kaggle:
        trainer = pl.Trainer(
            devices=config.gpus,
            strategy=strategy,
            profiler="simple",
            callbacks=CustomWriter(
                output_dir="./" if in_kaggle else exp_dir,
                filename=output_name,
                phase="test",
                write_interval="epoch",
            ),
        )
        pl_module_cls = getattr(pl_module, config.pl_module)
        executor = LightningExecutor(trainer, pl_module_cls, model, None, None, test_loader, config)

        del model
        torch.cuda.empty_cache()
        gc.collect()

        if not ckpt_path:
            print("checkpoint was not given. So search ckpt_path")
            ckpt_path = executor.get_best_ckpt_path(exp_dir)
            print(f"Use checkpoint:{ckpt_path}")
        else:
            print(f"Pretrained weight path: {ckpt_path}")
            executor.lazy_load_state_dict(ckpt_path)

        outputs = executor.predict(ckpt_path)
        preds = []
        for output in outputs:
            preds.append(output["pred"])

        outputs = np.concatenate(preds)
        pd.to_pickle(outputs, f"{submission_out_dir}/{output_name}.pkl")

        del outputs, executor, trainer, test_loader, preds
        torch.cuda.empty_cache()
        gc.collect()

    else:
        print("basic inference loop for Kaggle environment")
        model = PLModuleWrapper(model)
        ckpt = torch.load(ckpt_path)["state_dict"]
        model.load_state_dict(ckpt)

        all_preds = inference_loop(model, test_loader, device="cuda:0", precision=config.precision)
        pd.to_pickle(all_preds, f"{submission_out_dir}/{output_name}.pkl")

        del ckpt, model, all_preds
        torch.cuda.empty_cache()
        gc.collect()


class PLModuleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)


def inference_loop(model, dataloader, device="cuda:0", input_keys=["image"], precision="16-mixed"):
    import torchvision.transforms.functional as VF

    model.to(device)
    model.eval()

    use_amp = True if precision in [16, "16-mixed"] else False
    all_preds = []

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for i, batch in tqdm(enumerate(dataloader)):
            for k in input_keys:
                batch[k] = batch[k].to(device)
            preds = model(batch).sigmoid()
            preds = VF.resize(img=preds, size=(256, 256))
            preds = preds.detach().cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds).astype(np.float32)

    return all_preds


def predict_valid(
    config_path: str,
    *overrides: Sequence[str],
    output_name: str = "oof",
    tta_transform: Optional[str] = None,
    predict_train: bool = False,
    pred_bbox_path=None,
):
    config: TrainingConfig = load_yaml_as_dataclass(TrainingConfig, config_path, overrides)
    config.sampler_config = None

    config.exp_name = config_path.split("/")[-1].split(".")[0]
    suf = "_debug" if config.debug else ""
    exp_dir = f"{config.model_dir}/{config.exp_name}{suf}/{config.fold_index}"
    os.makedirs(exp_dir, exist_ok=True)

    train_dataset, valid_dataset, train_df, valid_df = prepare_train_valid_dataset(config)

    if predict_train:
        test_dataset = train_dataset
        test_dataset.phase = "test"
        test_dataset.transform = valid_dataset.transform
        test_df = train_df.copy()
    else:
        test_dataset = valid_dataset
        test_df = valid_df.copy()

    if pred_bbox_path is not None:
        pred_bbox = np.load(pred_bbox_path)
        test_dataset.pred_bbox = pred_bbox
    print("------------")
    print(len(test_dataset))
    print(len(test_df))
    print("------------")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    train_loader, valid_loader = None, None

    if config.save_pseudo_label:
        callbacks = None
    else:
        callbacks = CustomWriter(
            output_dir=exp_dir,
            filename=output_name,
            phase="test",
            write_interval="epoch",
        )
    trainer = pl.Trainer(
        devices=config.gpus,
        strategy="auto",
        profiler="simple",
        callbacks=callbacks,
    )

    model_params = config.model_config.params

    if model_params:
        model = getattr(models, config.model_config.name)(model_params)
    else:
        model = getattr(models, config.model_config.name)()

    pl_module_cls = getattr(pl_module, config.pl_module)
    executor = LightningExecutor(trainer, pl_module_cls, model, train_loader, valid_loader, test_loader, config)

    ckpt_path = config.ckpt_path

    if not ckpt_path:
        print("checkpoint was not given. So search ckpt_path")
        ckpt_path = executor.get_best_ckpt_path(exp_dir)
        print(f"Use checkpoint:{ckpt_path}")

    _ = executor.predict(ckpt_path)


def make_gathered_results(
    config_path: str,
    *overrides: Sequence[str],
    ckpt_path: str = None,
    phase="train",
    is_tta: bool = False,
    output_name: Optional[str] = None,
):
    config: TrainingConfig = load_yaml_as_dataclass(TrainingConfig, config_path, overrides)
    config.exp_name = config_path.split("/")[-1].split(".")[0]
    suf = "_debug" if config.debug else ""
    exp_dir = f"{config.model_dir}/{config.exp_name}{suf}/{config.fold_index}"

    if phase == "train":
        test_dataset, valid_dataset, train_df, valid_df = prepare_train_valid_dataset(config)
        target = test_dataset.targets
        output_name = "train_preds" if output_name is None else output_name
    elif phase == "valid":
        config.valid_on_every_n = None
        train_dataset, test_dataset, train_df, valid_df = prepare_train_valid_dataset(config)
        target = None
        output_name = "oof" if output_name is None else output_name
    elif phase == "test":
        test_dataset = prepare_test_dataset(config)
        target = None
        output_name = "submission" if output_name is None else output_name
    else:
        raise ValueError("Phase Error")

    if is_tta:
        output_name += "_tta"

    make_gathered_result_file(exp_dir, config.gpus, output_name)

    output = pd.read_pickle(f"{exp_dir}/{output_name}.pkl")

    if target is not None:
        output["target"] = target

    output["record_ids"] = test_dataset.record_ids

    pd.to_pickle(output, f"{exp_dir}/{output_name}.pkl")


if __name__ == "__main__":
    fire.Fire(
        {
            "run": run,
            "predict_test": predict_test,
            "predict_valid": predict_valid,
            "make_gathered_results": make_gathered_results,
        }
    )
