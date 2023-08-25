import copy
import random
from collections import defaultdict
from typing import Dict

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup

import laputamon.pytorch_util.laputa_optimizers as laputa_optimizers
import laputamon.pytorch_util.laputa_samplers as laputa_samplers
import laputamon.pytorch_util.laputa_schedulers as laputa_schedulers
from laputamon.pytorch_util.laputa_schedulers import (
    get_my_cosine_schedule_with_warmup,
    get_my_cosine_schedule_with_warmup_v2,
)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BasePLModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_summary = AverageMeter()

    def attach_train_dataset(self, train_dataset):
        self.train_dataset = train_dataset

    def instantiate_sampler(self):
        sampler_cls = getattr(laputa_samplers, self.config.sampler_config.name)
        return sampler_cls

    # def train_dataloader(self):
    #     sampler_cls = self.instantiate_sampler()
    #     sampler = sampler_cls(self.train_dataset, shuffle=True, drop_last=True)
    #     train_loader = DataLoader(
    #         self.train_dataset,
    #         batch_size=self.config.train_batch_size,
    #         sampler=sampler,
    #         shuffle=False,
    #         num_workers=self.config.num_workers,
    #         pin_memory=True,
    #         drop_last=True,
    #     )
    #     return train_loader

    def forward(self, input):
        return self.model.forward(input)

    def instantiate_loss_func(self):
        if self.config.loss_config.params is not None:
            loss_func = getattr(torch.nn, self.config.loss_config.name)(**self.config.loss_config.params)
        else:
            loss_func = getattr(torch.nn, self.config.loss_config.name)()
        return loss_func

    def custom_log(self, message: Dict):
        # for simple logging
        # if self.config.logger:
        #     self.config.logger.log(message)

        # for checkpoint callbacks
        for k, v in message.items():
            self.log(k, v, rank_zero_only=True)

    def calc_loss(self, loss_func, y_pred, batch):
        target_name = self.config.target_name
        y_true = batch[target_name]
        loss = loss_func(y_pred.view(-1), y_true)

        return loss

    def training_step(self, batch, batch_idx):
        y_pred = self.model.forward(batch)
        loss_func = self.instantiate_loss_func()
        loss = self.calc_loss(loss_func, y_pred, batch)

        self.loss_summary.update(loss, len(batch))

        return loss

    def on_train_epoch_end(self, outputs):
        all_summary = self.all_gather(self.loss_summary.avg.detach().cpu().numpy())
        self.custom_log({"train_loss": torch.mean(all_summary)})
        self.loss_summary.reset()

    def validation_step(self, batch, batch_idx):
        pred = self.model.forward(batch)
        target = batch[self.config.target_name]

        return {"pred": pred, "target": target}

    def on_validation_epoch_end(self, outputs):
        """do something with the all valid outputs"""
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        pred = self.model.forward(batch)
        return {"pred": pred}

    def test_epoch_end(self, outputs):
        """do something with the all test outputs"""
        raise NotImplementedError

    def configure_optimizers(self):
        # instantiate optimizer
        optimizer = self.instantiate_optimizer()
        lr_scheduler = self.instantiate_lr_scheduler(optimizer)

        # FIXME
        lr_scheduler_interval = self.config.lr_scheduler_config.interval

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": lr_scheduler_interval,
            },
        }

    def get_optimizer_params(self):
        """do something with the all test outputs"""
        raise NotImplementedError

    def instantiate_optimizer(self):
        optimizer_parameters = self.get_optimizer_params()

        if self.config.optimizer_config.name == "MADGRAD":
            optimizer = getattr(laputa_optimizers, self.config.optimizer_config.name)(
                optimizer_parameters, **self.config.optimizer_config.optimizer_params
            )
        else:
            optimizer = getattr(torch.optim, self.config.optimizer_config.name)(
                optimizer_parameters, **self.config.optimizer_config.optimizer_params
            )

        return optimizer

    def instantiate_lr_scheduler(self, optimizer):
        if self.config.lr_scheduler_config.name == "cosine_schedule_with_warmup":
            lr_params = self.config.lr_scheduler_config.lr_scheduler_params
            num_steps_per_epoch = lr_params["train_dataset_size"] // (self.config.train_batch_size * self.config.gpus)
            num_warmup_steps = int(lr_params["max_epochs"] * num_steps_per_epoch * lr_params["warmup_steps_ratio"])
            num_training_steps = int(lr_params["max_epochs"] * num_steps_per_epoch)
            num_cycles = 0.5
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            )
        elif self.config.lr_scheduler_config.name == "my_cosine_schedule_with_warmup":
            lr_scheduler = get_my_cosine_schedule_with_warmup(
                optimizer, **self.config.lr_scheduler_config.lr_scheduler_params
            )
        elif self.config.lr_scheduler_config.name == "my_cosine_schedule_with_warmup_v2":
            lr_scheduler = get_my_cosine_schedule_with_warmup_v2(
                optimizer, **self.config.lr_scheduler_config.lr_scheduler_params
            )
        else:
            try:
                lr_scheduler = getattr(laputa_schedulers, self.config.lr_scheduler_config.name)(
                    optimizer, **self.config.lr_scheduler_config.lr_scheduler_params
                )
            except:
                raise ValueError("Failured instantiate LR scheduler.")

        return lr_scheduler

    def adjust_output_length_for_each_process(self, phase="valid"):
        # 強制的にprocess数で割れるような数にbatchが補完されてしまうので、それを戻す
        # indexの分配規則は以下を参照
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/overrides/distributed.py#L81-L120
        if phase == "valid":
            org_dataset_size = len(self.trainer.val_dataloaders[0].dataset)
        elif phase == "predict":
            org_dataset_size = len(self.trainer.predict_dataloaders[0].dataset)
        else:
            raise ValueError()

        n_process = self.trainer.num_devices

        same_size_blocks = org_dataset_size // n_process
        residual = org_dataset_size - same_size_blocks * n_process
        indices = [same_size_blocks] * n_process
        if residual > 0:
            for r in range(residual):
                indices[r] += 1

        return indices
