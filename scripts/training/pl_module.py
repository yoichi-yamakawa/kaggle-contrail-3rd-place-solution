import math
import os
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef, mean_squared_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

import scripts.training.loss as loss
from laputamon.io_util.directory_in_zip import DirectoryInZip
from laputamon.pytorch_util.lightning_module import BasePLModule


class BasePLM(BasePLModule):
    def __init__(self, model, config):
        super(BasePLM, self).__init__(model, config)
        self.best_kaggle_metric = -1
        self.validation_step_outputs = []

    def train_dataloader(self):
        train_df = self.train_dataset.df

        cand1_idx = list(train_df.loc[(train_df.frame == 4)].index)

        if self.config.sampling_frame4:
            start_idx = self.current_epoch % 5
            cand1_idx = cand1_idx[start_idx::5]
        else:
            pass

        cand2_idx = list(train_df.loc[(train_df.frame != 4)].index)
        if self.config.max_epoch_using_pseudo_label >= self.current_epoch:
            start_idx = self.current_epoch % 5
            cand2_idx = cand2_idx[start_idx::5]
        else:
            cand2_idx = []
        sub_indices = cand1_idx + cand2_idx

        subset_data = Subset(self.train_dataset, sub_indices)
        train_dataloader = DataLoader(
            subset_data,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=None,
            pin_memory=False,
            drop_last=True,
        )

        return train_dataloader

    def instantiate_loss_func(self):
        try:
            loss_func = getattr(torch.nn, self.config.loss_config.name)
        except AttributeError:
            loss_func = getattr(loss, self.config.loss_config.name)

        if self.config.loss_config.params is not None:
            loss_func = loss_func(**self.config.loss_config.params)
        else:
            loss_func = loss_func()

        return loss_func

    def get_optimizer_params(self):
        no_decay = ["bias", "Layernorm.bias", "Layernorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in self.model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if "conv_lr" in self.config.optimizer_config.optimizer_params.keys():
            head_parameters = {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "model" not in n and not any(nd in n for nd in no_decay)
                ],
                "lr": self.config.optimizer_config.optimizer_params["conv_lr"],
            }

            optimizer_parameters.append(head_parameters)
            del self.config.optimizer_config.optimizer_params["conv_lr"]
            print("conv_lr appended !")

        return optimizer_parameters

    def calc_loss(self, loss_func, outputs, batch):
        loss = loss_func(outputs, batch)
        return loss

    def on_train_epoch_start(self):
        pass

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss_func = self.instantiate_loss_func()

        loss = self.calc_loss(loss_func, outputs, batch)
        self.loss_summary.update(loss.detach(), len(batch))

        self.log("train_loss_step", loss.detach(), prog_bar=True)
        self.log("lr", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0], prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        all_summary = self.all_gather(self.loss_summary.avg)
        self.custom_log({"train_loss": torch.mean(all_summary)})
        del all_summary
        self.loss_summary.reset()
        # torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idx = batch["idx"]
        y_pred = self.model(batch)
        target = batch["target"]
        output = {"idx": idx, "pred": y_pred, "target": target}

        self.validation_step_outputs.append(output)

        return output

    def concatenate_outputs_on_epoch_end(self, keys, outputs):
        """
        outputs: List of output
            output: {
            "pred": (shape: gpu_num x sample_num (of each process) x output)
            "target": (shape: gpu_num x sample_num (of each process) x output)
        }
        indices: [sample_num(gpu1), sample_num(gpu2), sample_num(gpu3), ... ]
        """
        oof_out = {}

        if self.config.gpus > 1:
            indices = self.adjust_output_length_for_each_process()
            outputs = self.all_gather(outputs)

            for key in keys:
                concatenated_out = torch.cat([out[key] for k, out in enumerate(outputs)], axis=1)
                concatenated_out = torch.cat(
                    [out[: indices[k]].view(-1, *out.shape[1:]) for k, out in enumerate(concatenated_out)]
                )
                oof_out[key] = concatenated_out
        else:
            for key in keys:
                concatenated_out = torch.cat([out[key] for k, out in enumerate(outputs)])
                oof_out[key] = concatenated_out

        return oof_out

    def calc_val_score(self, pred_binary, target):
        tp = (pred_binary * target).sum()
        pred_positive = pred_binary.sum()
        target_positive = target.sum()

        dice_score = 2 * tp / (pred_positive + target_positive)
        return dice_score

    def search_best_thresh(self, y_pred, y_true, thresh_min=0.00, thresh_max=1.01):
        val_score_on_best_thresh = -100
        best_thresh = 0.0

        for i, thresh in enumerate(np.arange(thresh_min, thresh_max, 0.01)):
            y_pred_binary = (y_pred > thresh).to(dtype=torch.int32)
            val_score = self.calc_val_score(y_pred_binary, y_true)
            if val_score_on_best_thresh < val_score:
                val_score_on_best_thresh = val_score
                best_thresh = thresh

            if i % 2 == 0:
                print(f"{thresh:.3f}:{val_score:.5f}")

        return val_score_on_best_thresh, best_thresh

    def on_validation_epoch_end(self):
        concat_keys = [
            "idx",
            "target",
            "pred",
        ]
        outputs = self.validation_step_outputs

        if len(self.trainer.val_dataloaders) == 1:
            outputs = [outputs]

        for dataloader_idx, outputs_for_signle_dataloader in enumerate(outputs):
            oof_out = self.concatenate_outputs_on_epoch_end(concat_keys, outputs_for_signle_dataloader)
            oof_preds = oof_out["pred"].sigmoid()

            if oof_preds.shape[2] != 256:
                oof_preds = VF.resize(oof_preds, size=(256, 256))

            # oof_preds = oof_preds.detach().cpu().numpy()
            # targets = oof_out["target"].detach().cpu().numpy()
            targets = oof_out["target"]
            print("-----------")
            print(oof_preds.shape)
            print(targets.shape)
            print("-----------")
            val_score_on_best_thresh, best_thresh = self.search_best_thresh(oof_preds, targets)

            self.update_best_score(val_score_on_best_thresh, method="max")
            suffix = dataloader_idx if dataloader_idx > 0 else ""

            self.custom_log(
                {
                    f"val_kaggle_score{suffix}": val_score_on_best_thresh,
                    # "val_loss": val_loss,
                    f"best_kaggle_score{suffix}": self.best_kaggle_metric,
                    f"best_thresh{suffix}": best_thresh,
                    f"output_len{suffix}": len(oof_preds),
                    "lr": self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
                }
            )

            del oof_preds, targets, outputs_for_signle_dataloader

        self.validation_step_outputs.clear()

    def update_best_score(self, score_on_epoch, method="max"):
        if method == "min":
            if self.best_kaggle_metric >= score_on_epoch:
                self.best_kaggle_metric = score_on_epoch
        else:
            if self.best_kaggle_metric < score_on_epoch:
                self.best_kaggle_metric = score_on_epoch

    def test_step(self, batch, batch_idx):
        pred = self.model(batch)

        return {"pred": pred}

    def predict_step(self, batch, batch_idx):
        idx = batch["idx"]
        y_pred = self.model(batch).sigmoid()

        # if self.config.train_on_multi_frames:
        #     new_dims = (256, 256)
        #     y_pred = F.interpolate(y_pred, size=new_dims, mode="bilinear", align_corners=False)
        #     y_pred = (y_pred > 0.5).to(dtype=torch.int)
        #

        if self.config.pred_on_all_frame:
            pl_dir = f"/mnt/nfs-mnj-hot-02/tmp/yo1mtrv/p20-contrail/pseudo_label/{self.config.exp_name}/"
            os.makedirs(pl_dir, exist_ok=True)
            new_dims = (256, 256)
            y_pred = F.interpolate(y_pred, size=new_dims, mode="bilinear", align_corners=False)
            y_pred = y_pred.detach().cpu().numpy()
            np.save(pl_dir + f"img{batch_idx}_{self.trainer.local_rank}.npy", y_pred)
            np.save(pl_dir + f"idx{batch_idx}_{self.trainer.local_rank}.npy", idx.detach().cpu().numpy())

        return {"idx": idx, "pred": y_pred}
