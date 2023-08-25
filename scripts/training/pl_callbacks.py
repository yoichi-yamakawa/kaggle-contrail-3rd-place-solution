import os
import time
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, filename: str, phase: str, write_interval: str = "epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.filename = filename
        self.phase = phase

    def write_on_batch_end(
        self,
        trainer,
        pl_module: "LightningModule",
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))

    def _check_save_completed(self, trainer, outdir: str):
        if os.path.exists(f"{outdir}/{self.filename}_{trainer.local_rank}.pkl"):
            return True
        else:
            print(f"{self.filename}_{trainer.local_rank}.pkl doesn't exist.")
            return False

    def _check_save_completed_on_all_ranks(self, trainer, outdir: str):
        device_num = len(trainer.device_ids)
        for i in range(device_num):
            if not os.path.exists(f"{outdir}/{self.filename}_{i}.pkl"):
                print(f"{self.filename}_{i}.pkl doesn't exist.")
                print("waiting 5 sec...")
                time.sleep(5)
                return False
        return True

    def write_on_epoch_end(
        self, trainer, pl_module: "LightningModule", predictions: List[Any], batch_indices: List[Any]
    ):
        # is_completed_on_local_rank = False
        # is_completed_on_all_ranks = False
        # print(predictions)
        # preds = predictions[0]
        all_result = {}

        for i, outputs in enumerate(predictions):
            # print(outputs)
            for k, v in outputs.items():
                all_result.setdefault(k, [])
                all_result[k].append(v)

        out_results = {}

        for k, v in all_result.items():
            out_results[k] = np.concatenate(v)

        # while not is_completed_on_local_rank:
        print(f"{self.output_dir}/{self.filename}_{trainer.local_rank}.pkl")
        pd.to_pickle(out_results, f"{self.output_dir}/{self.filename}_{trainer.local_rank}.pkl")
        trainer.strategy.barrier()  #  to ensure all processes are at the same line.
        # is_completed_on_local_rank = self._check_save_completed(trainer, self.output_dir)

        # if trainer.local_rank == 0:
        #     while not is_completed_on_all_ranks a:
        #         is_completed_on_all_ranks = self._check_save_completed_on_all_ranks(trainer, self.output_dir)

        # if trainer.local_rank == 0:
        #     if self.phase == "train":
        #         pd.to_pickle(trainer.val_dataloaders[0].dataset.df, f"{self.output_dir}/val_df.pkl")
        #     elif self.phase == "test":
        #         print(trainer.test_dataloaders[0])
        #         pd.to_pickle(trainer.test_dataloaders[0].dataset.df, f"{self.output_dir}/val_df.pkl")
