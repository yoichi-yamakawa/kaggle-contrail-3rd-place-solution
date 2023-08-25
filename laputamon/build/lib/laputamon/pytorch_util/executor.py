import time
import typing
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader


class LightningExecutor:
    def __init__(
        self,
        trainer: pl.Trainer,
        model_class: pl.LightningModule,
        model: nn.Module,
        train_loader: Optional[DataLoader],
        valid_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        params: Dict,
    ):
        self.model_class = model_class
        self.trainer = trainer
        self.params = params

        # prepare
        self.build_model(model)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def build_model(self, model):
        # instanciate model
        model_kwargs = {"model": model, "config": self.params}
        self.pl_model = self.model_class(**model_kwargs)

    def lazy_load_state_dict(self, ckpt_path: Optional[str] = None):
        ckpt = torch.load(ckpt_path)["state_dict"]
        self.pl_model.load_state_dict(ckpt, strict=False)
        print(f"Loaded pretrained weight {ckpt_path}")

    def fit(self, ckpt_path: Optional[str] = None):
        self.pl_model.attach_train_dataset(self.train_loader.dataset)
        self.prepare_fit_kwargs()
        if ckpt_path:
            # self.fit_kwargs["ckpt_path"] = ckpt_path
            print(f"================================")
            print(f"Resume Training from:{ckpt_path}")
            print(f"================================")
        self.trainer.fit(**self.fit_kwargs, ckpt_path=ckpt_path)

    def validate(self, ckpt_path):
        self.prepare_validate_kwargs()
        self.validate_kwargs["model"].load_state_dict(torch.load(ckpt_path)["state_dict"])
        self.trainer.validate(**self.validate_kwargs)

    def predict(self, ckpt_path):
        self.prepare_predict_kwargs()
        self.predict_kwargs["model"].load_state_dict(torch.load(ckpt_path)["state_dict"])
        preds = self.trainer.predict(**self.predict_kwargs)
        return preds

    def prepare_fit_kwargs(self) -> None:
        self.fit_kwargs = {
            "model": self.pl_model,
            "train_dataloaders": None if self.params.reload_dataloaders_every_n_epochs > 0 else self.train_loader,
            "val_dataloaders": self.valid_loader,
        }

    def prepare_validate_kwargs(self) -> None:
        self.validate_kwargs = {
            "model": self.pl_model,
            "val_dataloaders": self.valid_loader,
        }

    def prepare_predict_kwargs(self) -> None:
        self.predict_kwargs = {"model": self.pl_model, "dataloaders": self.test_loader, "return_predictions": True}

    def get_best_ckpt_path(self, dirpath) -> str:
        callbacks = torch.load(dirpath + "/last.ckpt")["callbacks"]
        best_model_path = None
        for k, v in callbacks.items():
            if isinstance(v, dict):
                if "best_model_path" in v.keys():
                    best_model_path = v["best_model_path"]

        return best_model_path

    def slim_down_ckpt(self, dirpath) -> str:
        if self.trainer.local_rank == 0:
            print("Saving only best-model-path in last.ckpt")
            # last_ckpt = torch.load(dirpath + "/last.ckpt")
            # del_keys = ["state_dict", "optimizer_states"]

            # for k in del_keys:
            #     if k in last_ckpt.keys():
            #         del last_ckpt[k]

            # torch.save(last_ckpt, dirpath + "/last.ckpt")
            best_model_path = self.get_best_ckpt_path(dirpath)

            print(f"Saving only state_dict in {best_model_path}")

            best_ckpt = torch.load(best_model_path)

            del_keys = ["optimizer_states"]

            for k in del_keys:
                if k in best_ckpt.keys():
                    del best_ckpt[k]

            torch.save(best_ckpt, best_model_path)
