import gc
import os
import random
import string
from copy import deepcopy
from multiprocessing import Pool
from typing import Any, Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pfio
import torch
import torch.nn.functional as F
from pfio.cache import HTTPCache, MultiprocessFileCache
from torch.utils.data import Dataset
from tqdm import tqdm

import scripts.training.comp_transform as comp_transform
from laputamon.io_util.directory_in_zip import DirectoryInZip
from scripts.training.config import TrainingConfig

_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)

BAND08_MEAN_STD = (233.5, 7.3)
BAND09_MEAN_STD = (272.7, 20.1)
BAND10_MEAN_STD = (250.1, 11.7)
BAND11_MEAN_STD = (272.9, 20.0)
BAND12_MEAN_STD = (254.1, 13.5)
BAND13_MEAN_STD = (275.2, 19.5)
BAND14_MEAN_STD = (273.7, 21.5)
BAND15_MEAN_STD = (270.9, 21.1)
BAND16_MEAN_STD = (259.3, 16.1)


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def normalize_mean_std(data, mean, std):
    return (data - mean) / std


class ContrailDatasetBase(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        training_config: TrainingConfig,
        phase: str,
        transform: A.Compose = None,
    ) -> None:
        self.df = df
        self.ids = self.df.index.values
        if "frame" not in self.df.columns:
            self.df["frame"] = 4

        if "row_min" in self.df.columns:
            self.row_min = self.df["row_min"].values / 10135611.5
            self.row_size = self.df["row_size"].values / 230.16455078125
            self.col_min = self.df["col_min"].values / 663554.46875
            self.col_size = self.df["col_size"].values / 298.611083984375
        else:
            self.row_min = None
            self.row_size = None
            self.col_min = None
            self.col_size = None

        self.frames = self.df["frame"].values
        self.record_ids = self.df.record_id.values
        self.from_train = self.df.from_train.values
        self.pred_bbox = None

        self.config = training_config
        self.dataset_config = training_config.dataset_config
        self.normalize_method = self.dataset_config.params.get("normalize_method", "range")
        self.get_mask_frame_only = self.dataset_config.params.get("get_mask_frame_only", False)
        self.use_individual_mask = self.dataset_config.params.get("use_individual_mask", False)
        self.use_max_min = self.dataset_config.params.get("use_max_min", False)
        self.start_frame_idx = self.dataset_config.params.get("start_frame_idx", 0)
        self.end_frame_idx = self.dataset_config.params.get("end_frame_idx", 8)
        self.add_pseudo_label = self.dataset_config.params.get("add_pseudo_label", False)
        self.phase = phase

        self.transform = transform

        if self.phase == "train":
            self.image_dir_paths = self.df.image_dir.values
            cache_dir = f"/tmp/p20-contrail/{self.phase}"
            gc.collect()
        elif self.phase == "valid":
            self.image_dir_paths = self.df.image_dir.values
            cache_dir = f"/tmp/p20-contrail/{self.phase}"
            gc.collect()

        elif self.phase == "test":
            self.image_dir_paths = self.df.image_dir.values
            self.config.scs_dataset_dir = None
            cache_dir = None
            gc.collect()

        if cache_dir:
            self._cache = MultiprocessFileCache(len(self.df), dir=cache_dir, do_pickle=True)
        else:
            self._cache = None

        if self.config.scs_dataset_dir is None:
            self._scs_cache = None
            self._do_put = False
        else:
            debug_suffix = "_debug" if self.config.debug else ""
            self._scs_cache = None
            self._do_put = True

    def _read_image_cached(self, file_path: str):
        img = self._read_from_storage(file_path)

        return img

    def _read_from_storage(self, file_path: str):
        img = np.load(file_path)
        return img

    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        band11_path = f"{image_dir}/{record_id}/band_11.npy"
        band14_path = f"{image_dir}/{record_id}/band_14.npy"
        band15_path = f"{image_dir}/{record_id}/band_15.npy"

        band11 = self._read_image_cached(band11_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        if get_mask_frame_only:
            band11 = band11[:, :, frame]
            band14 = band14[:, :, frame]
            band15 = band15[:, :, frame]

        if self.normalize_method == "mean_std":
            band11 = normalize_mean_std(band11, BAND11_MEAN_STD[0], BAND11_MEAN_STD[1])
            band14 = normalize_mean_std(band14, BAND14_MEAN_STD[0], BAND14_MEAN_STD[1])
            band15 = normalize_mean_std(band15, BAND15_MEAN_STD[0], BAND15_MEAN_STD[1])

            r = band15 - band14
            g = band14 - band11
            b = band14

        elif self.normalize_method == "range":
            r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
            g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
            b = normalize_range(band14, _T11_BOUNDS)

        else:
            ValueError("Unknown `normalize_method`")
        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

        return false_color

    def crop_by_preds(self, image, pred):  # image: (n_frames, 256, 256, ch), pred: (256, 256)
        pos_idx = np.where(pred > 0)
        if len(pos_idx[0]) > 0:
            y_min = pos_idx[0].min()
            y_max = pos_idx[0].max() + 1
            x_min = pos_idx[1].min()
            x_max = pos_idx[1].max() + 1
            h = y_max - y_min
            w = x_max - x_min
            h = max(h, 192)
            w = max(w, 192)
            y = (y_min + y_max) // 2
            x = (x_min + x_max) // 2
            x_min = max(int(x - w / 2) - 1, 0)
            x_max = min(int(x + w / 2) + 1, 256)
            y_min = max(int(y - h / 2) - 1, 0)
            y_max = min(int(y + h / 2) + 1, 256)
        else:
            x_min = 0
            x_max = 256
            y_min = 0
            y_max = 256
        image = [_image[y_min:y_max, x_min:x_max] for _image in image]
        return image, y_min, y_max, x_min, x_max

    def get_target_mask(
        self,
        human_pixel_path: str,
        use_individual_mask: bool = False,
        use_max_min: bool = False,
        from_train: bool = False,
    ) -> np.ndarray:
        target = self._read_image_cached(human_pixel_path)
        if len(target.shape) == 2:
            target = target[..., np.newaxis]

        if use_individual_mask:
            if from_train:
                individual_mask_path = human_pixel_path.replace("human_pixel_masks.npy", "human_individual_masks.npy")
                individual_target = self._read_image_cached(individual_mask_path)
                target_mean = individual_target.mean(axis=3)
            else:
                target_mean = target.copy().astype(np.float32)

            if not use_max_min:
                return (target, target_mean)
            else:
                target_max = individual_target.max(axis=3)
                target_min = individual_target.min(axis=3)
                return (target, target_mean, target_max, target_min)
        else:
            return target

    def get_position_image(self, row_min: float, row_size: float, col_size: float, col_min: float):
        size = self.config.transform_config.train_transform["transform_config"]["params"]["Resize"]["width"]
        rm = torch.ones((1, size, size)) * row_min
        rs = torch.ones((1, size, size)) * row_size
        cm = torch.ones((1, size, size)) * col_min
        cs = torch.ones((1, size, size)) * col_size

        pos_img = torch.cat([rm, rs, cm, cs], axis=0)

        return pos_img

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        record_id = self.record_ids[index]
        frame = self.frames[index]
        image_dir = self.image_dir_paths[index]
        from_train = self.from_train[index]
        img = self.get_ash_color_images(record_id, image_dir, get_mask_frame_only=self.get_mask_frame_only, frame=frame)

        if not self.get_mask_frame_only:
            img = img.transpose(0, 1, 3, 2).reshape(256, 256, -1)

        if self.pred_bbox is not None:
            bbox = self.pred_bbox[index]
            y_min, y_max, x_min, x_max = bbox
            img = img[y_min:y_max, x_min:x_max, :]

        if self.transform:
            t_image = {"image": img}

            if self.phase in ["train"]:
                if frame == 4:
                    human_pixel_path = f"{image_dir}/{record_id}/human_pixel_masks.npy"
                else:
                    human_pixel_path = f"{image_dir}/{record_id}/frame{frame}.npy"
                target = self.get_target_mask(
                    human_pixel_path, self.use_individual_mask, use_max_min=self.use_max_min, from_train=from_train
                )

                if self.use_individual_mask:
                    t_image["mask"] = target[0]
                    t_image["mask1"] = target[1]
                    if self.use_max_min:
                        t_image["mask2"] = target[2]
                        t_image["mask3"] = target[3]
                else:
                    t_image["mask"] = target

                if self.add_pseudo_label:
                    for k, frame in enumerate(range(self.start_frame_idx, self.end_frame_idx + 1)):
                        pseudo_human_pixel_path = f"{self.config.pseudo_label_dir}/{record_id}/frame{frame}.npy"
                        pseudo_label = self.get_target_mask(
                            pseudo_human_pixel_path,
                            use_individual_mask=False,
                            use_max_min=self.use_max_min,
                            from_train=from_train,
                        )
                        t_image[f"mask{2+k}"] = pseudo_label

            t_image = self.transform(**t_image)

            img = t_image["image"]

        if self.phase in ["train", "valid"]:
            if self.phase == "train":
                target = t_image["mask"].permute(2, 0, 1)
            else:
                human_pixel_path = f"{image_dir}/{record_id}/human_pixel_masks.npy"
                target = self.get_target_mask(human_pixel_path, False)
                target = torch.tensor(target.transpose(2, 0, 1), dtype=torch.float32)

            output = {
                "idx": torch.tensor([self.ids[index]]),
                "image": img,
                "target": target,
                "from_train": from_train,
            }
            if self.phase == "train":
                if self.use_individual_mask:
                    output["individual_mask"] = t_image["mask1"].permute(2, 0, 1)
                    if self.use_max_min:
                        output["individual_mask_max"] = t_image["mask2"].permute(2, 0, 1)
                        output["individual_mask_min"] = t_image["mask3"].permute(2, 0, 1)

                if self.add_pseudo_label:
                    for k, frame in enumerate(range(self.start_frame_idx, self.end_frame_idx + 1)):
                        output[f"pseudo_mask{k}"] = t_image[f"mask{2+k}"].permute(2, 0, 1)

            if self.row_min is not None:
                output["row_min"] = torch.tensor(self.row_min[index])
                output["row_size"] = torch.tensor(self.row_size[index])
                output["col_min"] = torch.tensor(self.col_min[index])
                output["col_size"] = torch.tensor(self.col_size[index])

            return output

        elif self.phase == "test":
            return {
                "idx": torch.tensor([self.ids[index]]),
                "image": img,
            }
        else:
            raise ValueError


class ContrailDatasetV1(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True) -> np.ndarray:
        band_list = self.dataset_config.params.get("band_list", list(range(8, 16)))
        band_paths = [f"{image_dir}/{record_id}/band_{str(band).zfill(2)}.npy" for band in band_list]

        bands = [self._read_image_cached(path) for path in band_paths]

        if get_mask_frame_only:
            bands = [band[:, :, 4] for band in bands]

        bands = np.stack(bands, axis=2)

        return bands


class ContrailDatasetV2(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True) -> np.ndarray:
        band11_path = f"{image_dir}/{record_id}/band_11.npy"
        band14_path = f"{image_dir}/{record_id}/band_14.npy"
        band15_path = f"{image_dir}/{record_id}/band_15.npy"

        band11 = self._read_image_cached(band11_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        if get_mask_frame_only:
            band11 = band11[:, :, 4]
            band14 = band14[:, :, 4]
            band15 = band15[:, :, 4]

        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)
        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

        band08_path = f"{image_dir}/{record_id}/band_08.npy"
        band09_path = f"{image_dir}/{record_id}/band_09.npy"
        band10_path = f"{image_dir}/{record_id}/band_10.npy"

        band08 = self._read_image_cached(band08_path)
        band09 = self._read_image_cached(band09_path)
        band10 = self._read_image_cached(band10_path)

        if get_mask_frame_only:
            band08 = band08[:, :, 4]
            band09 = band09[:, :, 4]
            band10 = band10[:, :, 4]

        r = normalize_range(band10 - band09, _TDIFF_BOUNDS)
        g = normalize_range(band09 - band08, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band08, _T11_BOUNDS)
        false_color2 = np.clip(np.stack([r, g, b], axis=2), 0, 1)

        false_color = np.concatenate([false_color, false_color2], axis=2)

        return false_color


class ContrailDatasetV3(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        band11_path = f"{image_dir}/{record_id}/band_11.npy"
        band14_path = f"{image_dir}/{record_id}/band_14.npy"
        band15_path = f"{image_dir}/{record_id}/band_15.npy"

        band11 = self._read_image_cached(band11_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        if get_mask_frame_only:
            band11 = band11[:, :, frame]
            band14 = band14[:, :, frame]
            band15 = band15[:, :, frame]

        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)
        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

        band08_path = f"{image_dir}/{record_id}/band_08.npy"
        band09_path = f"{image_dir}/{record_id}/band_09.npy"
        band10_path = f"{image_dir}/{record_id}/band_10.npy"

        band08 = self._read_image_cached(band08_path)
        band09 = self._read_image_cached(band09_path)
        band10 = self._read_image_cached(band10_path)

        if get_mask_frame_only:
            band08 = band08[:, :, frame]
            band09 = band09[:, :, frame]
            band10 = band10[:, :, frame]

        r = normalize_range(band10 - band09, _TDIFF_BOUNDS)
        g = normalize_range(band09 - band08, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band08, _T11_BOUNDS)
        false_color2 = np.clip(np.stack([r, g, b], axis=2), 0, 1)

        band1_path = f"{image_dir}/{record_id}/band_12.npy"
        band2_path = f"{image_dir}/{record_id}/band_13.npy"
        band3_path = f"{image_dir}/{record_id}/band_16.npy"

        band1 = self._read_image_cached(band1_path)
        band2 = self._read_image_cached(band2_path)
        band3 = self._read_image_cached(band3_path)

        if get_mask_frame_only:
            band1 = band1[:, :, frame]
            band2 = band2[:, :, frame]
            band3 = band3[:, :, frame]
        r = normalize_range(band3 - band2, _TDIFF_BOUNDS)
        g = normalize_range(band2 - band1, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band1, _T11_BOUNDS)
        false_color3 = np.clip(np.stack([r, g, b], axis=2), 0, 1)

        false_color = np.concatenate([false_color, false_color2, false_color3], axis=2)

        return false_color


class ContrailDatasetV4(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{image_dir}/{record_id}/band_11.npy"  # 8.4-μm
        band13_path = f"{image_dir}/{record_id}/band_13.npy"  # 10.3-μm
        band14_path = f"{image_dir}/{record_id}/band_14.npy"  # 11.2-μm
        band15_path = f"{image_dir}/{record_id}/band_15.npy"  # 12.3-μm

        band11 = self._read_image_cached(band11_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        if get_mask_frame_only:
            band11 = band11[:, :, frame]
            band13 = band13[:, :, frame]
            band14 = band14[:, :, frame]
            band15 = band15[:, :, frame]
        else:
            band11 = band11[:, :, self.start_frame_idx : self.end_frame_idx + 1]
            band13 = band13[:, :, self.start_frame_idx : self.end_frame_idx + 1]
            band14 = band14[:, :, self.start_frame_idx : self.end_frame_idx + 1]
            band15 = band15[:, :, self.start_frame_idx : self.end_frame_idx + 1]

        if self.normalize_method == "mean_std":
            band11 = normalize_mean_std(band11, BAND11_MEAN_STD[0], BAND11_MEAN_STD[1])
            band13 = normalize_mean_std(band13, BAND13_MEAN_STD[0], BAND13_MEAN_STD[1])
            band14 = normalize_mean_std(band14, BAND14_MEAN_STD[0], BAND14_MEAN_STD[1])
            band15 = normalize_mean_std(band15, BAND15_MEAN_STD[0], BAND15_MEAN_STD[1])

            r = band15 - band14
            g = band14 - band11
            b = band14

            false_color = np.stack([r, g, b], axis=2)

            bands = np.stack([band11, band13, band15], axis=2)

            false_color = np.concatenate([false_color, bands], axis=2)  # h x w x channel x frame

        elif self.normalize_method == "range":
            r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
            g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
            b = normalize_range(band14, _T11_BOUNDS)

            false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

            band11 = normalize_range(band11, _T11_BOUNDS)
            band13 = normalize_range(band13, _T11_BOUNDS)
            band15 = normalize_range(band15, _T11_BOUNDS)

            bands = np.clip(np.stack([band11, band13, band15], axis=2), 0, 1)

            false_color = np.concatenate([false_color, bands], axis=2)
        else:
            ValueError("Unknown `normalize_method`")

        return false_color


class ContrailDatasetV5(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{image_dir}/{record_id}/band_11.npy"  # 8.4-μm
        band13_path = f"{image_dir}/{record_id}/band_13.npy"  # 10.3-μm
        band14_path = f"{image_dir}/{record_id}/band_14.npy"  # 11.2-μm
        band15_path = f"{image_dir}/{record_id}/band_15.npy"  # 12.3-μm

        band11 = self._read_image_cached(band11_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        if get_mask_frame_only:
            band11 = band11[:, :, 4]
            band13 = band13[:, :, 4]
            band14 = band14[:, :, 4]
            band15 = band15[:, :, 4]
        else:
            band11 = band11[:, :, 1:5]
            band13 = band13[:, :, 1:5]
            band14 = band14[:, :, 1:5]
            band15 = band15[:, :, 1:5]

        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)

        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

        band11 = normalize_range(band11, _T11_BOUNDS)
        band13 = normalize_range(band13, _T11_BOUNDS)
        band15 = normalize_range(band15, _T11_BOUNDS)

        bands = np.clip(np.stack([band11, band13, band15], axis=2), 0, 1)

        false_color = np.concatenate([false_color, bands], axis=2)

        return false_color


class ContrailDatasetV6(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{image_dir}/{record_id}/band_11.npy"  # 8.4-μm
        band13_path = f"{image_dir}/{record_id}/band_13.npy"  # 10.3-μm
        band14_path = f"{image_dir}/{record_id}/band_14.npy"  # 11.2-μm
        band15_path = f"{image_dir}/{record_id}/band_15.npy"  # 12.3-μm

        band11 = self._read_image_cached(band11_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        band11 = band11[:, :, 3:6]
        band13 = band13[:, :, 3:6]
        band14 = band14[:, :, 3:6]
        band15 = band15[:, :, 3:6]

        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)

        # get diff
        r[..., 0] = r[..., 0] - r[..., 1]
        r[..., 2] = r[..., 2] - r[..., 1]

        g[..., 0] = g[..., 0] - g[..., 1]
        g[..., 2] = g[..., 2] - g[..., 1]

        b[..., 0] = b[..., 0] - b[..., 1]
        b[..., 2] = b[..., 2] - b[..., 1]

        false_color = np.concatenate([r, g, b], axis=2)

        return false_color


class ContrailDatasetV7(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{image_dir}/{record_id}/band_11.npy"  # 8.4-μm
        band13_path = f"{image_dir}/{record_id}/band_13.npy"  # 10.3-μm
        band14_path = f"{image_dir}/{record_id}/band_14.npy"  # 11.2-μm
        band15_path = f"{image_dir}/{record_id}/band_15.npy"  # 12.3-μm

        band11 = self._read_image_cached(band11_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        band11 = band11[:, :, 3:6]
        band13 = band13[:, :, 3:6]
        band14 = band14[:, :, 3:6]
        band15 = band15[:, :, 3:6]

        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)
        band13 /= 255

        # get diff
        r[..., 0] = r[..., 0] - r[..., 1]
        r[..., 2] = r[..., 2] - r[..., 1]

        g[..., 0] = g[..., 0] - g[..., 1]
        g[..., 2] = g[..., 2] - g[..., 1]

        b[..., 0] = b[..., 0] - b[..., 1]
        b[..., 2] = b[..., 2] - b[..., 1]

        band13[..., 0] = band13[..., 0] - band13[..., 1]
        band13[..., 2] = band13[..., 2] - band13[..., 1]

        false_color = np.concatenate([r, g, b, band13], axis=2)

        return false_color


class ContrailDatasetV8(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{image_dir}/{record_id}/band_11.npy"  # 8.4-μm
        band13_path = f"{image_dir}/{record_id}/band_13.npy"  # 10.3-μm
        band14_path = f"{image_dir}/{record_id}/band_14.npy"  # 11.2-μm
        band15_path = f"{image_dir}/{record_id}/band_15.npy"  # 12.3-μm

        band11 = self._read_image_cached(band11_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        if get_mask_frame_only:
            band11 = band11[:, :, frame]
            band13 = band13[:, :, frame]
            band14 = band14[:, :, frame]
            band15 = band15[:, :, frame]

        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)

        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

        band11 = normalize_range(band11, _T11_BOUNDS)
        band13 = normalize_range(band13, _T11_BOUNDS)
        band15 = normalize_range(band15, _T11_BOUNDS)

        bands = np.clip(np.stack([band11, band13, band15], axis=2), 0, 1)

        false_color = np.concatenate([false_color, bands], axis=2)

        return false_color


class ContrailDatasetV9(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{image_dir}/{record_id}/band_11.npy"  # 8.4-μm
        band12_path = f"{image_dir}/{record_id}/band_12.npy"  # 9.6-μm
        band13_path = f"{image_dir}/{record_id}/band_13.npy"  # 10.3-μm
        band14_path = f"{image_dir}/{record_id}/band_14.npy"  # 11.2-μm
        band15_path = f"{image_dir}/{record_id}/band_15.npy"  # 12.3-μm
        band16_path = f"{image_dir}/{record_id}/band_16.npy"  # 13.3-μm

        band11 = self._read_image_cached(band11_path)
        band12 = self._read_image_cached(band12_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)
        band16 = self._read_image_cached(band16_path)

        if get_mask_frame_only:
            band11 = band11[:, :, frame]
            band12 = band12[:, :, frame]
            band13 = band13[:, :, frame]
            band14 = band14[:, :, frame]
            band15 = band15[:, :, frame]
            band16 = band16[:, :, frame]

        if self.normalize_method == "mean_std":
            band11 = normalize_mean_std(band11, BAND11_MEAN_STD[0], BAND11_MEAN_STD[1])
            band12 = normalize_mean_std(band12, BAND12_MEAN_STD[0], BAND12_MEAN_STD[1])
            band13 = normalize_mean_std(band13, BAND13_MEAN_STD[0], BAND13_MEAN_STD[1])
            band14 = normalize_mean_std(band14, BAND14_MEAN_STD[0], BAND14_MEAN_STD[1])
            band15 = normalize_mean_std(band15, BAND15_MEAN_STD[0], BAND15_MEAN_STD[1])
            band16 = normalize_mean_std(band16, BAND16_MEAN_STD[0], BAND16_MEAN_STD[1])

            r = band15 - band14
            g = band14 - band11
            b = band14

            false_color = np.stack([r, g, b], axis=2)

            bands = np.stack([band11, band13, band15, band12, band16], axis=2)

            false_color = np.concatenate([false_color, bands], axis=2)

        elif self.normalize_method == "range":
            r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
            g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
            b = normalize_range(band14, _T11_BOUNDS)

            false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

            band11 = normalize_range(band11, _T11_BOUNDS)
            band13 = normalize_range(band13, _T11_BOUNDS)
            band15 = normalize_range(band15, _T11_BOUNDS)

            bands = np.clip(np.stack([band11, band13, band15], axis=2), 0, 1)

            false_color = np.concatenate([false_color, bands], axis=2)
        else:
            ValueError("Unknown `normalize_method`")

        return false_color


class ContrailDataset3DV1(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{image_dir}/{record_id}/band_11.npy"  # 8.4-μm
        band13_path = f"{image_dir}/{record_id}/band_13.npy"  # 10.3-μm
        band14_path = f"{image_dir}/{record_id}/band_14.npy"  # 11.2-μm
        band15_path = f"{image_dir}/{record_id}/band_15.npy"  # 12.3-μm

        band11 = self._read_image_cached(band11_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        if get_mask_frame_only:
            band11 = band11[:, :, frame]
            band13 = band13[:, :, frame]
            band14 = band14[:, :, frame]
            band15 = band15[:, :, frame]
        else:
            band11 = band11[:, :, 1:5]
            band13 = band13[:, :, 1:5]
            band14 = band14[:, :, 1:5]
            band15 = band15[:, :, 1:5]

        if self.normalize_method == "mean_std":
            band11 = normalize_mean_std(band11, BAND11_MEAN_STD[0], BAND11_MEAN_STD[1])
            band13 = normalize_mean_std(band13, BAND13_MEAN_STD[0], BAND13_MEAN_STD[1])
            band14 = normalize_mean_std(band14, BAND14_MEAN_STD[0], BAND14_MEAN_STD[1])
            band15 = normalize_mean_std(band15, BAND15_MEAN_STD[0], BAND15_MEAN_STD[1])

            r = band15 - band14
            g = band14 - band11
            b = band14
            false_color = np.stack([r, g, b], axis=2)

        elif self.normalize_method == "range":
            r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
            g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
            b = normalize_range(band14, _T11_BOUNDS)
            false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

        else:
            ValueError("Unknown `normalize_method`")

        return false_color


class ContrailDataset3DV2(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{image_dir}/{record_id}/band_11.npy"  # 8.4-μm
        band13_path = f"{image_dir}/{record_id}/band_13.npy"  # 10.3-μm
        band14_path = f"{image_dir}/{record_id}/band_14.npy"  # 11.2-μm
        band15_path = f"{image_dir}/{record_id}/band_15.npy"  # 12.3-μm

        band11 = self._read_image_cached(band11_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        if get_mask_frame_only:
            band11 = band11[:, :, frame]
            band13 = band13[:, :, frame]
            band14 = band14[:, :, frame]
            band15 = band15[:, :, frame]
        else:
            band11 = band11[:, :, 1:5]
            band13 = band13[:, :, 1:5]
            band14 = band14[:, :, 1:5]
            band15 = band15[:, :, 1:5]

        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = normalize_range(band14, _T11_BOUNDS)

        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

        band11 = normalize_range(band11, _T11_BOUNDS)
        band13 = normalize_range(band13, _T11_BOUNDS)
        band15 = normalize_range(band15, _T11_BOUNDS)

        bands = np.clip(np.stack([band11, band13, band15], axis=2), 0, 1)

        false_color = np.concatenate([false_color, bands], axis=2)

        return false_color


class ContrailDatasetV10(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{image_dir}/{record_id}/band_11.npy"  # 8.4-μm
        band13_path = f"{image_dir}/{record_id}/band_13.npy"  # 10.3-μm
        band14_path = f"{image_dir}/{record_id}/band_14.npy"  # 11.2-μm
        band15_path = f"{image_dir}/{record_id}/band_15.npy"  # 12.3-μm

        band11 = self._read_image_cached(band11_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        if get_mask_frame_only:
            band11 = band11[:, :, frame]
            band13 = band13[:, :, frame]
            band14 = band14[:, :, frame]
            band15 = band15[:, :, frame]

        if self.normalize_method == "mean_std":
            band11 = normalize_mean_std(band11, BAND11_MEAN_STD[0], BAND11_MEAN_STD[1])
            band13 = normalize_mean_std(band13, BAND13_MEAN_STD[0], BAND13_MEAN_STD[1])
            band14 = normalize_mean_std(band14, BAND14_MEAN_STD[0], BAND14_MEAN_STD[1])
            band15 = normalize_mean_std(band15, BAND15_MEAN_STD[0], BAND15_MEAN_STD[1])

            r = band15 - band14
            g = band14 - band11
            b = band14

            false_color = np.stack([r, g, b, band13], axis=2)

        elif self.normalize_method == "range":
            r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
            g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
            b = normalize_range(band14, _T11_BOUNDS)

            band11 = normalize_range(band11, _T11_BOUNDS)
            band13 = normalize_range(band13, _T11_BOUNDS)
            band15 = normalize_range(band15, _T11_BOUNDS)

            false_color = np.clip(np.stack([r, g, b, band13], axis=2), 0, 1)

        else:
            ValueError("Unknown `normalize_method`")

        return false_color


class ContrailDatasetV11(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{image_dir}/{record_id}/band_11.npy"  # 8.4-μm
        band13_path = f"{image_dir}/{record_id}/band_13.npy"  # 10.3-μm
        band14_path = f"{image_dir}/{record_id}/band_14.npy"  # 11.2-μm
        band15_path = f"{image_dir}/{record_id}/band_15.npy"  # 12.3-μm

        band11 = self._read_image_cached(band11_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        if get_mask_frame_only:
            band11 = band11[:, :, frame]
            band13 = band13[:, :, frame]
            band14 = band14[:, :, frame]
            band15 = band15[:, :, frame]

        if self.normalize_method == "mean_std":
            band11 = normalize_mean_std(band11, BAND11_MEAN_STD[0], BAND11_MEAN_STD[1])
            band13 = normalize_mean_std(band13, BAND13_MEAN_STD[0], BAND13_MEAN_STD[1])
            band14 = normalize_mean_std(band14, BAND14_MEAN_STD[0], BAND14_MEAN_STD[1])
            band15 = normalize_mean_std(band15, BAND15_MEAN_STD[0], BAND15_MEAN_STD[1])

            false_color = np.stack([band11, band13, band14, band15], axis=2)

        elif self.normalize_method == "range":
            r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
            g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
            b = normalize_range(band14, _T11_BOUNDS)

            band11 = normalize_range(band11, _T11_BOUNDS)
            band13 = normalize_range(band13, _T11_BOUNDS)
            band15 = normalize_range(band15, _T11_BOUNDS)

            false_color = np.clip(np.stack([r, g, b, band13], axis=2), 0, 1)

        else:
            ValueError("Unknown `normalize_method`")

        return false_color


class ContrailDatasetV12(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band_list = self.dataset_config.params["band_list"]
        band_data = {}
        band_mean_std = {
            "band_08": BAND08_MEAN_STD,
            "band_09": BAND09_MEAN_STD,
            "band_10": BAND10_MEAN_STD,
            "band_11": BAND11_MEAN_STD,
            "band_12": BAND12_MEAN_STD,
            "band_13": BAND13_MEAN_STD,
            "band_14": BAND14_MEAN_STD,
            "band_15": BAND15_MEAN_STD,
            "band_16": BAND16_MEAN_STD,
        }

        for band in band_list:
            band_path = f"{image_dir}/{record_id}/{band}.npy"
            band_data[band] = self._read_image_cached(band_path)

        if get_mask_frame_only:
            for band in band_list:
                band_data[band] = band_data[band][:, :, frame]

        if self.normalize_method == "mean_std":
            for band in band_list:
                band_data[band] = normalize_mean_std(band_data[band], *band_mean_std[band])

            r = band_data["band_13"] - band_data["band_12"]
            g = band_data["band_12"] - band_data["band_10"]
            b = band_data["band_12"]

            false_color = np.stack([r, g, b], axis=2)

            bands = np.stack([band_data["band_10"], band_data["band_13"], band_data["band_09"]], axis=2)

            false_color = np.concatenate([false_color, bands], axis=2)

        elif self.normalize_method == "range":
            r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
            g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
            b = normalize_range(band14, _T11_BOUNDS)

            false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

            band11 = normalize_range(band11, _T11_BOUNDS)
            band13 = normalize_range(band13, _T11_BOUNDS)
            band15 = normalize_range(band15, _T11_BOUNDS)

            bands = np.clip(np.stack([band11, band13, band15], axis=2), 0, 1)

            false_color = np.concatenate([false_color, bands], axis=2)
        else:
            ValueError("Unknown `normalize_method`")

        return false_color


class ContrailDatasetV13(ContrailDatasetBase):
    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band_list = self.dataset_config.params["band_list"]
        band_data = {}
        band_mean_std = {
            "band_08": BAND08_MEAN_STD,
            "band_09": BAND09_MEAN_STD,
            "band_10": BAND10_MEAN_STD,
            "band_11": BAND11_MEAN_STD,
            "band_12": BAND12_MEAN_STD,
            "band_13": BAND13_MEAN_STD,
            "band_14": BAND14_MEAN_STD,
            "band_15": BAND15_MEAN_STD,
            "band_16": BAND16_MEAN_STD,
        }

        for band in band_list:
            band_path = f"{image_dir}/{record_id}/{band}.npy"
            band_data[band] = self._read_image_cached(band_path)

        if get_mask_frame_only:
            for band in band_list:
                band_data[band] = band_data[band][:, :, frame]

        if self.normalize_method == "mean_std":
            for band in band_list:
                band_data[band] = normalize_mean_std(band_data[band], *band_mean_std[band])

            r = band_data["band_15"] - band_data["band_14"]
            g = band_data["band_14"] - band_data["band_11"]
            b = band_data["band_14"]

            main_false_color = np.stack(
                [
                    r,
                    g,
                    b,
                    band_data["band_11"],
                    band_data["band_13"],
                    band_data["band_15"],
                ],
                axis=2,
            )

            r = band_data["band_13"] - band_data["band_12"]
            g = band_data["band_12"] - band_data["band_10"]
            b = band_data["band_12"]

            false_color = np.stack(
                [
                    r,
                    g,
                    b,
                    band_data["band_10"],
                    band_data["band_13"],
                    band_data["band_09"],
                ],
                axis=2,
            )

            false_color = np.concatenate([main_false_color, false_color], axis=2)

        elif self.normalize_method == "range":
            r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
            g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
            b = normalize_range(band14, _T11_BOUNDS)

            false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

            band11 = normalize_range(band11, _T11_BOUNDS)
            band13 = normalize_range(band13, _T11_BOUNDS)
            band15 = normalize_range(band15, _T11_BOUNDS)

            bands = np.clip(np.stack([band11, band13, band15], axis=2), 0, 1)

            false_color = np.concatenate([false_color, bands], axis=2)
        else:
            ValueError("Unknown `normalize_method`")

        return false_color


class ContrailDatasetPLV1(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        training_config: TrainingConfig,
        phase: str,
        transform: A.Compose = None,
    ) -> None:
        self.df = df
        self.ids = self.df.index.values
        if "frame" not in self.df.columns:
            self.df["frame"] = 4

        if "row_min" in self.df.columns:
            self.row_min = self.df["row_min"].values / 10135611.5
            self.row_size = self.df["row_size"].values / 230.16455078125
            self.col_min = self.df["col_min"].values / 663554.46875
            self.col_size = self.df["col_size"].values / 298.611083984375
        else:
            self.row_min = None
            self.row_size = None
            self.col_min = None
            self.col_size = None

        self.frames = self.df["frame"].values
        self.record_ids = self.df.record_id.values
        self.from_train = self.df.from_train.values
        self.pred_mask = None

        self.config = training_config
        self.dataset_config = training_config.dataset_config
        self.normalize_method = self.dataset_config.params.get("normalize_method", "range")
        self.get_mask_frame_only = self.dataset_config.params.get("get_mask_frame_only", False)
        self.use_individual_mask = self.dataset_config.params.get("use_individual_mask", False)
        self.use_max_min = self.dataset_config.params.get("use_max_min", False)
        self.start_frame_idx = self.dataset_config.params.get("start_frame_idx", 0)
        self.end_frame_idx = self.dataset_config.params.get("end_frame_idx", 8)
        self.add_pseudo_label = self.dataset_config.params.get("add_pseudo_label", False)
        self.round_soft_label = self.dataset_config.params.get("round_soft_label", False)
        self.phase = phase

        self.transform = transform

        if self.phase == "train":
            self.image_dir_paths = self.df.image_dir.values
            self.org_image_dir = self.config.train_image_dir
            cache_dir = f"/tmp/p20-contrail/{self.phase}"
            gc.collect()
        elif self.phase == "valid":
            self.image_dir_paths = self.df.image_dir.values
            self.org_image_dir = self.config.valid_image_dir
            cache_dir = f"/tmp/p20-contrail/{self.phase}"
            gc.collect()

        elif self.phase == "test":
            self.image_dir_paths = self.df.image_dir.values
            self.org_image_dir = self.config.test_image_dir
            self.config.scs_dataset_dir = None
            cache_dir = None
            gc.collect()

        if cache_dir:
            self._cache = MultiprocessFileCache(len(self.df), dir=cache_dir, do_pickle=True)
        else:
            self._cache = None

        if self.config.scs_dataset_dir is None:
            self._scs_cache = None
            self._do_put = False
        else:
            debug_suffix = "_debug" if self.config.debug else ""
            self._do_put = True

    def _read_image_cached(self, file_path: str):
        img = self._read_from_storage(file_path)

        return img

    def _read_from_storage(self, file_path: str):
        img = np.load(file_path)
        return img

    def get_ash_color_images(self, record_id: str, image_dir: str, get_mask_frame_only=True, frame=4) -> np.ndarray:
        # https://arxiv.org/pdf/2304.02122.pdf
        # 8µm, 10µm, 11µm, 12µm, difference between 12µm and 11µm, and difference between 11µm and 8µm
        band11_path = f"{image_dir}/{record_id}/band_11.npy"  # 8.4-μm
        band13_path = f"{image_dir}/{record_id}/band_13.npy"  # 10.3-μm
        band14_path = f"{image_dir}/{record_id}/band_14.npy"  # 11.2-μm
        band15_path = f"{image_dir}/{record_id}/band_15.npy"  # 12.3-μm

        band11 = self._read_image_cached(band11_path)
        band13 = self._read_image_cached(band13_path)
        band14 = self._read_image_cached(band14_path)
        band15 = self._read_image_cached(band15_path)

        if get_mask_frame_only:
            band11 = band11[:, :, frame]
            band13 = band13[:, :, frame]
            band14 = band14[:, :, frame]
            band15 = band15[:, :, frame]
        else:
            band11 = band11[:, :, self.start_frame_idx : self.end_frame_idx + 1]
            band13 = band13[:, :, self.start_frame_idx : self.end_frame_idx + 1]
            band14 = band14[:, :, self.start_frame_idx : self.end_frame_idx + 1]
            band15 = band15[:, :, self.start_frame_idx : self.end_frame_idx + 1]

        if self.normalize_method == "mean_std":
            band11 = normalize_mean_std(band11, BAND11_MEAN_STD[0], BAND11_MEAN_STD[1])
            band13 = normalize_mean_std(band13, BAND13_MEAN_STD[0], BAND13_MEAN_STD[1])
            band14 = normalize_mean_std(band14, BAND14_MEAN_STD[0], BAND14_MEAN_STD[1])
            band15 = normalize_mean_std(band15, BAND15_MEAN_STD[0], BAND15_MEAN_STD[1])

            r = band15 - band14
            g = band14 - band11
            b = band14

            false_color = np.stack([r, g, b], axis=2)

            bands = np.stack([band11, band13, band15], axis=2)

            false_color = np.concatenate([false_color, bands], axis=2)  # h x w x channel x frame

        elif self.normalize_method == "range":
            r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
            g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
            b = normalize_range(band14, _T11_BOUNDS)

            false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

            band11 = normalize_range(band11, _T11_BOUNDS)
            band13 = normalize_range(band13, _T11_BOUNDS)
            band15 = normalize_range(band15, _T11_BOUNDS)

            bands = np.clip(np.stack([band11, band13, band15], axis=2), 0, 1)

            false_color = np.concatenate([false_color, bands], axis=2)
        else:
            ValueError("Unknown `normalize_method`")

        return false_color

    def get_target_mask(
        self,
        human_pixel_path: str,
        use_individual_mask: bool = False,
        use_max_min: bool = False,
        from_train: bool = False,
    ) -> np.ndarray:
        target = self._read_image_cached(human_pixel_path)
        if len(target.shape) == 2:
            target = target[..., np.newaxis]

        if use_individual_mask:
            if from_train:
                thresh = 0.6
                hard_target = (target > thresh).astype(np.uint8)
                if self.round_soft_label:
                    target = np.round(target * 4.0) / 4.0
                return (hard_target, target.astype(np.float32))
            else:
                return target
        else:
            if self.phase == "train":
                thresh = 0.6
                hard_target = (target > thresh).astype(np.uint8)
                return hard_target
            else:
                return target

    def get_position_image(self, row_min: float, row_size: float, col_size: float, col_min: float):
        size = self.config.transform_config.train_transform["transform_config"]["params"]["Resize"]["width"]
        rm = torch.ones((1, size, size)) * row_min
        rs = torch.ones((1, size, size)) * row_size
        cm = torch.ones((1, size, size)) * col_min
        cs = torch.ones((1, size, size)) * col_size

        pos_img = torch.cat([rm, rs, cm, cs], axis=0)

        return pos_img

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        record_id = self.record_ids[index]
        frame = self.frames[index]
        image_dir = self.image_dir_paths[index]
        from_train = self.from_train[index]
        img = self.get_ash_color_images(
            record_id, self.org_image_dir, get_mask_frame_only=self.get_mask_frame_only, frame=frame
        )

        if not self.get_mask_frame_only:
            img = img.transpose(0, 1, 3, 2).reshape(256, 256, -1)

        if self.transform:
            t_image = {"image": img}

            if self.phase in ["train"]:
                human_pixel_path = f"{image_dir}/{record_id}/frame{frame}.npy"
                target = self.get_target_mask(
                    human_pixel_path, self.use_individual_mask, use_max_min=self.use_max_min, from_train=from_train
                )

                if self.use_individual_mask:
                    t_image["mask"] = target[0]
                    t_image["mask1"] = target[1]
                    if self.use_max_min:
                        t_image["mask2"] = target[2]
                        t_image["mask3"] = target[3]
                else:
                    t_image["mask"] = target

                if self.add_pseudo_label:
                    for k, frame in enumerate(range(self.start_frame_idx, self.end_frame_idx + 1)):
                        pseudo_human_pixel_path = f"{self.config.pseudo_label_dir}/{record_id}/frame{frame}.npy"
                        pseudo_label = self.get_target_mask(
                            pseudo_human_pixel_path,
                            use_individual_mask=False,
                            use_max_min=self.use_max_min,
                            from_train=from_train,
                        )
                        t_image[f"mask{2+k}"] = pseudo_label

            t_image = self.transform(**t_image)

            img = t_image["image"]

        if self.phase in ["train", "valid"]:
            if self.phase == "train":
                target = t_image["mask"].permute(2, 0, 1)
            else:
                human_pixel_path = f"{image_dir}/{record_id}/human_pixel_masks.npy"
                target = self.get_target_mask(human_pixel_path, False)
                target = torch.tensor(target.transpose(2, 0, 1), dtype=torch.float32)

            output = {
                "idx": torch.tensor([self.ids[index]]),
                "image": img,
                "target": target,
                "from_train": from_train,
            }
            if self.phase == "train":
                if self.use_individual_mask:
                    output["individual_mask"] = t_image["mask1"].permute(2, 0, 1)
                    if self.use_max_min:
                        output["individual_mask_max"] = t_image["mask2"].permute(2, 0, 1)
                        output["individual_mask_min"] = t_image["mask3"].permute(2, 0, 1)

                if self.add_pseudo_label:
                    for k, frame in enumerate(range(self.start_frame_idx, self.end_frame_idx + 1)):
                        output[f"pseudo_mask{k}"] = t_image[f"mask{2+k}"].permute(2, 0, 1)

            if self.row_min is not None:
                output["row_min"] = torch.tensor(self.row_min[index])
                output["row_size"] = torch.tensor(self.row_size[index])
                output["col_min"] = torch.tensor(self.col_min[index])
                output["col_size"] = torch.tensor(self.col_size[index])

            return output

        elif self.phase == "test":
            return {
                "idx": torch.tensor([self.ids[index]]),
                "image": img,
            }
        else:
            raise ValueError
