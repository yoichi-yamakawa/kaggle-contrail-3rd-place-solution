import glob
import os
import random
import re
import shlex
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from laputamon.io_util.directory_in_zip import DirectoryInZip

_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)


def get_band_images(idx: str, image_dir: str, band: str) -> np.array:
    return np.load(f"{image_dir}/{idx}/band_{band}.npy")


def get_mask_images(idx: str, image_dir: str) -> np.array:
    return np.load(f"{image_dir}/{idx}/human_pixel_masks.npy")


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_ash_color_images(idx: str, image_dir: str, get_mask_frame_only=True) -> np.array:
    band11 = get_band_images(idx, image_dir, "11")
    band14 = get_band_images(idx, image_dir, "14")
    band15 = get_band_images(idx, image_dir, "15")

    if get_mask_frame_only:
        band11 = band11[:, :, 4]
        band14 = band14[:, :, 4]
        band15 = band15[:, :, 4]

    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return false_color


def slim_train_v1(df: pd.DataFrame, use_idx_mod100=24):
    df = df.loc[df.index % 100 <= use_idx_mod100].reset_index(drop=True)

    return df


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


# pytorch lightning helper
def get_best_ckpt_path(dirpath) -> str:
    callbacks = torch.load(dirpath + "/last.ckpt")["callbacks"]
    best_model_path = None
    for k, v in callbacks.items():
        if isinstance(v, dict):
            if "best_model_path" in v.keys():
                best_model_path = v["best_model_path"]

    return best_model_path


def make_gathered_result_file(outdir, n_process, filename):
    all_result = {}

    for i in range(n_process):
        output = pd.read_pickle(f"{outdir}/{filename}_{i}.pkl")
        print(f"{outdir}/{filename}_{i}.pkl")

        for k, v in output.items():
            if k != "target":
                all_result.setdefault(k, [])
                all_result[k].append(v)

    for k, v in all_result.items():
        all_result[k] = np.concatenate(v)

    sorted_idx = np.argsort(all_result["idx"].reshape(-1))

    new_all_result = {}
    for k, v in all_result.items():
        new_all_result[k] = all_result[k][sorted_idx]

    pd.to_pickle(new_all_result, f"{outdir}/{filename}.pkl")


def get_gpu_info():
    """
    Returns size of total GPU RAM and used GPU RAM.

    Parameters
    ----------
    None

    Returns
    -------
    info : dict
        Total GPU RAM in integer for key 'total_MiB'.
        Used GPU RAM in integer for key 'used_MiB'.
    """

    command = 'nvidia-smi -q -d MEMORY | sed -n "/FB Memory Usage/,/Free/p" | sed -e "1d" -e "4d" -e "s/ MiB//g" | cut -d ":" -f 2 | cut -c2-'
    commands = [shlex.split(part) for part in command.split(" | ")]
    for i, cmd in enumerate(commands):
        if i == 0:
            res = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        else:
            res = subprocess.Popen(cmd, stdin=res.stdout, stdout=subprocess.PIPE)
    total, used = map(int, res.communicate()[0].decode("utf-8").strip().split("\n"))
    info = {"total_MiB": total, "used_MiB": used}
    return info
