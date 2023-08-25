import math

import albumentations as A
import cv2
from albumentations import (
    Blur,
    ChannelShuffle,
    CoarseDropout,
    ColorJitter,
    Compose,
    Cutout,
    GaussianBlur,
    GaussNoise,
    HorizontalFlip,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    MotionBlur,
    Normalize,
    OneOf,
    Perspective,
    RandomBrightness,
    RandomBrightnessContrast,
    RandomContrast,
    RandomCrop,
    RandomResizedCrop,
    RandomRotate90,
    Resize,
    Rotate,
    ShiftScaleRotate,
    ToGray,
    Transpose,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2


def BaseTransformV1(params):
    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            ToTensorV2(),
        ],
    )


def BaseTransformV1Resize(params):
    if "additional_targets" in params.keys():
        additional_targets = {
            "mask1": "mask",
        }
    else:
        additional_targets = None

    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            Resize(**params["Resize"]),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def BaseTransformV1Norm(params):
    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            Normalize(
                mean=[0.5] * params["Normalize"]["normalize_ch_num"],
                std=[0.225] * params["Normalize"]["normalize_ch_num"],
            ),
            ToTensorV2(),
        ],
    )


def BaseTransformV2(params):
    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            ShiftScaleRotate(**params["ShiftScaleRotate"]),
            RandomBrightnessContrast(**params["RandomBrightnessContrast"]),
            ToTensorV2(),
        ],
    )


def BaseTransformV2Norm(params):
    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            ShiftScaleRotate(**params["ShiftScaleRotate"]),
            RandomBrightnessContrast(**params["RandomBrightnessContrast"]),
            Normalize(
                mean=[0.5] * params["Normalize"]["normalize_ch_num"],
                std=[0.225] * params["Normalize"]["normalize_ch_num"],
            ),
            ToTensorV2(),
        ],
    )


def BaseTransformV3(params):
    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            ShiftScaleRotate(**params["ShiftScaleRotate"]),
            RandomBrightnessContrast(**params["RandomBrightnessContrast"]),
            Resize(**params["Resize"]),
            ToTensorV2(),
        ],
    )


def BaseTransformV4(params):
    return A.Compose(
        [
            RandomCrop(**params["RandomCrop"]),
            HorizontalFlip(**params["HorizontalFlip"]),
            ShiftScaleRotate(**params["ShiftScaleRotate"]),
            RandomBrightnessContrast(**params["RandomBrightnessContrast"]),
            A.Cutout(**params["Cutout"]),
            Normalize(
                mean=[0.5] * params["Normalize"]["normalize_ch_num"],
                std=[0.225] * params["Normalize"]["normalize_ch_num"],
            ),
            Resize(**params["Resize"]),
            ToTensorV2(),
        ],
    )


def BaseTransformV5(params):
    if "additional_targets" in params.keys():
        additional_targets = {
            "mask1": "mask",
        }
    else:
        additional_targets = None

    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            # RandomResizedCrop(**params["RandomResizedCrop"]),
            # Resize(**params["Resize"]),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def BaseTransformV6(params):
    if "additional_targets" in params.keys():
        additional_targets = {
            "mask1": "mask",
        }
    else:
        additional_targets = None

    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            RandomBrightnessContrast(**params["RandomBrightnessContrast"]),
            Resize(**params["Resize"]),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def BaseTransformV7(params):
    if "additional_targets" in params.keys():
        n_additional = params["additional_targets"]
        additional_targets = {f"mask{i}": "mask" for i in range(1, n_additional + 1)}
    else:
        additional_targets = None

    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            RandomResizedCrop(**params["RandomResizedCrop"]),
            Resize(**params["Resize"]),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def BaseTransformV8(params):
    if "additional_targets" in params.keys():
        n_additional = params["additional_targets"]
        additional_targets = {f"mask{i}": "mask" for i in range(1, n_additional + 1)}
    else:
        additional_targets = None

    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            ShiftScaleRotate(**params["ShiftScaleRotate"]),
            RandomResizedCrop(**params["RandomResizedCrop"]),
            Resize(**params["Resize"]),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def BaseTransformV9(params):
    if "additional_targets" in params.keys():
        n_additional = params["additional_targets"]
        additional_targets = {f"mask{i}": "mask" for i in range(1, n_additional + 1)}
    else:
        additional_targets = None

    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            A.RandomBrightnessContrast(**params["RandomBrightnessContrast"]),
            ShiftScaleRotate(**params["ShiftScaleRotate"]),
            RandomResizedCrop(**params["RandomResizedCrop"]),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=[10, 50]),
                    A.GaussianBlur(),
                    A.MotionBlur(),
                ],
                **params["OneOf"],
            ),
            Resize(**params["Resize"]),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def BaseTransformV10(params):
    if "additional_targets" in params.keys():
        n_additional = params["additional_targets"]
        additional_targets = {f"mask{i}": "mask" for i in range(1, n_additional + 1)}
    else:
        additional_targets = None

    return A.Compose(
        [
            HorizontalFlip(**params["HorizontalFlip"]),
            VerticalFlip(**params["VerticalFlip"]),
            RandomRotate90(**params["RandomRotate90"]),
            Resize(**params["Resize"]),
            ShiftScaleRotate(**params["ShiftScaleRotate"]),
            RandomResizedCrop(**params["RandomResizedCrop"]),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )
