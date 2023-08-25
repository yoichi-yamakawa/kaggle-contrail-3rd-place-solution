from typing import Any, Dict

import numpy as np
import segmentation_models_pytorch as smp
import timm
import torch
import torchvision.transforms.functional as VF
from segmentation_models_pytorch.base.heads import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torch import nn
from torch.cuda.amp import autocast
from torch.distributions import Beta
from torch.nn import functional as F

from scripts.training import model_util
from scripts.training.model_util import Conv3dBlock, Conv3dBlockV2, GeMP, PositionalFeaturesBlockV1

# from scripts.training.resnet3d_csn import ResNet3dCSN


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


class UNetBaseV1(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.classification_params = self.model_params.get("classification_params", None)
        self.phase = phase

        self.model = smp.Unet(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            classes=1,
            aux_params=self.classification_params,
        )

        grad_checkpointing = self.model_params.get("grad_checkpointing", False)
        self.model.encoder.model.set_grad_checkpointing(grad_checkpointing)
        self.mixup_p = self.model_params.get("mixup_p", 0.0)
        self.resize = self.model_params.get("resize", None)

        self.stride = self.model_params.get("stride", None)

        if self.stride is not None:
            self.replace_first_conv(self.stride)

    def replace_first_conv(self, stride=(1, 1)):
        conv_stem_weight = self.model.encoder.model.conv1[0].weight
        self.model.encoder.model.conv1[0] = nn.Conv2d(
            self.model.encoder.model.conv1[0].in_channels,
            self.model.encoder.model.conv1[0].out_channels,
            self.model.encoder.model.conv1[0].kernel_size,
            stride=stride,
            padding=self.model.encoder.model.conv1[0].padding,
        )

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]

        if self.mixup_p > 0.0:
            if self.training:
                bs = x.shape[0]
                targets = inputs["target"]
                aux_targets = inputs["individual_mask"]

                if np.random.uniform() < self.model_params["mixup_p"]:
                    perm = torch.randperm(bs)
                    coeffs = torch.rand(bs).type_as(x).reshape(bs, 1, 1, 1)  # bs x ch x h x w
                    x = x * coeffs + (x[perm]) * (1 - coeffs)

                    output = self.model(x)

                    targets = targets * coeffs + (targets[perm]) * (1 - coeffs)
                    aux_targets = aux_targets * coeffs + (aux_targets[perm]) * (1 - coeffs)

                    return output, targets, aux_targets
                else:
                    output = self.model(x)
                    return output, targets, aux_targets

        if self.classification_params is not None:
            output, logit_meta = self.model(x)
            if self.training:
                return (output, logit_meta)
            else:
                return output

        output = self.model(x)

        if self.stride:
            output = F.interpolate(output, scale_factor=0.5, mode="nearest")
        return output


class UNetEmbMixupV1(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.phase = phase

        self.model = smp.Unet(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            classes=1,
        )
        self.mixup_p = self.model_params.get("mixup_p", 0.0)
        self.resize = self.model_params.get("resize", None)

    def mixup_features(self, x, perm, coeffs):
        self.model.check_input_shape(x)
        features = self.model.encoder(x)

        mixed_features = []

        for feat in features:
            feat = feat * coeffs + (feat[perm]) * (1 - coeffs)
            mixed_features.append(feat)

        return mixed_features

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]
        if self.resize is not None:
            x = VF.resize(img=x, size=(self.resize, self.resize))

        if self.mixup_p > 0.0:
            if self.training:
                bs = x.shape[0]
                targets = inputs["target"]
                aux_targets = inputs["individual_mask"]

                if np.random.uniform() < self.model_params["mixup_p"]:
                    perm = torch.randperm(bs)
                    coeffs = torch.rand(bs).type_as(x).reshape(bs, 1, 1, 1)  # bs x ch x h x w

                    features = self.mixup_features(x, perm, coeffs)
                    decoder_output = self.model.decoder(*features)
                    output = self.model.segmentation_head(decoder_output)

                    targets = targets * coeffs + (targets[perm]) * (1 - coeffs)
                    aux_targets = aux_targets * coeffs + (aux_targets[perm]) * (1 - coeffs)

                    return output, targets, aux_targets
                else:
                    output = self.model(x)
                    return output, targets, aux_targets

        output = self.model(x)

        if self.resize is not None:
            output = VF.resize(img=output, size=(256, 256))

        return output


class UNetStackV1(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.phase = phase

        self.model = smp.Unet(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            classes=1,
        )

        self.conv = nn.Conv2d(
            in_channels=8,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]
        bs, _, h, w = x.shape
        c, im_num = 3, 8

        x = x.view(bs, im_num, c, h, w)
        x = x.contiguous().view(-1, c, h, w)  # -> bs x im_num, c, h, w
        output = self.model(x)

        output = output.reshape(bs, im_num, h, w)
        output = self.conv(output)

        return output


class UNetBaseV2(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.phase = phase

        self.model = smp.Unet(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            classes=self.model_params["out_channel_num"],
        )

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]

        output = self.model(x)
        if self.training:
            return output

        output = output[:, 0:1, ...]

        return output


class UNetBaseV3(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.phase = phase

        self.model = smp.Unet(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            # classes=self.model_params["out_channel_num"],
            classes=1,
        )
        self.mixup_p = self.model_params.get("mixup_p", 0.0)
        self.resize = self.model_params.get("resize", None)
        self.in_channel_num = self.model_params["in_channel_num"]

        self.act_layer = nn.ReLU()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(
                num_features=16,
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]

        bs, c, h, w = x.shape
        x = x.view(-1, self.in_channel_num, h, w)  # assuming c is eve

        output = self.model(x)
        output = output.reshape(bs, -1, h, w)

        output = self.conv_block1(self.act_layer(output))
        output = self.conv2(output)

        return output


class UNetBaseV4(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.phase = phase

        self.model = smp.Unet(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            # classes=self.model_params["out_channel_num"],
            classes=1,
        )
        self.mixup_p = self.model_params.get("mixup_p", 0.0)
        self.resize = self.model_params.get("resize", None)
        self.in_channel_num = self.model_params["in_channel_num"]
        self.frame_num = self.model_params["frame_num"]

        self.conv_block1 = nn.Sequential(
            nn.BatchNorm2d(
                num_features=self.frame_num,
            ),
            nn.Conv2d(
                in_channels=self.frame_num,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.BatchNorm2d(
                num_features=64,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=self.frame_num,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.conv = nn.Conv2d(
            in_channels=self.frame_num,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]

        bs, c, h, w = x.shape
        x = x.view(-1, self.in_channel_num, h, w)  # (bs * frames) x channel x h x w

        self.model.check_input_shape(x)
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(*features)
        frames_output = self.model.segmentation_head(decoder_output)
        frames_output = frames_output.reshape(bs, -1, h, w)  # bs x out_ch (from each frame) x h x w

        residual = frames_output.clone()

        output = self.conv_block1(frames_output)
        output = self.conv_block2(output)

        output = output + residual

        output = self.conv(output)

        if self.training:
            return output, frames_output
        else:
            return output


class UNet25DV1(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.phase = phase

        self.model = smp.Unet(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            classes=self.model_params["mid_channle_num"],
        )
        self.mid_ch = self.model_params["mid_channle_num"]
        self.mixup_p = self.model_params.get("mixup_p", 0.0)
        self.resize = self.model_params.get("resize", None)
        self.in_channel_num = self.model_params["in_channel_num"]
        self.conv3d_1 = Conv3dBlockV2(self.model_params["mid_channle_num"])
        # self.conv3d_2 = Conv3dBlockV2(self.model_params["mid_channle_num"])
        self.head_2d = torch.nn.Sequential(
            torch.nn.Conv2d(
                self.model_params["mid_channle_num"],
                self.model_params["out_channel_num"],
                3,
                padding=1,
                padding_mode="replicate",
            ),
        )

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]

        bs, c, h, w = x.shape
        x = x.view(-1, self.in_channel_num, h, w)  # (bs * frames) x channel x h x w
        frames_output = self.model(x)

        # -> bs x out_ch (from each frame) x frame x h x w
        frames_output = frames_output.reshape(bs, -1, self.mid_ch, h, w).permute(0, 2, 1, 3, 4)

        output = self.conv3d_1(frames_output)
        output = output.mean(dim=2)
        output = self.head_2d(output).squeeze(2)

        return output


class UNetPlusPlusBaseV1(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.phase = phase

        self.model = smp.UnetPlusPlus(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            classes=1,
        )

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]

        output = self.model(x)
        return output


class UNet3DV1(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.phase = phase
        self.in_channel_num = self.model_params["in_channel_num"]

        self.model = smp.Unet(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            classes=1,
        )

        # for i, out_ch in enumerate(self.model.encoder.out_channels):
        #     block_name = f"conv3d_{i}"
        #     conv_d = {
        #         "out_channel_num": out_ch,
        #     }
        self.conv3d_block1 = Conv3dBlock({"out_channel_num": self.model.encoder.out_channels[0]})
        self.conv3d_block2 = Conv3dBlock({"out_channel_num": self.model.encoder.out_channels[1]})
        self.conv3d_block3 = Conv3dBlock({"out_channel_num": self.model.encoder.out_channels[2]})

        self.conv_list = [
            self.conv3d_block1,
            self.conv3d_block2,
            self.conv3d_block3,
        ]

    def conv_3d_feaures(self, features, bs):
        conv_features = []

        for i, feat in enumerate(features):
            feat = feat.reshape(bs, -1, *feat.shape[1:]).permute(0, 2, 1, 3, 4)
            # conv3d_block = getattr(self, f"conv3d_{i}")
            if i < 3:
                feat = self.conv_list[i](feat).mean(dim=2)
            else:
                feat = feat[:, :, -1, ...]

            conv_features.append(feat)
        return conv_features

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]

        bs, c, h, w = x.shape
        x = x.view(-1, self.in_channel_num, h, w)  # (bs * frames) x channel x h x w
        self.model.check_input_shape(x)
        features = self.model.encoder(x)
        features = self.conv_3d_feaures(features, bs)

        decoder_output = self.model.decoder(*features)
        output = self.model.segmentation_head(decoder_output)

        return output


# class Resnet3dCSNV1(nn.Module):
#     def __init__(self, model_params: Dict[str, Any], phase="test"):
#         super().__init__()
#         self.model_params = model_params["model_params"]
#         self.phase = phase

#         depth = self.model_params["depth"]
#         in_channels = self.model_params["in_channels"]
#         bottleneck_mode = self.model_params["bottleneck_mode"]
#         drop_path_rate = self.model_params.get("drop_path_rate", 0.0)
#         self.in_chans = in_channels
#         self.out_channel_num = self.model_params["out_channel_num"]
#         self.use_conv1_weight_on_additional_channel = self.model_params.get(
#             "use_conv1_weight_on_additional_channel", False
#         )

#         self.backbone = ResNet3dCSN(
#             pretrained2d=False,
#             in_channels=in_channels,
#             pretrained=None,
#             depth=int(depth),
#             with_pool2=False,
#             bottleneck_mode=bottleneck_mode,
#             norm_eval=False,
#             zero_init_residual=False,
#             drop_path_rate=drop_path_rate,
#         )

#         if phase == "train":
#             ckpt = torch.load(self.model_params["pretrain_path"])
#             if (
#                 "vmz_ircsn_ig65m_pretrained_r50_32x2x1_58e_kinetics400_rgb_20210617-86d33018.pth"
#                 not in self.model_params["pretrain_path"]
#             ):
#                 org_state_dict = ckpt["state_dict"]
#             else:
#                 org_state_dict = ckpt
#             state_dict = intersect_dicts(org_state_dict, self.state_dict(), exclude=[])  # intersect
#             self.load_state_dict(state_dict, strict=False)

#             print(
#                 "Transferred %g/%g items from %s"
#                 % (len(state_dict), len(self.state_dict()), self.model_params["pretrain_path"])
#             )  # report

#             if self.use_conv1_weight_on_additional_channel:
#                 if self.in_chans != 3:
#                     conv1_weight = self.backbone.conv1.conv.weight.clone()
#                     idx = 0
#                     while idx < self.in_chans:
#                         end_idx = min(self.in_chans, idx + 3)
#                         conv1_weight[:, idx:end_idx, ...] = org_state_dict["backbone.conv1.conv.weight"].data
#                         idx = end_idx
#                         print(f"COPIED WEIGHTS {idx}: {end_idx}")
#                     self.backbone.conv1.conv.weight = nn.Parameter(conv1_weight)
#             del org_state_dict

#             self.model = self.backbone

#         self.unet_decoder = UnetDecoder(
#             encoder_channels=(-1, 64, 256, 512, 1024, 2048),
#             decoder_channels=(256, 128, 64, 32, 16),
#             use_batchnorm=True,
#         )
#         self.extract_type = self.model_params.get("extract_type", "max")
#         self.do_normalize = self.model_params.get("do_normalize", False)
#         self.head = SegmentationHead(
#             in_channels=16,
#             out_channels=self.out_channel_num,
#         )

#     def forward(self, inputs: Dict[str, Any]):
#         x = inputs["image"]
#         if self.do_normalize:
#             mean = x.mean(dim=(1, 2, 3), keepdim=True)
#             std = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
#             x = (x - mean) / std

#         bs, chxnum, h, w = x.shape
#         c, im_num = self.in_chans, chxnum // self.in_chans

#         x = x.reshape(bs, im_num, c, h, w)
#         x = x.permute(0, 2, 1, 3, 4)  # -> bs c, im_num, c, h, w

#         img_feats = self.backbone(x)

#         x = [
#             None,
#         ]

#         for i in range(5):
#             if self.extract_type == "max":
#                 _x = img_feats[i].amax(2)
#             elif self.extract_type == "mean":
#                 _x = img_feats[i].mean(2)
#             elif self.extract_type == "last":
#                 _x = img_feats[i][:, :, -1, ...]
#             else:
#                 ValueError("Unknown `extract_type`")

#             x.append(_x)
#         output = self.unet_decoder(*x)
#         output = self.head(output)

#         if self.out_channel_num > 1:
#             if self.training:
#                 return output
#             else:
#                 output = output[:, 0:1, ...]
#                 return output

#         return output


class Conv3dBlockV3(torch.nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int, int], padding: tuple[int, int, int]
    ):
        super().__init__(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, padding_mode="replicate"),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
        )


class UNet25DV2(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.classification_params = self.model_params.get("classification_params", None)
        self.phase = phase

        self.model = smp.Unet(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            classes=1,
            aux_params=self.classification_params,
        )

        self.in_channel_num = self.model_params["in_channel_num"]
        grad_checkpointing = self.model_params.get("grad_checkpointing", False)
        self.model.encoder.model.set_grad_checkpointing(grad_checkpointing)

        channels = self.model.encoder.out_channels
        conv3ds = [
            torch.nn.Sequential(
                Conv3dBlockV3(ch, ch, (2, 3, 3), (0, 1, 1)),
                Conv3dBlockV3(ch, ch, (2, 3, 3), (0, 1, 1)),
            )
            for ch in self.model.encoder.out_channels[1:-1]
        ]
        conv3ds.append(Conv3dBlockV3(channels[-1], channels[-1], (3, 3, 3), (0, 1, 1)))
        self.conv3ds = torch.nn.ModuleList(conv3ds)
        self.n_frames = 3

    def _to2d(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        ret = []
        for conv3d, feature in zip(self.conv3ds, features):
            total_batch, ch, H, W = feature.shape
            feat_3d = feature.reshape(total_batch // self.n_frames, self.n_frames, ch, H, W).permute(0, 2, 1, 3, 4)
            feat_2d = conv3d(feat_3d).squeeze(2)
            ret.append(feat_2d)
        return ret

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]

        bs, c, h, w = x.shape
        x = x.view(-1, self.in_channel_num, h, w)  # (bs * frames) x channel x h x w

        self.model.check_input_shape(x)
        features = self.model.encoder(x)

        features[1:] = self._to2d(features[1:])

        # -> bs x out_ch (from each frame) x frame x h x w
        decoder_output = self.model.decoder(*features)
        output = self.model.segmentation_head(decoder_output)

        return output


class Conv3dBlockV4(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int, int], padding: tuple[int, int, int]
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, padding_mode="replicate"),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
        )

        self.conv2 = nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, padding_mode="replicate"),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
        )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)
        x += residual[:, :, -2:-1, ...]

        return x


class UNet25DV3(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.classification_params = self.model_params.get("classification_params", None)
        self.phase = phase

        self.model = smp.Unet(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            classes=1,
            aux_params=self.classification_params,
        )

        self.in_channel_num = self.model_params["in_channel_num"]
        grad_checkpointing = self.model_params.get("grad_checkpointing", False)
        self.model.encoder.model.set_grad_checkpointing(grad_checkpointing)

        channels = self.model.encoder.out_channels
        conv3ds = [
            torch.nn.Sequential(
                Conv3dBlockV4(ch, ch, (2, 3, 3), (0, 1, 1)),
            )
            for ch in self.model.encoder.out_channels[1:-1]
        ]
        conv3ds.append(Conv3dBlockV4(channels[-1], channels[-1], (2, 3, 3), (0, 1, 1)))
        self.conv3ds = torch.nn.ModuleList(conv3ds)
        self.n_frames = 3

    def _to2d(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        ret = []
        for conv3d, feature in zip(self.conv3ds, features):
            total_batch, ch, H, W = feature.shape
            feat_3d = feature.reshape(total_batch // self.n_frames, self.n_frames, ch, H, W).permute(0, 2, 1, 3, 4)
            feat_2d = conv3d(feat_3d).squeeze(2)
            ret.append(feat_2d)
        return ret

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]

        bs, c, h, w = x.shape
        x = x.view(-1, self.in_channel_num, h, w)  # (bs * frames) x channel x h x w

        self.model.check_input_shape(x)
        features = self.model.encoder(x)

        features[1:] = self._to2d(features[1:])

        # -> bs x out_ch (from each frame) x frame x h x w
        decoder_output = self.model.decoder(*features)
        output = self.model.segmentation_head(decoder_output)

        return output


class Conv3dBlockV5(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        padding: tuple[int, int, int],
        use_residual: bool = True,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            torch.nn.Conv3d(in_channels, in_channels * 2, kernel_size, padding=padding, padding_mode="replicate"),
            torch.nn.BatchNorm3d(in_channels * 2),
            torch.nn.LeakyReLU(),
        )

        self.conv2 = nn.Sequential(
            torch.nn.Conv3d(in_channels * 2, out_channels, kernel_size, padding=padding, padding_mode="replicate"),
            torch.nn.BatchNorm3d(out_channels),
        )
        self.use_residual = use_residual
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.use_residual:
            x += residual[:, :, -2:-1, ...]

        x = self.act(x)

        return x


class UNet25DV4(nn.Module):
    def __init__(self, model_params: Dict[str, Any], phase="test"):
        super().__init__()
        self.model_params = model_params["model_params"]
        self.model_name = self.model_params["backbone"]
        self.classification_params = self.model_params.get("classification_params", None)
        self.phase = phase

        self.model = smp.Unet(
            encoder_name=f"tu-{self.model_name}",
            encoder_weights="imagenet" if self.model_params["from_pretrained"] else None,
            in_channels=self.model_params["in_channel_num"],
            classes=1,
            aux_params=self.classification_params,
        )

        self.in_channel_num = self.model_params["in_channel_num"]
        grad_checkpointing = self.model_params.get("grad_checkpointing", False)
        self.model.encoder.model.set_grad_checkpointing(grad_checkpointing)
        self.use_residual = self.model_params.get("use_residual", True)

        channels = self.model.encoder.out_channels
        conv3ds = [
            torch.nn.Sequential(
                Conv3dBlockV5(ch, ch, (2, 3, 3), (0, 1, 1), self.use_residual),
            )
            for ch in self.model.encoder.out_channels[1:-1]
        ]
        conv3ds.append(Conv3dBlockV5(channels[-1], channels[-1], (2, 3, 3), (0, 1, 1), self.use_residual))
        self.conv3ds = torch.nn.ModuleList(conv3ds)
        self.n_frames = 3

    def _to2d(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        ret = []
        for conv3d, feature in zip(self.conv3ds, features):
            total_batch, ch, H, W = feature.shape
            feat_3d = feature.reshape(total_batch // self.n_frames, self.n_frames, ch, H, W).permute(0, 2, 1, 3, 4)
            feat_2d = conv3d(feat_3d).squeeze(2)
            ret.append(feat_2d)
        return ret

    def forward(self, inputs: Dict[str, Any]):
        x = inputs["image"]

        bs, c, h, w = x.shape
        x = x.view(-1, self.in_channel_num, h, w)  # (bs * frames) x channel x h x w

        self.model.check_input_shape(x)
        features = self.model.encoder(x)

        features[1:] = self._to2d(features[1:])

        # -> bs x out_ch (from each frame) x frame x h x w
        decoder_output = self.model.decoder(*features)
        output = self.model.segmentation_head(decoder_output)

        return output
