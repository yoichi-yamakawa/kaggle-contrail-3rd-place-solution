import math
from typing import *

import numpy as np
import torch
import torchaudio.transforms as at
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class GeMP(nn.Module):
    """from: https://github.com/knjcode/kaggle-seti-2021/blob/master/working/model.py
    referred at https://www.kaggle.com/c/seti-breakthrough-listen/discussion/266403
    """

    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        super().__init__()
        self._p = p
        self._learn_p = learn_p
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.set_learn_p(flag=learn_p)

    def set_learn_p(self, flag):
        self._learn_p = flag
        self.p.requires_grad = flag

    def forward(self, x):
        # x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), 1)).pow(1.0 / self.p)

        return x


class SwinGeMP(nn.Module):
    """from: https://github.com/knjcode/kaggle-seti-2021/blob/master/working/model.py
    referred at https://www.kaggle.com/c/seti-breakthrough-listen/discussion/266403
    """

    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        super().__init__()
        self._p = p
        self._learn_p = learn_p
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.set_learn_p(flag=learn_p)

    def set_learn_p(self, flag):
        self._learn_p = flag
        self.p.requires_grad = flag

    def forward(self, x):
        x = F.adaptive_avg_pool1d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p)

        return x


class GeM1d(nn.Module):
    """
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    """

    def __init__(self, kernel_size=8, stride=None, p=3, eps=1e-6):
        super(GeM1d, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.kernel_size = kernel_size
        self.eps = eps
        self.stride = stride

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size, self.stride).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Mixup(nn.Module):
    """from: https://www.kaggle.com/ilu000/2nd-place-birdclef2021-inference"""

    def __init__(self, mix_beta, label_mix_type="mix"):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.label_mix_type = label_mix_type

    def forward(self, X, Y, weight=None):
        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).type_as(X)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        y_coeffs = coeffs
        if self.label_mix_type == "mix":
            Y = y_coeffs * Y + (1 - y_coeffs) * Y[perm]
        elif self.label_mix_type == "max":
            Y = Y + Y[perm] - Y * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight


class PositionalFeaturesBlockV1(nn.Module):
    def __init__(self, pfb_params: Dict[str, Any]):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(pfb_params["input_dim"], pfb_params["dim1"]),
            nn.LayerNorm(pfb_params["dim1"]),
            nn.ReLU(),
            nn.Dropout(pfb_params["drop_out_p"]),
            nn.Linear(pfb_params["dim1"], 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(pfb_params["drop_out_p"]),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Conv3dBlock(nn.Module):
    def __init__(self, conv_params: Dict[str, Any]):
        super().__init__()
        out_ch = conv_params["out_channel_num"]

        self.conv1 = nn.Conv3d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv3d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
        )
        self.bn3 = nn.BatchNorm3d(out_ch)
        self.act3 = nn.ReLU()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x += residual

        x = self.act3(x)

        return x


class Conv3dBlockV2(nn.Module):
    def __init__(self, mid_channle_num: int):
        super().__init__()
        self.mid_ch = mid_channle_num
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(self.mid_ch, self.mid_ch, (3, 9, 9), padding=(1, 4, 4), padding_mode="replicate"),
            torch.nn.BatchNorm3d(self.mid_ch),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(self.mid_ch, self.mid_ch, (3, 9, 9), padding=(1, 4, 4), padding_mode="replicate"),
            torch.nn.BatchNorm3d(self.mid_ch),
            torch.nn.LeakyReLU(),
        )

        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x += shortcut
        x = self.act(x)

        return x
