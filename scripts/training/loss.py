import math

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.nn import functional as F


class DiceLossWithConstantCorrection(nn.Module):
    def __init__(self, smooth=1e-6, numerator=0.0, denominator=0.0):
        super(DiceLossWithConstantCorrection, self).__init__()
        self.smooth = smooth
        self.numerator = numerator
        self.denominator = denominator

    def forward(self, pred, label):
        pred = pred.flatten()
        label = label.flatten()
        intersection = (label * pred).sum()
        return (2.0 * intersection + self.smooth + self.numerator) / (
            label.sum() + pred.sum() + self.smooth + self.denominator
        )


class DiceLoss(nn.Module):
    def __init__(self, mode, classes=None, log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07):
        super(DiceLoss, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )

    def forward(self, outputs, batch):
        loss = self.loss_func(outputs, batch["target"])
        return loss


class DiceLossV2(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.5,
    ):
        super(DiceLossV2, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.base_weight = base_weight
        self.aux_weight = aux_weight

    def forward(self, outputs, batch):
        loss = self.loss_func(outputs, batch["target"])
        aux_loss = self.loss_func(outputs, batch["individual_mask"])

        loss = loss * self.base_weight + aux_loss * self.aux_weight
        return loss


class DiceLossV3(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.5,
    ):
        super(DiceLossV3, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.base_weight = base_weight
        self.aux_weight = aux_weight

    def forward(self, outputs, batch):
        loss = self.loss_func(outputs[:, 0:1, ...], batch["target"])
        aux_loss = self.loss_func(outputs[:, 1:, ...], batch["individual_mask"])

        loss = loss * self.base_weight + aux_loss * self.aux_weight
        return loss


class DiceLossMaxMinV1(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.5,
        aux_max_loss_weight=0.5,
        aux_min_loss_weight=0.5,
    ):
        super(DiceLossMaxMinV1, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.base_weight = base_weight
        self.aux_weight = aux_weight
        self.aux_max_loss_weight = aux_max_loss_weight
        self.aux_min_loss_weight = aux_min_loss_weight

    def forward(self, outputs, batch):
        loss = self.loss_func(outputs[:, 0:1, ...], batch["target"])
        aux_loss = self.loss_func(outputs[:, 1:2, ...], batch["individual_mask"])
        aux_max_loss = self.loss_func(outputs[:, 1:2, ...], batch["individual_mask_max"])
        aux_min_loss = self.loss_func(outputs[:, 2:3, ...], batch["individual_mask_min"])

        loss = (
            loss * self.base_weight
            + aux_loss * self.aux_weight
            + aux_max_loss * self.aux_max_loss_weight
            + aux_min_loss * self.aux_min_loss_weight
        )
        return loss


class DiceFLIPLossV1(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.1,
    ):
        super(DiceFLIPLossV1, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.base_weight = base_weight
        self.aux_weight = aux_weight

    def forward(self, outputs, batch):
        loss = self.loss_func(outputs[:, 0:1, ...], batch["target"])
        flip_target = 1 - batch["target"]
        aux_loss = self.loss_func(outputs[:, 1:, ...], flip_target)

        loss = loss * self.base_weight + aux_loss * self.aux_weight
        return loss


class DiceFLIPLossV2(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_ind_weight=0.1,
        aux_flip_weight=0.1,
    ):
        super(DiceFLIPLossV2, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.base_weight = base_weight
        self.aux_flip_weight = aux_flip_weight
        self.aux_ind_weight = aux_ind_weight

    def forward(self, outputs, batch):
        loss = self.loss_func(outputs[:, 0:1, ...], batch["target"])
        flip_target = 1 - batch["target"]
        aux_flip_loss = self.loss_func(outputs[:, 2:, ...], flip_target)
        aux_ind_loss = self.loss_func(outputs[:, 1:2, ...], batch["individual_mask"])

        loss = loss * self.base_weight + aux_flip_loss * self.aux_flip_weight + aux_ind_loss * self.aux_ind_weight
        return loss


class DiceBCEAuxChannelLossV1(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.1,
    ):
        super(DiceBCEAuxChannelLossV1, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.bce_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.base_weight = base_weight
        self.aux_weight = aux_weight

    def forward(self, outputs, batch):
        loss = self.loss_func(outputs[:, 0:1, ...], batch["target"])
        aux_loss = self.bce_func(outputs[:, 1:, ...], batch["target"].type_as(outputs))

        loss = loss * self.base_weight + aux_loss * self.aux_weight
        return loss


class DiceBCEAuxChannelLossV2(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_bce_weight=0.1,
        aux_ind_weight=0.5,
    ):
        super(DiceBCEAuxChannelLossV2, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.bce_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.base_weight = base_weight
        self.aux_bce_weight = aux_bce_weight
        self.aux_ind_weight = aux_ind_weight

    def forward(self, outputs, batch):
        loss = self.loss_func(outputs[:, 0:1, ...], batch["target"])
        aux_ind_loss = self.loss_func(outputs[:, 1:2, ...], batch["individual_mask"])
        aux_loss = self.bce_func(outputs[:, 2:, ...], batch["target"].type_as(outputs))

        loss = loss * self.base_weight + aux_loss * self.aux_bce_weight + aux_ind_loss * self.aux_ind_weight
        return loss


class DiceFocalLossV1(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        gamma=2.0,
        dice_weight=0.5,
        focal_weight=0.5,
    ):
        super(DiceFocalLossV1, self).__init__()
        self.dice_loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )

        self.focal_loss_func = smp.losses.FocalLoss(
            mode,
            gamma=gamma,
        )

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, outputs, batch):
        dice_loss = self.dice_loss_func(outputs, batch["target"])
        focal_loss = self.focal_loss_func(outputs.sigmoid(), batch["target"])

        loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

        return loss


class DiceBCEWithLogitsLossV1(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.5,
        base_bce_weight=0.5,
        aux_bce_weight=0.5,
        use_mixup=False,
        numerator=0.0,
        denominator=0.0,
    ):
        super(DiceBCEWithLogitsLossV1, self).__init__()

        if numerator > 0:
            self.loss_func = DiceLossWithConstantCorrection(
                numerator=numerator,
                denominator=denominator,
            )
        else:
            self.loss_func = smp.losses.DiceLoss(
                mode,
                classes,
                log_loss,
                from_logits,
                smooth,
                ignore_index,
                eps,
            )
        self.use_mixup = use_mixup
        self.bce_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.base_weight = base_weight
        self.aux_weight = aux_weight
        self.base_bce_weight = base_bce_weight
        self.aux_bce_weight = aux_bce_weight

    def forward(self, outputs, batch):
        from_train_idx = batch["from_train"]
        if self.use_mixup:
            outputs, target, aux_target = outputs
        else:
            target = batch["target"]
            aux_target = batch["individual_mask"]

        loss = self.loss_func(outputs, target)
        bce_loss = self.bce_func(outputs, target.type_as(outputs))

        if from_train_idx.sum() > 0:
            aux_loss = self.loss_func(outputs[from_train_idx], aux_target[from_train_idx])
            aux_bce_loss = self.bce_func(outputs[from_train_idx], aux_target[from_train_idx])
        else:
            aux_loss = torch.tensor(0).type_as(outputs)
            aux_bce_loss = torch.tensor(0).type_as(outputs)

        dice_loss = loss * self.base_weight + aux_loss * self.aux_weight
        bce_loss = bce_loss * self.base_bce_weight + aux_bce_loss * self.aux_bce_weight

        loss = dice_loss + bce_loss

        return loss


class DiceBCEWithLogitsLossPseudoLabelV1(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.5,
        base_bce_weight=0.5,
        aux_bce_weight=0.5,
        pseudo_loss_weights=[0.2, 0.2, 0.2, 0.5],
        base_loss_weight=1.0,
        pseudo_loss_weight=1.0,
    ):
        super(DiceBCEWithLogitsLossPseudoLabelV1, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.bce_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.base_weight = base_weight
        self.aux_weight = aux_weight
        self.base_bce_weight = base_bce_weight
        self.aux_bce_weight = aux_bce_weight
        self.pseudo_loss_weights = pseudo_loss_weights
        self.pseudo_loss_weight = pseudo_loss_weight
        self.base_loss_weight = base_loss_weight

    def forward(self, outputs, batch):
        # base target loss
        from_train_idx = batch["from_train"]
        target = batch["target"]
        aux_target = batch["individual_mask"]

        outputs, frames_output = outputs

        loss = self.loss_func(outputs, target)
        bce_loss = self.bce_func(outputs, target.type_as(outputs))

        if from_train_idx.sum() > 0:
            aux_loss = self.loss_func(outputs[from_train_idx], aux_target[from_train_idx])
            aux_bce_loss = self.bce_func(outputs[from_train_idx], aux_target[from_train_idx])
        else:
            aux_loss = torch.tensor(0).type_as(outputs)
            aux_bce_loss = torch.tensor(0).type_as(outputs)

        dice_loss = loss * self.base_weight + aux_loss * self.aux_weight
        bce_loss = bce_loss * self.base_bce_weight + aux_bce_loss * self.aux_bce_weight

        base_loss = dice_loss + bce_loss

        # pseudo label loss
        for frame, weight in enumerate(self.pseudo_loss_weights):
            out_frame = frames_output[:, frame : frame + 1, ...]

            if frame == len(self.pseudo_loss_weights) - 1:
                target = batch["target"]
            else:
                target = batch[f"pseudo_mask{frame}"]

            # print(f"frame: {frame} / {target.shape}")
            frame_loss = self.loss_func(out_frame, target)
            if frame == 0:
                pseudo_loss = frame_loss * weight
            else:
                pseudo_loss = pseudo_loss + frame_loss * weight

        loss = base_loss * self.base_loss_weight + pseudo_loss * self.pseudo_loss_weight

        return loss


class MultiCalssFocalLoss(nn.Module):
    """refer to https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py"""

    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(reduction="none")

    def forward(self, x, y):
        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce
        loss = loss.mean()

        # if self.reduction == 'mean':
        #     loss = loss.mean()
        # elif self.reduction == 'sum':
        #     loss = loss.sum()

        return loss


class CrossEntropyLossWithSoftlabel(nn.Module):
    def __init__(self):
        super(CrossEntropyLossWithSoftlabel, self).__init__()

    def forward(self, x, labels, perm_labels, label_coeffs):
        soft_labels = torch.zeros_like(x).type_as(x)
        soft_labels2 = torch.zeros_like(x).type_as(x)

        soft_labels.scatter_(1, labels.view(-1, 1), label_coeffs.view(-1, 1).type_as(x))
        soft_labels2.scatter_(1, perm_labels.view(-1, 1), 1 - label_coeffs.view(-1, 1).type_as(x))

        soft_labels += soft_labels2

        logprobs = torch.nn.functional.log_softmax(x, dim=-1, dtype=torch.float32)
        loss = -logprobs * soft_labels
        loss = loss.sum(-1)

        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, reduction="mean", alpha=1, gamma=2):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce_loss
        if self.reduction == "none":
            loss = loss
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


class FocalMSEMixLoss(torch.nn.Module):
    def __init__(self, reduction="mean", alpha=1.0, gamma=2.0):
        super(FocalMSEMixLoss, self).__init__()
        self.eps = 1e-6
        self.focal_loss = FocalLoss(reduction="mean", alpha=alpha, gamma=gamma)

    def forward(self, logits, y):
        mse_loss = nn.MSELoss()
        loss = self.focal_loss(logits, y) + mse_loss(logits.sigmoid(), y)

        return loss


class DiceLovaszLossV1(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.5,
        lovasz_weight=0.5,
    ):
        super(DiceLovaszLossV1, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.lovasz_loss_func = smp.losses.LovaszLoss(
            mode,
            from_logits,
        )
        self.base_weight = base_weight
        self.aux_weight = aux_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, outputs, batch):
        loss = self.loss_func(outputs, batch["target"])
        lovasz_loss = self.lovasz_loss_func(outputs, batch["target"])
        aux_loss = self.loss_func(outputs, batch["individual_mask"])

        loss = loss * self.base_weight + aux_loss * self.aux_weight + self.lovasz_weight * lovasz_loss
        return loss


class DiceTverskyLossV1(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.5,
        tversky_weight=0.5,
        alpha=0.5,
        beta=0.5,
    ):
        super(DiceTverskyLossV1, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.tversky_loss_func = smp.losses.TverskyLoss(
            mode=mode,
            alpha=alpha,
            beta=beta,
            from_logits=from_logits,
        )
        self.base_weight = base_weight
        self.aux_weight = aux_weight
        self.tversky_weight = tversky_weight

    def forward(self, outputs, batch):
        loss = self.loss_func(outputs, batch["target"])
        tversky_loss = self.tversky_loss_func(outputs, batch["target"])
        aux_loss = self.loss_func(outputs, batch["individual_mask"])

        loss = loss * self.base_weight + aux_loss * self.aux_weight + self.tversky_weight * tversky_loss
        return loss


class DiceLossMixupV1(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.5,
    ):
        super(DiceLossMixupV1, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.base_weight = base_weight
        self.aux_weight = aux_weight

    def forward(self, outputs, batch):
        output, targets, aux_targets = outputs
        loss = self.loss_func(output, targets)
        aux_loss = self.loss_func(output, aux_targets)

        loss = loss * self.base_weight + aux_loss * self.aux_weight
        return loss


class DiceLossMixupV2(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.5,
        base_bce_weight=0.5,
        aux_bce_weight=0.5,
    ):
        super(DiceLossMixupV2, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.bce_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.base_weight = base_weight
        self.aux_weight = aux_weight
        self.base_bce_weight = base_bce_weight
        self.aux_bce_weight = aux_bce_weight

    def forward(self, outputs, batch):
        output, targets, aux_targets = outputs
        loss = self.loss_func(output, targets)
        aux_loss = self.loss_func(output, aux_targets)

        loss = loss * self.base_weight + aux_loss * self.aux_weight
        return loss

    def forward(self, outputs, batch):
        output, targets, aux_targets = outputs

        loss = self.loss_func(output, targets)
        aux_loss = self.loss_func(output, aux_targets)

        dice_loss = loss * self.base_weight + aux_loss * self.aux_weight

        bce_loss = self.bce_func(output, targets.type_as(output))
        aux_bce_loss = self.bce_func(output, aux_targets)
        bce_loss = bce_loss * self.base_bce_weight + aux_bce_loss * self.aux_bce_weight

        loss = dice_loss + bce_loss

        return loss


class DiceBCEWithLogitsMetaLossV1(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.5,
        base_bce_weight=0.5,
        aux_bce_weight=0.5,
        meta_weight=0.2,
    ):
        super(DiceBCEWithLogitsMetaLossV1, self).__init__()
        self.loss_func = smp.losses.DiceLoss(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
        )
        self.bce_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.base_weight = base_weight
        self.aux_weight = aux_weight
        self.base_bce_weight = base_bce_weight
        self.aux_bce_weight = aux_bce_weight
        self.meta_weight = meta_weight
        self.meta_loss_func = nn.MSELoss()

    def forward(self, outputs, batch):
        from_train_idx = batch["from_train"]
        outputs, meta_logit = outputs
        target = batch["target"]
        aux_target = batch["individual_mask"]

        loss = self.loss_func(outputs, target)
        bce_loss = self.bce_func(outputs, target.type_as(outputs))

        if from_train_idx.sum() > 0:
            aux_loss = self.loss_func(outputs[from_train_idx], aux_target[from_train_idx])
            aux_bce_loss = self.bce_func(outputs[from_train_idx], aux_target[from_train_idx])
        else:
            aux_loss = torch.tensor(0).type_as(outputs)
            aux_bce_loss = torch.tensor(0).type_as(outputs)

        dice_loss = loss * self.base_weight + aux_loss * self.aux_weight
        bce_loss = bce_loss * self.base_bce_weight + aux_bce_loss * self.aux_bce_weight

        loss = dice_loss + bce_loss

        # meta loss
        meta_target = torch.stack(
            [batch["row_min"], batch["row_size"], batch["col_min"], batch["col_size"]], axis=1
        ).type_as(meta_logit)
        meta_loss = self.meta_loss_func(meta_logit, meta_target)

        loss += meta_loss * self.meta_weight

        return loss


class DiceFocalLossV2(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        gamma=2.0,
        base_weight=0.5,
        aux_weight=0.5,
        base_focal_weight=0.5,
        aux_focal_weight=0.5,
        numerator=0.0,
        denominator=0.0,
    ):
        super(DiceFocalLossV2, self).__init__()

        if numerator > 0:
            self.loss_func = DiceLossWithConstantCorrection(
                numerator=numerator,
                denominator=denominator,
            )
        else:
            self.loss_func = smp.losses.DiceLoss(
                mode,
                classes,
                log_loss,
                from_logits,
                smooth,
                ignore_index,
                eps,
            )
        self.focal_loss_func = smp.losses.FocalLoss(
            mode,
            gamma=gamma,
        )
        self.base_weight = base_weight
        self.aux_weight = aux_weight
        self.base_focal_weight = base_focal_weight
        self.aux_focal_weight = aux_focal_weight

    def forward(self, outputs, batch):
        from_train_idx = batch["from_train"]
        target = batch["target"]
        aux_target = batch["individual_mask"]

        loss = self.loss_func(outputs, target)
        focal_loss = self.focal_loss_func(outputs, target.type_as(outputs))

        if from_train_idx.sum() > 0:
            aux_loss = self.loss_func(outputs[from_train_idx], aux_target[from_train_idx])
            aux_focal_loss = self.focal_loss_func(outputs[from_train_idx], aux_target[from_train_idx])
        else:
            aux_loss = torch.tensor(0).type_as(outputs)
            aux_focal_loss = torch.tensor(0).type_as(outputs)

        dice_loss = loss * self.base_weight + aux_loss * self.aux_weight
        focal_loss = focal_loss * self.base_focal_weight + aux_focal_loss * self.aux_focal_weight

        loss = dice_loss + focal_loss

        return loss


class DiceBCEWithLogitsLossV2(nn.Module):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-07,
        base_weight=0.5,
        aux_weight=0.5,
        base_bce_weight=0.5,
        aux_bce_weight=0.5,
        use_mixup=False,
        numerator=0.0,
        denominator=0.0,
    ):
        super(DiceBCEWithLogitsLossV2, self).__init__()

        if numerator > 0:
            self.loss_func = DiceLossWithConstantCorrection(
                numerator=numerator,
                denominator=denominator,
            )
        else:
            self.loss_func = smp.losses.DiceLoss(
                mode,
                classes,
                log_loss,
                from_logits,
                smooth,
                ignore_index,
                eps,
            )
        self.use_mixup = use_mixup
        self.bce_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.aux_bce_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.base_weight = base_weight
        self.aux_weight = aux_weight
        self.base_bce_weight = base_bce_weight
        self.aux_bce_weight = aux_bce_weight

    def forward(self, outputs, batch):
        from_train_idx = batch["from_train"]
        if self.use_mixup:
            outputs, target, aux_target = outputs
        else:
            target = batch["target"]
            aux_target = batch["individual_mask"]

        loss = self.loss_func(outputs, target)
        bce_loss = self.bce_func(outputs, target.type_as(outputs))

        if from_train_idx.sum() > 0:
            aux_loss = self.loss_func(outputs[from_train_idx], aux_target[from_train_idx])
            # modify aggregation
            aux_loss *= len(outputs[from_train_idx])
            aux_loss /= outputs.shape[0]
            aux_bce_loss = self.aux_bce_func(outputs[from_train_idx], aux_target[from_train_idx])
            aux_bce_loss *= len(outputs[from_train_idx])
            aux_bce_loss /= len(outputs)
        else:
            aux_loss = torch.tensor(0).type_as(outputs)
            aux_bce_loss = torch.tensor(0).type_as(outputs)

        dice_loss = loss * self.base_weight + aux_loss * self.aux_weight
        bce_loss = bce_loss * self.base_bce_weight + aux_bce_loss * self.aux_bce_weight

        loss = dice_loss + bce_loss
        if loss > 10000:
            print(f"dice_loss: {dice_loss}")
            print(f"bce_loss: {bce_loss}")
            print(f"aux_loss: {aux_loss}")
            print(f"aux_bce_loss: {aux_bce_loss}")

        return loss
