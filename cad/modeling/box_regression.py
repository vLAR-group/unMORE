# Copyright (c) Facebook, Inc. and its affiliates.
# Original code from: https://github.com/shahaf-arica/CuVLER/blob/main/cad/modeling/box_regression.py

from typing import List, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss

from detectron2.layers import cat, ciou_loss, diou_loss
from detectron2.structures import Boxes

from detectron2.modeling.box_regression import Box2BoxTransform


def _soft_dense_box_regression_loss(
    anchors: List[Union[Boxes, torch.Tensor]],
    box2box_transform: Box2BoxTransform,
    pred_anchor_deltas: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    fg_weights: torch.Tensor,
    fg_mask: torch.Tensor,
    box_reg_loss_type="smooth_l1",
    smooth_l1_beta=0.0,
    reduction="sum",
):
    """
    Compute loss for dense multi-level box regression.
    Loss is accumulated over ``fg_mask``.

    Args:
        anchors: #lvl anchor boxes, each is (HixWixA, 4)
        pred_anchor_deltas: #lvl predictions, each is (N, HixWixA, 4)
        gt_boxes: N ground truth boxes, each has shape (R, 4) (R = sum(Hi * Wi * A))
        fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
        box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou",
            "diou", "ciou".
        smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
            use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
    """
    if isinstance(anchors[0], Boxes):
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
    else:
        anchors = cat(anchors)
    if box_reg_loss_type == "smooth_l1":
        gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
        loss_box_reg = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[fg_mask],
            gt_anchor_deltas[fg_mask],
            beta=smooth_l1_beta,
            reduction=reduction,
        )
    elif box_reg_loss_type == "giou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = giou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction=reduction
        )
    elif box_reg_loss_type == "diou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = diou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction=reduction
        )
    elif box_reg_loss_type == "ciou":
        pred_boxes = [
            box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
        ]
        loss_box_reg = ciou_loss(
            torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction=reduction
        )
    else:
        raise ValueError(f"Invalid dense box regression loss type '{box_reg_loss_type}'")
    fg_weights = fg_weights.unsqueeze(0).unsqueeze(-1)
    loss_box_reg = loss_box_reg * fg_weights
    loss_box_reg = loss_box_reg.sum()
    return loss_box_reg
