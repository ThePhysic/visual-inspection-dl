"""Pinned torchvision SSDLite construction and image/geometry conventions.

Images are EXIF-transposed, converted to RGB and supplied as float CHW [0,1].
Torchvision resizes (without letterboxing) to 320x320, normalizes with mean/std
0.5, and maps boxes back to the EXIF-oriented original frame in eval mode.
Class zero is background; num_classes includes that background class.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

ARCHITECTURE = "ssdlite320_mobilenet_v3_large"
PREPROCESSING = "ssdlite320_rgb_unit_v1"
PRETRAINED_URL = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.url


def build_detector(
    num_classes: int,
    pretrained: bool = False,
    *,
    score_threshold: float = 0.001,
    nms_threshold: float = 0.55,
    max_detections: int = 100,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build exact release architecture; no network access unless pretrained=True."""
    if isinstance(num_classes, bool) or num_classes < 2:
        raise ValueError("num_classes must include background and a foreground class")
    if not 0 <= score_threshold <= 1 or not 0 < nms_threshold <= 1:
        raise ValueError("invalid score/NMS threshold")
    if max_detections < 1:
        raise ValueError("max_detections must be positive")
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,
        num_classes=None if pretrained else num_classes,
        score_thresh=score_threshold,
        nms_thresh=nms_threshold,
        detections_per_img=max_detections,
        topk_candidates=300,
    )
    if pretrained:
        # Keep learned backbone and box regression; replace COCO class predictions.
        old_head = model.head.classification_head
        channels = [block[0][0].in_channels for block in old_head.module_list]
        anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            channels,
            anchors,
            num_classes,
            partial(nn.BatchNorm2d, eps=0.001, momentum=0.03),
        )
    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    return model


def image_tensor(image: Image.Image) -> torch.Tensor:
    """Decode-oriented RGB tensor; copy ensures a writable contiguous array."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255)


def horizontal_flip(image: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reflect exclusive xyxy boxes with their image, preserving empty targets."""
    width = image.shape[-1]
    reflected = boxes.clone()
    reflected[:, 0] = width - boxes[:, 2]
    reflected[:, 2] = width - boxes[:, 0]
    return image.flip(-1), reflected


def load_checkpoint_model(checkpoint: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load a trusted project checkpoint using tensor-only deserialization."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint architecture mismatch")
    classes = payload["classes"]
    model = build_detector(len(classes) + 1)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload
