from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.dataset.road_damage_dataset import NUM_DETECTION_CLASSES
from src.utils.common import resolve_inference_weights


def build_faster_rcnn(num_classes: int = NUM_DETECTION_CLASSES, pretrained: bool = True) -> torch.nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    # Explicitly disable implicit ImageNet backbone weights when loading a local
    # detector checkpoint; torchvision otherwise downloads them even though the
    # complete state dict is replaced immediately afterwards.
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_faster_rcnn(weights: str | Path, device: str) -> torch.nn.Module:
    weight_path = resolve_inference_weights(weights, "runs/faster_rcnn", ".pth")
    model = build_faster_rcnn(pretrained=False)
    # Loading on CPU avoids duplicating the optimizer state in scarce GPU memory.
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    print(build_faster_rcnn(pretrained=False).__class__.__name__)


if __name__ == "__main__":
    main()
