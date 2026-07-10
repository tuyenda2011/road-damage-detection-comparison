from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def match_predictions(
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be between 0 and 1.")
    if pred_boxes.shape != (len(pred_labels), 4) or len(pred_labels) != len(pred_scores):
        raise ValueError("Prediction boxes, labels and scores have inconsistent shapes.")
    if gt_boxes.shape != (len(gt_labels), 4):
        raise ValueError("Ground-truth boxes and labels have inconsistent shapes.")
    if not np.isfinite(pred_boxes).all() or not np.isfinite(pred_scores).all() or not np.isfinite(gt_boxes).all():
        raise ValueError("Detection arrays must contain finite values.")
    order = np.argsort(-pred_scores) if len(pred_scores) else []
    matched_gt: set[int] = set()
    tp = 0
    fp = 0

    for pred_idx in order:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt or int(gt_labels[gt_idx]) != int(pred_labels[pred_idx]):
                continue
            iou = calculate_iou(pred_boxes[pred_idx], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def precision_score(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall_score(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) else 0.0


def average_precision_for_class(
    predictions: list[dict],
    ground_truths: list[dict],
    class_id: int,
    iou_threshold: float = 0.5,
) -> float:
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length.")
    rows: list[tuple[int, float, np.ndarray]] = []
    total_gt = 0
    gt_by_image: dict[int, list[np.ndarray]] = {}
    matched: dict[int, set[int]] = {}

    for image_idx, gt in enumerate(ground_truths):
        boxes = gt["boxes"][gt["labels"] == class_id]
        gt_by_image[image_idx] = [box for box in boxes]
        matched[image_idx] = set()
        total_gt += len(boxes)

    if total_gt == 0:
        return 0.0

    for image_idx, pred in enumerate(predictions):
        keep = pred["labels"] == class_id
        for box, score in zip(pred["boxes"][keep], pred["scores"][keep]):
            rows.append((image_idx, float(score), box))

    if not rows:
        return 0.0
    rows.sort(key=lambda item: item[1], reverse=True)

    tp = np.zeros(len(rows), dtype=np.float32)
    fp = np.zeros(len(rows), dtype=np.float32)
    for idx, (image_idx, _, pred_box) in enumerate(rows):
        gt_boxes = gt_by_image.get(image_idx, [])
        best_iou = 0.0
        best_gt = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched[image_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_iou >= iou_threshold and best_gt >= 0:
            matched[image_idx].add(best_gt)
            tp[idx] = 1.0
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(total_gt, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    change = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[change + 1] - recalls[change]) * precisions[change + 1]))


def map50(predictions: list[dict], ground_truths: list[dict], num_classes: int = 4) -> float:
    """Calculate mAP@50 over classes represented in the ground truth.

    Ignoring absent classes matches common detection-tool behavior and prevents a
    small evaluation split from being penalized merely for not containing a class.
    """
    present_classes = {
        int(class_id)
        for ground_truth in ground_truths
        for class_id in np.asarray(ground_truth["labels"]).reshape(-1)
        if 0 <= int(class_id) < num_classes
    }
    aps = [
        average_precision_for_class(predictions, ground_truths, class_id)
        for class_id in sorted(present_classes)
    ]
    return float(np.mean(aps)) if aps else 0.0


def aggregate_precision_recall(predictions: list[dict], ground_truths: list[dict]) -> tuple[float, float]:
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length.")
    tp_total = fp_total = fn_total = 0
    for pred, gt in zip(predictions, ground_truths):
        tp, fp, fn = match_predictions(
            pred["boxes"], pred["labels"], pred["scores"], gt["boxes"], gt["labels"], iou_threshold=0.5
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn
    return precision_score(tp_total, fp_total), recall_score(tp_total, fn_total)


def measure_fps(infer_fn: Callable[[], object], warmup: int = 3, repeat: int = 20) -> float:
    if warmup < 0 or repeat <= 0:
        raise ValueError("warmup cannot be negative and repeat must be positive.")
    for _ in range(warmup):
        infer_fn()
    start = time.perf_counter()
    for _ in range(repeat):
        infer_fn()
    elapsed = max(time.perf_counter() - start, 1e-9)
    return repeat / elapsed


def main() -> None:
    box_a = [0, 0, 100, 100]
    box_b = [50, 50, 150, 150]
    print(f"Sample IoU: {calculate_iou(box_a, box_b):.4f}")


if __name__ == "__main__":
    main()
