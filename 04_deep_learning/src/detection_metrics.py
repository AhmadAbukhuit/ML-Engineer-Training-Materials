"""
Object Detection Spatial Metrics: IoU and Non-Maximum Suppression (NMS).

This module implements:
1. compute_iou_matrix: Fully vectorized pairwise Intersection over Union between two sets of boxes.
2. non_max_suppression: Greedy bounding box suppression based on confidence scores and IoU threshold.
"""

import numpy as np


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Intersection over Union (IoU) between two sets of boxes.
    Boxes are expected in [x1, y1, x2, y2] format.

    Args:
        boxes_a: Array of shape (N, 4)
        boxes_b: Array of shape (M, 4)

    Returns:
        iou_matrix: Array of shape (N, M) containing IoU values in [0.0, 1.0].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Broadcast to shape (N, M, 2)
    # Top-left intersection coordinates: max(x1_a, x1_b), max(y1_a, y1_b)
    tl = np.maximum(boxes_a[:, np.newaxis, :2], boxes_b[np.newaxis, :, :2])
    # Bottom-right intersection coordinates: min(x2_a, x2_b), min(y2_a, y2_b)
    br = np.minimum(boxes_a[:, np.newaxis, 2:], boxes_b[np.newaxis, :, 2:])

    # Width and height of intersection (clamp at 0 for non-overlapping boxes)
    intersection_wh = np.maximum(0.0, br - tl)
    intersection_area = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]

    # Individual box areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # Union area = Area(A) + Area(B) - Intersection(A, B)
    union_area = area_a[:, np.newaxis] + area_b[np.newaxis, :] - intersection_area

    # Avoid division by zero
    eps = 1e-12
    return intersection_area / np.maximum(union_area, eps)


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.25
) -> np.ndarray:
    """
    Applies Non-Maximum Suppression (NMS) to eliminate redundant candidate bounding boxes.

    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Array of shape (N,) containing confidence scores.
        iou_threshold: Overlap threshold for suppressing overlapping boxes.
        score_threshold: Minimum confidence score to consider.

    Returns:
        keep_indices: Array of indices of preserved boxes, sorted by descending score.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    # 1. Filter out candidate boxes below score threshold
    valid_mask = scores >= score_threshold
    if not np.any(valid_mask):
        return np.array([], dtype=int)

    valid_indices = np.where(valid_mask)[0]
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]

    # 2. Sort candidate indices by confidence score descending
    order = np.argsort(-valid_scores)
    keep = []

    while len(order) > 0:
        # Highest score candidate
        current = order[0]
        keep.append(valid_indices[current])

        if len(order) == 1:
            break

        # Compute IoU between current box and all remaining boxes
        current_box = valid_boxes[current : current + 1]
        remaining_boxes = valid_boxes[order[1:]]
        ious = compute_iou_matrix(current_box, remaining_boxes)[0]

        # Keep only remaining boxes whose IoU is strictly below the threshold
        remaining_mask = ious < iou_threshold
        order = order[1:][remaining_mask]

    return np.array(keep, dtype=int)
