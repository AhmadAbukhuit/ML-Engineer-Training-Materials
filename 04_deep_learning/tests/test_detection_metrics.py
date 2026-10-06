import sys
from pathlib import Path
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detection_metrics import compute_iou_matrix, non_max_suppression


def test_iou_exact_and_disjoint():
    # Box format: [x1, y1, x2, y2]
    # Box A: [0, 0, 10, 10] (area 100)
    # Box B: [0, 0, 10, 10] (identical -> IoU 1.0)
    # Box C: [20, 20, 30, 30] (disjoint -> IoU 0.0)
    # Box D: [5, 0, 15, 10] (intersection area: 5*10=50. Union: 100+100-50=150. IoU: 50/150 = 1/3)

    boxes_a = np.array([[0.0, 0.0, 10.0, 10.0]])
    boxes_b = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 30.0, 30.0],
        [5.0, 0.0, 15.0, 10.0]
    ])

    ious = compute_iou_matrix(boxes_a, boxes_b)

    assert ious.shape == (1, 3)
    assert np.isclose(ious[0, 0], 1.0)
    assert np.isclose(ious[0, 1], 0.0)
    assert np.isclose(ious[0, 2], 1.0 / 3.0)


def test_non_max_suppression():
    # 3 candidate boxes:
    # Box 0: High score 0.95, [0, 0, 10, 10]
    # Box 1: Overlapping with Box 0 (IoU ~0.8), score 0.85 -> Should be suppressed
    # Box 2: Disjoint, score 0.75, [50, 50, 60, 60] -> Should be kept
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 10.0, 10.0],
        [50.0, 50.0, 60.0, 60.0]
    ])
    scores = np.array([0.95, 0.85, 0.75])

    keep = non_max_suppression(boxes, scores, iou_threshold=0.5, score_threshold=0.5)

    # Should keep Box 0 and Box 2
    assert list(keep) == [0, 2]
