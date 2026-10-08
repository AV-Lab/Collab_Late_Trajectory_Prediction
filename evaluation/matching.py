import numpy as np
from scipy.optimize import linear_sum_assignment


def _axis_aligned_bbox(box):
    x, y, _, length, width, _ = np.asarray(box, dtype=np.float64)[:6]
    return np.array(
        [x - length / 2.0, y - width / 2.0, x + length / 2.0, y + width / 2.0],
        dtype=np.float64,
    )


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    intersection_width = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    intersection_height = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    intersection = intersection_width * intersection_height
    if intersection == 0.0:
        return 0.0

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return float(intersection / (area_a + area_b - intersection))


def match_predictions(predictions, ground_truth, iou_threshold):
    """One-to-one, category-compatible Hungarian matching using 2D AABB IoU."""
    gt_items = list(ground_truth.items())
    if not gt_items or not predictions:
        return [], [gt_id for gt_id, _ in gt_items], list(range(len(predictions)))

    similarities = np.zeros((len(gt_items), len(predictions)), dtype=np.float64)
    compatible = np.zeros_like(similarities, dtype=bool)

    for gt_index, (_, gt_object) in enumerate(gt_items):
        gt_category = str(gt_object["category"]).lower()
        gt_box = _axis_aligned_bbox(gt_object["current_state"])

        for prediction_index, prediction in enumerate(predictions):
            prediction_category = str(prediction["category"]).lower()
            if prediction_category != gt_category:
                continue
            compatible[gt_index, prediction_index] = True
            similarities[gt_index, prediction_index] = _iou(
                gt_box,
                _axis_aligned_bbox(prediction["cur_location"]),
            )

    valid = compatible & (similarities >= iou_threshold)
    cost = 1.0 - similarities
    cost[~valid] = 2.0
    rows, columns = linear_sum_assignment(cost)

    matches = []
    matched_gt = set()
    matched_predictions = set()
    for row, column in zip(rows, columns):
        if valid[row, column]:
            gt_id = gt_items[row][0]
            matches.append((gt_id, int(column)))
            matched_gt.add(gt_id)
            matched_predictions.add(int(column))

    missed_gt = [gt_id for gt_id, _ in gt_items if gt_id not in matched_gt]
    false_predictions = [
        index for index in range(len(predictions)) if index not in matched_predictions
    ]
    return matches, missed_gt, false_predictions
