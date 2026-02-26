"""IoU and NMS helpers."""

from __future__ import annotations

from app.schemas import BBox, Detection


def compute_iou(a: BBox, b: BBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def nms_per_class(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy NMS applied independently per class label."""
    by_class: dict[str, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.label, []).append(det)

    kept: list[Detection] = []
    for class_dets in by_class.values():
        class_dets.sort(key=lambda d: d.confidence, reverse=True)
        suppressed = [False] * len(class_dets)
        for i, det_i in enumerate(class_dets):
            if suppressed[i]:
                continue
            kept.append(det_i)
            for j in range(i + 1, len(class_dets)):
                if not suppressed[j] and compute_iou(det_i.bbox, class_dets[j].bbox) >= iou_threshold:
                    suppressed[j] = True
    return kept
