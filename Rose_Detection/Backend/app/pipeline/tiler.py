"""SAHI-style sliding-window tiling with NMS and spatial merge."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.pipeline.classifier import TFLiteClassifier
from app.schemas import BBox, Detection


def generate_tile_coords(
    img_w: int,
    img_h: int,
    tile_w: int,
    tile_h: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) for each tile with edge clamping."""
    step_x = max(1, int(tile_w * (1 - overlap)))
    step_y = max(1, int(tile_h * (1 - overlap)))

    coords: list[tuple[int, int, int, int]] = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + tile_w, img_w)
            y2 = min(y + tile_h, img_h)
            # Clamp start so tile is always tile_w x tile_h when possible
            x1 = max(0, x2 - tile_w)
            y1 = max(0, y2 - tile_h)
            coords.append((x1, y1, x2, y2))
            if x2 >= img_w:
                break
            x += step_x
        if y2 >= img_h:
            break
        y += step_y

    return coords


def compute_iou(a: BBox, b: BBox) -> float:
    """Compute intersection-over-union between two bounding boxes."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def nms_per_class(
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    """Greedy NMS applied independently per class label."""
    by_class: dict[str, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.label, []).append(det)

    kept: list[Detection] = []
    for class_dets in by_class.values():
        # Sort by confidence descending
        class_dets.sort(key=lambda d: d.confidence, reverse=True)
        suppressed = [False] * len(class_dets)
        for i, det_i in enumerate(class_dets):
            if suppressed[i]:
                continue
            kept.append(det_i)
            for j in range(i + 1, len(class_dets)):
                if suppressed[j]:
                    continue
                if compute_iou(det_i.bbox, class_dets[j].bbox) >= iou_threshold:
                    suppressed[j] = True

    return kept


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    """Return True if two boxes overlap at all (intersection area > 0)."""
    return not (a.x2 <= b.x1 or b.x2 <= a.x1 or a.y2 <= b.y1 or b.y2 <= a.y1)


def merge_overlapping_per_class(detections: list[Detection]) -> list[Detection]:
    """Merge overlapping same-class detections into super-boxes via union-find.

    For each connected component of overlapping same-class tiles:
    - bbox = bounding rectangle of all tiles in the group
    - confidence = max confidence in the group
    - One merged Detection per cluster
    """
    if not detections:
        return []

    by_class: dict[str, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.label, []).append(det)

    merged: list[Detection] = []
    for label, class_dets in by_class.items():
        n = len(class_dets)
        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if _boxes_overlap(class_dets[i].bbox, class_dets[j].bbox):
                    union(i, j)

        # Group by root
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        for indices in groups.values():
            dets = [class_dets[i] for i in indices]
            super_bbox = BBox(
                x1=min(d.bbox.x1 for d in dets),
                y1=min(d.bbox.y1 for d in dets),
                x2=max(d.bbox.x2 for d in dets),
                y2=max(d.bbox.y2 for d in dets),
            )
            best = max(dets, key=lambda d: d.confidence)
            merged.append(
                Detection(
                    bbox=super_bbox,
                    label=label,
                    confidence=best.confidence,
                    frame_index=best.frame_index,
                )
            )

    return merged


def detect_in_image(
    image: Image.Image,
    classifier: TFLiteClassifier,
    frame_index: int,
    tile_size: int = 224,
    tile_overlap: float = 0.5,
    confidence_threshold: float = 0.5,
    nms_iou_threshold: float = 0.45,
) -> list[Detection]:
    """Tile an image, classify each tile, filter by confidence, run NMS."""
    img_w, img_h = image.size
    img_arr = np.array(image.convert("RGB"), dtype=np.float32)

    # Small images: classify whole image as one detection
    if img_w <= tile_size and img_h <= tile_size:
        resized = image.resize((tile_size, tile_size))
        tile_arr = np.array(resized, dtype=np.float32)
        label, conf, _ = classifier.classify_tile(tile_arr)
        if conf >= confidence_threshold:
            return [
                Detection(
                    bbox=BBox(x1=0, y1=0, x2=img_w, y2=img_h),
                    label=label,
                    confidence=round(conf, 4),
                    frame_index=frame_index,
                )
            ]
        return []

    coords = generate_tile_coords(img_w, img_h, tile_size, tile_size, tile_overlap)

    raw_detections: list[Detection] = []
    for x1, y1, x2, y2 in coords:
        crop = img_arr[y1:y2, x1:x2]

        # Resize if the crop is smaller than tile_size (edge tiles)
        if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
            crop_img = Image.fromarray(crop.astype(np.uint8)).resize((tile_size, tile_size))
            crop = np.array(crop_img, dtype=np.float32)

        label, conf, _ = classifier.classify_tile(crop)
        if conf >= confidence_threshold:
            raw_detections.append(
                Detection(
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    label=label,
                    confidence=round(conf, 4),
                    frame_index=frame_index,
                )
            )

    after_nms = nms_per_class(raw_detections, nms_iou_threshold)
    return merge_overlapping_per_class(after_nms)
