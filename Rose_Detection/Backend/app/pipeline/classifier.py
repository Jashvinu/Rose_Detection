"""YOLO11s TFLite object-detection wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image as PILImage

from app.schemas import BBox, Detection


class YOLOTFLiteDetector:
    """Wraps a YOLO11s TFLite detection model.

    Input:  (1, 640, 640, 3) float32, pixel values normalised [0, 1]
    Output: (1, 10, 8400) float32
            channels 0-3 : cx, cy, w, h in 640-px input space
            channels 4-9 : per-class probabilities (sigmoid applied on export)
    """

    INPUT_SIZE: int = 640

    def __init__(self, model_path: Path, labels_path: Path, **_kwargs):
        self.labels: list[str] = labels_path.read_text().strip().splitlines()

        self._interpreter = tflite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def detect(
        self,
        crop: np.ndarray,
        confidence_threshold: float,
        frame_index: int,
    ) -> list[Detection]:
        """Run inference on a (H, W, 3) uint8 crop.

        Returns detections in crop-pixel coordinates.
        Confidence filtering is applied here; NMS is handled by the tiler.
        """
        h, w = crop.shape[:2]

        # Resize to INPUT_SIZE × INPUT_SIZE and normalise to [0, 1]
        inp_img = PILImage.fromarray(crop).resize(
            (self.INPUT_SIZE, self.INPUT_SIZE), PILImage.BILINEAR
        )
        inp = np.array(inp_img, dtype=np.float32)[np.newaxis] / 255.0  # (1, 640, 640, 3)

        self._interpreter.set_tensor(self._input_details[0]["index"], inp)
        self._interpreter.invoke()

        # (1, 10, 8400) → (8400, 10)
        preds = self._interpreter.get_tensor(self._output_details[0]["index"])[0].T

        class_scores = preds[:, 4:]                                        # (8400, 6)
        class_ids = np.argmax(class_scores, axis=1)                        # (8400,)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]  # (8400,)

        mask = confidences >= confidence_threshold
        if not mask.any():
            return []

        boxes = preds[mask, :4]      # cx, cy, w, h in 640-px space
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Scale from 640-px space to crop-pixel space
        sx, sy = w / self.INPUT_SIZE, h / self.INPUT_SIZE
        cx_a = boxes[:, 0] * sx
        cy_a = boxes[:, 1] * sy
        bw_a = boxes[:, 2] * sx
        bh_a = boxes[:, 3] * sy

        x1 = np.clip(cx_a - bw_a / 2, 0, w)
        y1 = np.clip(cy_a - bh_a / 2, 0, h)
        x2 = np.clip(cx_a + bw_a / 2, 0, w)
        y2 = np.clip(cy_a + bh_a / 2, 0, h)

        detections: list[Detection] = []
        for i in range(len(class_ids)):
            if x2[i] <= x1[i] or y2[i] <= y1[i]:
                continue
            label = (
                self.labels[int(class_ids[i])]
                if int(class_ids[i]) < len(self.labels)
                else "unknown"
            )
            detections.append(
                Detection(
                    bbox=BBox(
                        x1=float(x1[i]),
                        y1=float(y1[i]),
                        x2=float(x2[i]),
                        y2=float(y2[i]),
                    ),
                    label=label,
                    confidence=round(float(confidences[i]), 4),
                    frame_index=frame_index,
                )
            )
        return detections


# Backward-compatible alias so any stale import still resolves
TFLiteClassifier = YOLOTFLiteDetector
