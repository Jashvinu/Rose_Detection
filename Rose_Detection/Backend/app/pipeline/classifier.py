"""TFLite classifier wrapper extracted from main.py."""

from pathlib import Path

import numpy as np
import tflite_runtime.interpreter as tflite


class TFLiteClassifier:
    """Wraps a TFLite image-classification model."""

    def __init__(self, model_path: Path, labels_path: Path, tile_size: int = 224):
        self.tile_size = tile_size
        self.labels = labels_path.read_text().strip().splitlines()

        self._interpreter = tflite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def classify_tile(self, tile: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Classify a single (224, 224, 3) float32 tile.

        Returns (label, confidence, {label: prob}).
        """
        if tile.ndim == 3:
            tile = tile[np.newaxis, ...]  # (1, H, W, 3)

        self._interpreter.set_tensor(self._input_details[0]["index"], tile.astype(np.float32))
        self._interpreter.invoke()
        probs = self._interpreter.get_tensor(self._output_details[0]["index"])[0]

        idx = int(np.argmax(probs))
        label = self.labels[idx]
        confidence = float(probs[idx])
        prob_dict = {self.labels[i]: round(float(probs[i]), 4) for i in range(len(self.labels))}
        return label, confidence, prob_dict
