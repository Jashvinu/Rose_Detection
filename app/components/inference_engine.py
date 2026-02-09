"""
Inference Engine for PlantVillage Rose Edition.

Provides unified interface for running both Mode A (disease detection)
and Mode B (mite counting) models.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from PIL import Image
import cv2


class InferenceEngine:
    """
    Unified inference engine for rose disease and mite detection.

    Supports:
    - YOLOv8 PyTorch models
    - TFLite FP16 models
    - Single image and batch inference
    - Video frame processing
    """

    # Mode A: Disease classes
    DISEASE_CLASSES = ["black_spot", "rust", "downy_mildew", "stippling", "healthy"]

    # Mode B: Mite classes
    MITE_CLASSES = ["egg", "larva", "nymph", "adult_female", "adult_male"]

    def __init__(
        self,
        model_path: str,
        mode: str = "a",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to model weights (.pt or .tflite)
            mode: 'a' for disease detection, 'b' for mite counting
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Inference device
        """
        self.model_path = Path(model_path)
        self.mode = mode.lower()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        self.model = None
        self.tflite_interpreter = None
        self.is_tflite = self.model_path.suffix.lower() == ".tflite"

        # Set class names based on mode
        self.class_names = self.DISEASE_CLASSES if self.mode == "a" else self.MITE_CLASSES

        # Image size based on mode
        self.img_size = 416 if self.mode == "a" else 640

    def load_model(self) -> bool:
        """Load the model."""
        try:
            if self.is_tflite:
                return self._load_tflite()
            else:
                return self._load_yolo()
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def _load_yolo(self) -> bool:
        """Load YOLOv8 model."""
        from ultralytics import YOLO
        self.model = YOLO(str(self.model_path))
        return True

    def _load_tflite(self) -> bool:
        """Load TFLite model."""
        import tensorflow as tf

        self.tflite_interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
        self.tflite_interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()

        return True

    def predict(self, image: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image: Image as numpy array (RGB), or path to image file

        Returns:
            Dictionary with detections and statistics
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = self._load_image(str(image))

        if self.is_tflite:
            return self._predict_tflite(image)
        else:
            return self._predict_yolo(image)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from path."""
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def _predict_yolo(self, image: np.ndarray) -> Dict[str, Any]:
        """Run YOLO inference."""
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )

        detections = []
        class_counts = {name: 0 for name in self.class_names}

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

                detections.append({
                    "bbox": bbox,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf
                })

                if cls_name in class_counts:
                    class_counts[cls_name] += 1

        return {
            "detections": detections,
            "class_counts": class_counts,
            "total_count": len(detections),
            "image_size": (image.shape[1], image.shape[0])
        }

    def _predict_tflite(self, image: np.ndarray) -> Dict[str, Any]:
        """Run TFLite inference."""
        # Preprocess image
        input_shape = self.input_details[0]["shape"]
        h, w = input_shape[1], input_shape[2]

        # Resize and normalize
        img_resized = cv2.resize(image, (w, h))
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Add batch dimension if needed
        if len(input_shape) == 4:
            img_input = np.expand_dims(img_normalized, axis=0)
        else:
            img_input = img_normalized

        # Run inference
        self.tflite_interpreter.set_tensor(self.input_details[0]["index"], img_input)
        self.tflite_interpreter.invoke()

        # Get output
        output = self.tflite_interpreter.get_tensor(self.output_details[0]["index"])

        # Parse YOLOv8 TFLite output
        detections = self._parse_yolo_output(output, image.shape[:2])

        # Count by class
        class_counts = {name: 0 for name in self.class_names}
        for det in detections:
            cls_name = det["class_name"]
            if cls_name in class_counts:
                class_counts[cls_name] += 1

        return {
            "detections": detections,
            "class_counts": class_counts,
            "total_count": len(detections),
            "image_size": (image.shape[1], image.shape[0])
        }

    def _parse_yolo_output(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int]
    ) -> List[Dict]:
        """Parse YOLOv8 TFLite output format."""
        detections = []
        orig_h, orig_w = original_size

        # YOLOv8 output shape: (1, num_classes + 4, num_detections)
        # Transpose to (num_detections, num_classes + 4)
        if output.ndim == 3:
            output = output[0].T

        num_classes = len(self.class_names)

        for detection in output:
            # First 4 values are x_center, y_center, width, height (normalized)
            x_center, y_center, width, height = detection[:4]

            # Remaining values are class probabilities
            class_scores = detection[4:4 + num_classes]

            # Get best class
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.confidence_threshold:
                continue

            # Convert to xyxy format and scale to original size
            x1 = (x_center - width / 2) * orig_w
            y1 = (y_center - height / 2) * orig_h
            x2 = (x_center + width / 2) * orig_w
            y2 = (y_center + height / 2) * orig_h

            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence
            })

        # Apply NMS
        return self._apply_nms(detections)

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression."""
        if len(detections) == 0:
            return []

        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array([d["confidence"] for d in detections])

        # Use OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.iou_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]

        return []

    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        frame_skip: int = 1,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Process video file frame by frame.

        Args:
            video_path: Path to input video
            output_path: Path for annotated output video (optional)
            frame_skip: Process every Nth frame
            progress_callback: Function to call with progress updates

        Returns:
            Dictionary with frame-by-frame results and summary
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_results = []
        total_counts = {name: 0 for name in self.class_names}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run inference
                result = self.predict(frame_rgb)

                frame_results.append({
                    "frame": frame_idx,
                    "detections": result["detections"],
                    "class_counts": result["class_counts"]
                })

                # Accumulate counts
                for cls, count in result["class_counts"].items():
                    total_counts[cls] += count

                # Draw annotations for output video
                if writer:
                    annotated = self.draw_detections(frame_rgb, result["detections"])
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    writer.write(annotated_bgr)

            elif writer:
                # Write original frame for skipped frames
                writer.write(frame)

            frame_idx += 1

            if progress_callback:
                progress_callback(frame_idx / total_frames)

        cap.release()
        if writer:
            writer.release()

        return {
            "frame_results": frame_results,
            "total_counts": total_counts,
            "total_frames": total_frames,
            "processed_frames": len(frame_results),
            "fps": fps,
            "video_size": (width, height)
        }

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        draw_labels: bool = True
    ) -> np.ndarray:
        """Draw detection boxes on image."""
        output = image.copy()

        # Color maps
        if self.mode == "a":
            colors = {
                "black_spot": (0, 0, 0),         # Black
                "rust": (255, 140, 0),           # Orange
                "downy_mildew": (128, 128, 128), # Gray
                "stippling": (255, 255, 0),      # Yellow
                "healthy": (0, 255, 0),          # Green
            }
        else:
            colors = {
                "egg": (255, 255, 0),           # Yellow
                "larva": (0, 255, 255),         # Cyan
                "nymph": (0, 255, 0),           # Green
                "adult_female": (255, 0, 255),  # Magenta
                "adult_male": (0, 0, 255),      # Red
            }

        for det in detections:
            bbox = det["bbox"]
            class_name = det.get("class_name", f"class_{det['class_id']}")
            confidence = det["confidence"]

            x1, y1, x2, y2 = map(int, bbox)
            color = colors.get(class_name, (128, 128, 128))

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if draw_labels:
                label = f"{class_name}: {confidence:.2f}"
                font_scale = 0.5
                thickness = 1

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                # Background for text
                cv2.rectangle(output, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
                cv2.putText(output, label, (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        return output

    def get_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        detections = results.get("detections", [])
        class_counts = results.get("class_counts", {})

        confidences = [d["confidence"] for d in detections]

        stats = {
            "total_detections": len(detections),
            "class_counts": class_counts,
        }

        if confidences:
            stats.update({
                "mean_confidence": float(np.mean(confidences)),
                "min_confidence": float(np.min(confidences)),
                "max_confidence": float(np.max(confidences))
            })

        # Mode-specific stats
        if self.mode == "a":
            # Disease prevalence
            non_healthy = sum(v for k, v in class_counts.items() if k != "healthy")
            stats["disease_detected"] = non_healthy > 0
            stats["primary_condition"] = max(class_counts, key=class_counts.get) if detections else "none"
        else:
            # Mite population stats
            adults = class_counts.get("adult_female", 0) + class_counts.get("adult_male", 0)
            immature = class_counts.get("egg", 0) + class_counts.get("larva", 0) + class_counts.get("nymph", 0)
            stats["adult_count"] = adults
            stats["immature_count"] = immature
            stats["population_stage"] = "growing" if immature > adults else "mature"

        return stats
