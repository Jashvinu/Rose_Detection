"""
SAHI (Slicing Aided Hyper Inference) Processor for Mode B.

Handles sliced inference on high-resolution macro photos for
detecting small objects (TSSM mites).
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from PIL import Image


class SAHIProcessor:
    """
    SAHI-based processor for detecting small objects in high-resolution images.

    Uses sliding window approach to detect tiny mites that would be missed
    in full-image inference.
    """

    # TSSM life stage class names
    CLASS_NAMES = ["egg", "larva", "nymph", "adult_female", "adult_male"]

    def __init__(
        self,
        model_path: str,
        slice_size: int = 256,
        overlap_ratio: float = 0.2,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize SAHI processor.

        Args:
            model_path: Path to YOLO model weights
            slice_size: Size of each slice in pixels
            overlap_ratio: Overlap between slices (0.0 to 0.5)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Inference device (cpu/cuda)
        """
        self.model_path = model_path
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        self.model = None
        self.detection_model = None

    def load_model(self):
        """Load YOLO model for SAHI inference."""
        try:
            from sahi import AutoDetectionModel

            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=self.model_path,
                confidence_threshold=self.confidence_threshold,
                device=self.device
            )
            return True

        except ImportError:
            print("SAHI not installed. Using fallback method.")
            return self._load_fallback_model()

    def _load_fallback_model(self) -> bool:
        """Load model without SAHI for basic sliced inference."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run SAHI sliced inference on an image.

        Args:
            image: Input image as numpy array (RGB)

        Returns:
            Dictionary with detections and statistics
        """
        if self.detection_model is not None:
            return self._predict_sahi(image)
        elif self.model is not None:
            return self._predict_fallback(image)
        else:
            raise RuntimeError("Model not loaded. Call load_model() first.")

    def _predict_sahi(self, image: np.ndarray) -> Dict[str, Any]:
        """Run prediction using SAHI library."""
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            image=image,
            detection_model=self.detection_model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=self.iou_threshold,
            verbose=0
        )

        # Parse SAHI result
        detections = []
        class_counts = {name: 0 for name in self.CLASS_NAMES}

        for obj in result.object_prediction_list:
            bbox = obj.bbox.to_xyxy()
            class_id = obj.category.id
            class_name = self.CLASS_NAMES[class_id] if class_id < len(self.CLASS_NAMES) else f"class_{class_id}"
            confidence = obj.score.value

            detections.append({
                "bbox": bbox,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence
            })

            if class_name in class_counts:
                class_counts[class_name] += 1

        return {
            "detections": detections,
            "class_counts": class_counts,
            "total_count": len(detections),
            "image_size": (image.shape[1], image.shape[0])
        }

    def _predict_fallback(self, image: np.ndarray) -> Dict[str, Any]:
        """Run sliced prediction without SAHI library."""
        height, width = image.shape[:2]

        # Calculate slice positions
        step = int(self.slice_size * (1 - self.overlap_ratio))
        slices = []

        for y in range(0, height, step):
            for x in range(0, width, step):
                x2 = min(x + self.slice_size, width)
                y2 = min(y + self.slice_size, height)
                x1 = max(0, x2 - self.slice_size)
                y1 = max(0, y2 - self.slice_size)
                slices.append((x1, y1, x2, y2))

        # Run inference on each slice
        all_detections = []

        for x1, y1, x2, y2 in slices:
            slice_img = image[y1:y2, x1:x2]

            # Run YOLO inference
            results = self.model(
                slice_img,
                conf=self.confidence_threshold,
                verbose=False
            )

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    # Get box in slice coordinates
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())

                    # Convert to full image coordinates
                    full_box = [
                        box[0] + x1,
                        box[1] + y1,
                        box[2] + x1,
                        box[3] + y1
                    ]

                    all_detections.append({
                        "bbox": full_box,
                        "class_id": cls,
                        "confidence": float(conf)
                    })

        # Apply NMS to merged detections
        final_detections = self._apply_nms(all_detections)

        # Count by class
        class_counts = {name: 0 for name in self.CLASS_NAMES}
        for det in final_detections:
            class_id = det["class_id"]
            class_name = self.CLASS_NAMES[class_id] if class_id < len(self.CLASS_NAMES) else f"class_{class_id}"
            det["class_name"] = class_name
            if class_name in class_counts:
                class_counts[class_name] += 1

        return {
            "detections": final_detections,
            "class_counts": class_counts,
            "total_count": len(final_detections),
            "image_size": (width, height)
        }

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to merged detections."""
        if len(detections) == 0:
            return []

        # Group by class
        class_detections = {}
        for det in detections:
            cls = det["class_id"]
            if cls not in class_detections:
                class_detections[cls] = []
            class_detections[cls].append(det)

        # Apply NMS per class
        final_detections = []

        for cls, dets in class_detections.items():
            boxes = np.array([d["bbox"] for d in dets])
            scores = np.array([d["confidence"] for d in dets])

            # Simple NMS implementation
            keep = []
            order = scores.argsort()[::-1]

            while len(order) > 0:
                i = order[0]
                keep.append(i)

                if len(order) == 1:
                    break

                # Calculate IoU with remaining boxes
                remaining = order[1:]
                ious = self._calculate_iou(boxes[i], boxes[remaining])

                # Keep boxes with IoU below threshold
                mask = ious < self.iou_threshold
                order = remaining[mask]

            final_detections.extend([dets[i] for i in keep])

        return final_detections

    def _calculate_iou(self, box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU between one box and array of boxes."""
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        areas2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = area1 + areas2 - intersection

        return intersection / (union + 1e-6)

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        draw_labels: bool = True
    ) -> np.ndarray:
        """Draw detection boxes on image."""
        import cv2

        output = image.copy()

        # Color map for classes
        colors = {
            "egg": (255, 255, 0),       # Yellow
            "larva": (0, 255, 255),     # Cyan
            "nymph": (0, 255, 0),       # Green
            "adult_female": (255, 0, 255),  # Magenta
            "adult_male": (0, 0, 255),  # Red
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
                cv2.rectangle(output, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
                cv2.putText(output, label, (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        return output

    def export_to_coco(
        self,
        detections: List[Dict],
        image_path: str,
        image_size: Tuple[int, int]
    ) -> Dict:
        """Export detections to COCO format."""
        width, height = image_size

        coco_annotations = []
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1

            coco_annotations.append({
                "id": i + 1,
                "image_id": 1,
                "category_id": det["class_id"] + 1,  # COCO uses 1-indexed
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "score": det["confidence"]
            })

        return {
            "images": [{
                "id": 1,
                "file_name": str(Path(image_path).name),
                "width": width,
                "height": height
            }],
            "annotations": coco_annotations,
            "categories": [
                {"id": i + 1, "name": name}
                for i, name in enumerate(self.CLASS_NAMES)
            ]
        }
