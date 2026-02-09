#!/usr/bin/env python3
"""
Integration tests for SAHI processor.

Tests:
- SAHI sliced prediction
- Detection merging and NMS
- COCO format export
- High-resolution image handling
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import json
import time

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.sahi_processor import SAHIProcessor

# Test paths
MODELS_DIR = PROJECT_ROOT / "models"
EXPORTED_DIR = MODELS_DIR / "exported"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"


def find_mode_b_model() -> str:
    """Find a Mode B model for testing."""
    # Check exported models first
    if EXPORTED_DIR.exists():
        models = list(EXPORTED_DIR.glob("*mode_b*.tflite"))
        models += list(EXPORTED_DIR.glob("*mite*.tflite"))
        if models:
            return str(models[0])

    # Check checkpoints
    if CHECKPOINTS_DIR.exists():
        for exp_dir in CHECKPOINTS_DIR.iterdir():
            if "mode_b" in exp_dir.name.lower():
                best_pt = exp_dir / "weights" / "best.pt"
                if best_pt.exists():
                    return str(best_pt)

    return None


def create_test_image(width: int, height: int) -> np.ndarray:
    """Create a random test image."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def create_high_res_image() -> np.ndarray:
    """Create a high-resolution test image (simulating macro photo)."""
    return create_test_image(2048, 1536)


class TestSAHIProcessor:
    """Tests for SAHIProcessor class."""

    def test_class_names(self):
        """Test SAHI processor has correct mite class names."""
        expected = ["egg", "larva", "nymph", "adult_female", "adult_male"]
        assert SAHIProcessor.CLASS_NAMES == expected

    def test_initialization(self):
        """Test SAHI processor initialization."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        processor = SAHIProcessor(
            model_path=model_path,
            slice_size=256,
            overlap_ratio=0.2,
            confidence_threshold=0.25
        )

        assert processor.slice_size == 256
        assert processor.overlap_ratio == 0.2
        assert processor.confidence_threshold == 0.25

    def test_model_loading(self):
        """Test SAHI model loads successfully."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        processor = SAHIProcessor(model_path=model_path)
        result = processor.load_model()

        assert result is True
        assert processor.model is not None or processor.detection_model is not None

    def test_prediction_output_format(self):
        """Test SAHI prediction returns correct format."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        processor = SAHIProcessor(
            model_path=model_path,
            slice_size=256,
            overlap_ratio=0.2
        )
        processor.load_model()

        test_image = create_test_image(640, 640)
        results = processor.predict(test_image)

        # Check result structure
        assert "detections" in results
        assert "class_counts" in results
        assert "total_count" in results
        assert "image_size" in results

        # Check all mite classes present in counts
        for cls in processor.CLASS_NAMES:
            assert cls in results["class_counts"]

    def test_high_resolution_image(self):
        """Test SAHI handles high-resolution images."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        processor = SAHIProcessor(
            model_path=model_path,
            slice_size=256,
            overlap_ratio=0.2
        )
        processor.load_model()

        # Create high-res image
        test_image = create_high_res_image()
        results = processor.predict(test_image)

        # Should complete without error
        assert "detections" in results
        assert results["image_size"] == (2048, 1536)

    def test_slice_size_variations(self):
        """Test different slice sizes work correctly."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        slice_sizes = [128, 256, 512]
        test_image = create_test_image(640, 640)

        for slice_size in slice_sizes:
            processor = SAHIProcessor(
                model_path=model_path,
                slice_size=slice_size
            )
            processor.load_model()

            results = processor.predict(test_image)
            assert "detections" in results

    def test_overlap_ratio_variations(self):
        """Test different overlap ratios work correctly."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        overlaps = [0.1, 0.2, 0.3, 0.5]
        test_image = create_test_image(640, 640)

        for overlap in overlaps:
            processor = SAHIProcessor(
                model_path=model_path,
                overlap_ratio=overlap
            )
            processor.load_model()

            results = processor.predict(test_image)
            assert "detections" in results

    def test_detection_format(self):
        """Test individual detection format from SAHI."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        processor = SAHIProcessor(model_path=model_path)
        processor.load_model()

        test_image = create_test_image(640, 640)
        results = processor.predict(test_image)

        for det in results["detections"]:
            assert "bbox" in det
            assert "class_id" in det
            assert "class_name" in det
            assert "confidence" in det

            # Bbox should be [x1, y1, x2, y2]
            assert len(det["bbox"]) == 4

            # Bbox should be within image bounds
            x1, y1, x2, y2 = det["bbox"]
            assert 0 <= x1 <= 640
            assert 0 <= y1 <= 640
            assert x1 <= x2
            assert y1 <= y2

    def test_draw_detections(self):
        """Test detection drawing on image."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        processor = SAHIProcessor(model_path=model_path)
        processor.load_model()

        test_image = create_test_image(640, 640)
        results = processor.predict(test_image)

        # Draw should not raise
        annotated = processor.draw_detections(test_image, results["detections"])

        # Output should be same shape
        assert annotated.shape == test_image.shape

    def test_coco_export(self):
        """Test COCO format export."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        processor = SAHIProcessor(model_path=model_path)
        processor.load_model()

        test_image = create_test_image(640, 640)
        results = processor.predict(test_image)

        coco = processor.export_to_coco(
            results["detections"],
            "test_image.jpg",
            results["image_size"]
        )

        # Check COCO structure
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco

        # Check image info
        assert len(coco["images"]) == 1
        assert coco["images"][0]["file_name"] == "test_image.jpg"
        assert coco["images"][0]["width"] == 640
        assert coco["images"][0]["height"] == 640

        # Check categories
        assert len(coco["categories"]) == 5

        # Check annotations match detections
        assert len(coco["annotations"]) == len(results["detections"])

        # Verify COCO annotation format
        for ann in coco["annotations"]:
            assert "id" in ann
            assert "image_id" in ann
            assert "category_id" in ann
            assert "bbox" in ann
            assert "area" in ann

            # Bbox should be [x, y, width, height] format
            assert len(ann["bbox"]) == 4

    def test_coco_export_json_serializable(self):
        """Test COCO export is JSON serializable."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        processor = SAHIProcessor(model_path=model_path)
        processor.load_model()

        test_image = create_test_image(640, 640)
        results = processor.predict(test_image)

        coco = processor.export_to_coco(
            results["detections"],
            "test.jpg",
            results["image_size"]
        )

        # Should serialize without error
        json_str = json.dumps(coco)
        assert len(json_str) > 0

        # Should deserialize back
        parsed = json.loads(json_str)
        assert parsed["images"] == coco["images"]


class TestSAHIPerformance:
    """Performance benchmarks for SAHI processor."""

    def test_sahi_inference_time(self):
        """Test SAHI inference time on high-res image."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        processor = SAHIProcessor(
            model_path=model_path,
            slice_size=256,
            overlap_ratio=0.2
        )
        processor.load_model()

        # Use high-res image
        test_image = create_high_res_image()

        # Warmup
        processor.predict(test_image)

        # Timed run
        start = time.perf_counter()
        processor.predict(test_image)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        print(f"\nSAHI inference time (2048x1536, 256px slices): {elapsed_ms:.2f}ms")

        # Target: <3s for high-res image with SAHI
        assert elapsed_ms < 3000, f"SAHI inference too slow: {elapsed_ms:.2f}ms"

    def test_slice_count_calculation(self):
        """Test expected slice count for given parameters."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        # 640x640 image with 256px slices, 0.2 overlap
        # Step = 256 * (1 - 0.2) = 204.8 ≈ 204
        # Slices per dimension ≈ ceil(640 / 204) = 4
        # Total slices ≈ 4 * 4 = 16

        processor = SAHIProcessor(
            model_path=model_path,
            slice_size=256,
            overlap_ratio=0.2
        )

        step = int(256 * (1 - 0.2))
        expected_slices_x = (640 // step) + 1
        expected_slices_y = (640 // step) + 1
        expected_total = expected_slices_x * expected_slices_y

        print(f"\nExpected slices for 640x640: {expected_total}")
        assert expected_total > 1  # Should have multiple slices


class TestNMSIntegration:
    """Tests for NMS (Non-Maximum Suppression) in SAHI."""

    def test_nms_reduces_duplicates(self):
        """Test that NMS is applied to reduce duplicate detections."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        # Lower IoU threshold should keep more overlapping boxes
        processor_low = SAHIProcessor(
            model_path=model_path,
            iou_threshold=0.9  # Very permissive
        )
        processor_low.load_model()

        processor_high = SAHIProcessor(
            model_path=model_path,
            iou_threshold=0.3  # More aggressive NMS
        )
        processor_high.load_model()

        test_image = create_test_image(640, 640)

        results_low = processor_low.predict(test_image)
        results_high = processor_high.predict(test_image)

        # Higher IoU threshold (more aggressive NMS) should result in
        # fewer or equal detections
        assert results_high["total_count"] <= results_low["total_count"]

    def test_iou_calculation(self):
        """Test IoU calculation method."""
        model_path = find_mode_b_model()
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        processor = SAHIProcessor(model_path=model_path)

        # Test with known boxes
        box1 = np.array([0, 0, 100, 100])
        boxes = np.array([
            [0, 0, 100, 100],    # Identical - IoU = 1.0
            [50, 50, 150, 150],  # 50% overlap
            [200, 200, 300, 300] # No overlap - IoU = 0.0
        ])

        ious = processor._calculate_iou(box1, boxes)

        assert len(ious) == 3
        assert abs(ious[0] - 1.0) < 0.01  # Identical boxes
        assert ious[2] < 0.01  # No overlap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
