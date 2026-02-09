#!/usr/bin/env python3
"""
Unit tests for inference engine.

Tests:
- Model loading validation
- Single image inference
- Class count verification
- Performance benchmarks
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import time

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.inference_engine import InferenceEngine

# Test paths
MODELS_DIR = PROJECT_ROOT / "models"
EXPORTED_DIR = MODELS_DIR / "exported"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"


def find_test_model(mode: str) -> str:
    """Find a model for testing."""
    # Check exported models first
    if EXPORTED_DIR.exists():
        pattern = f"*mode_{mode}*.tflite"
        models = list(EXPORTED_DIR.glob(pattern))
        if models:
            return str(models[0])

    # Check checkpoints
    if CHECKPOINTS_DIR.exists():
        for exp_dir in CHECKPOINTS_DIR.iterdir():
            if f"mode_{mode}" in exp_dir.name.lower():
                best_pt = exp_dir / "weights" / "best.pt"
                if best_pt.exists():
                    return str(best_pt)

    return None


def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a random test image."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


class TestInferenceEngine:
    """Tests for InferenceEngine class."""

    def test_mode_a_class_names(self):
        """Test Mode A has correct class names."""
        expected = ["black_spot", "rust", "downy_mildew", "stippling", "healthy"]
        assert InferenceEngine.DISEASE_CLASSES == expected

    def test_mode_b_class_names(self):
        """Test Mode B has correct class names."""
        expected = ["egg", "larva", "nymph", "adult_female", "adult_male"]
        assert InferenceEngine.MITE_CLASSES == expected

    def test_engine_initialization_mode_a(self):
        """Test engine initializes correctly for Mode A."""
        model_path = find_test_model("a")
        if model_path is None:
            pytest.skip("No Mode A model available for testing")

        engine = InferenceEngine(
            model_path=model_path,
            mode="a",
            confidence_threshold=0.25
        )

        assert engine.mode == "a"
        assert engine.img_size == 416
        assert len(engine.class_names) == 5

    def test_engine_initialization_mode_b(self):
        """Test engine initializes correctly for Mode B."""
        model_path = find_test_model("b")
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        engine = InferenceEngine(
            model_path=model_path,
            mode="b",
            confidence_threshold=0.25
        )

        assert engine.mode == "b"
        assert engine.img_size == 640
        assert len(engine.class_names) == 5

    def test_model_loading_mode_a(self):
        """Test Mode A model loads successfully."""
        model_path = find_test_model("a")
        if model_path is None:
            pytest.skip("No Mode A model available for testing")

        engine = InferenceEngine(model_path=model_path, mode="a")
        result = engine.load_model()

        assert result is True
        if model_path.endswith(".tflite"):
            assert engine.tflite_interpreter is not None
        else:
            assert engine.model is not None

    def test_model_loading_mode_b(self):
        """Test Mode B model loads successfully."""
        model_path = find_test_model("b")
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        engine = InferenceEngine(model_path=model_path, mode="b")
        result = engine.load_model()

        assert result is True

    def test_inference_output_format_mode_a(self):
        """Test Mode A inference returns correct format."""
        model_path = find_test_model("a")
        if model_path is None:
            pytest.skip("No Mode A model available for testing")

        engine = InferenceEngine(model_path=model_path, mode="a")
        engine.load_model()

        test_image = create_test_image(416, 416)
        results = engine.predict(test_image)

        # Check result structure
        assert "detections" in results
        assert "class_counts" in results
        assert "total_count" in results
        assert "image_size" in results

        # Check class counts has all classes
        for cls in engine.class_names:
            assert cls in results["class_counts"]

        # Check total_count matches detections
        assert results["total_count"] == len(results["detections"])

    def test_inference_output_format_mode_b(self):
        """Test Mode B inference returns correct format."""
        model_path = find_test_model("b")
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        engine = InferenceEngine(model_path=model_path, mode="b")
        engine.load_model()

        test_image = create_test_image(640, 640)
        results = engine.predict(test_image)

        # Check result structure
        assert "detections" in results
        assert "class_counts" in results
        assert "total_count" in results
        assert "image_size" in results

        # Check all mite classes present
        for cls in engine.class_names:
            assert cls in results["class_counts"]

    def test_detection_format(self):
        """Test individual detection format."""
        model_path = find_test_model("a")
        if model_path is None:
            pytest.skip("No Mode A model available for testing")

        engine = InferenceEngine(model_path=model_path, mode="a")
        engine.load_model()

        test_image = create_test_image(416, 416)
        results = engine.predict(test_image)

        # If there are detections, verify format
        for det in results["detections"]:
            assert "bbox" in det
            assert "class_id" in det
            assert "class_name" in det
            assert "confidence" in det

            # Bbox should be [x1, y1, x2, y2]
            assert len(det["bbox"]) == 4

            # Confidence should be 0-1
            assert 0 <= det["confidence"] <= 1

            # Class ID should be valid
            assert 0 <= det["class_id"] < len(engine.class_names)

    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        model_path = find_test_model("a")
        if model_path is None:
            pytest.skip("No Mode A model available for testing")

        # High threshold should give fewer detections
        engine_high = InferenceEngine(
            model_path=model_path,
            mode="a",
            confidence_threshold=0.9
        )
        engine_high.load_model()

        engine_low = InferenceEngine(
            model_path=model_path,
            mode="a",
            confidence_threshold=0.1
        )
        engine_low.load_model()

        test_image = create_test_image(416, 416)

        results_high = engine_high.predict(test_image)
        results_low = engine_low.predict(test_image)

        # Low threshold should have >= detections than high
        assert results_low["total_count"] >= results_high["total_count"]

    def test_draw_detections(self):
        """Test detection drawing doesn't crash."""
        model_path = find_test_model("a")
        if model_path is None:
            pytest.skip("No Mode A model available for testing")

        engine = InferenceEngine(model_path=model_path, mode="a")
        engine.load_model()

        test_image = create_test_image(416, 416)
        results = engine.predict(test_image)

        # This should not raise
        annotated = engine.draw_detections(test_image, results["detections"])

        # Output should be same shape as input
        assert annotated.shape == test_image.shape

    def test_summary_stats_mode_a(self):
        """Test summary statistics for Mode A."""
        model_path = find_test_model("a")
        if model_path is None:
            pytest.skip("No Mode A model available for testing")

        engine = InferenceEngine(model_path=model_path, mode="a")
        engine.load_model()

        test_image = create_test_image(416, 416)
        results = engine.predict(test_image)
        stats = engine.get_summary_stats(results)

        assert "total_detections" in stats
        assert "class_counts" in stats
        assert "disease_detected" in stats
        assert "primary_condition" in stats

    def test_summary_stats_mode_b(self):
        """Test summary statistics for Mode B."""
        model_path = find_test_model("b")
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        engine = InferenceEngine(model_path=model_path, mode="b")
        engine.load_model()

        test_image = create_test_image(640, 640)
        results = engine.predict(test_image)
        stats = engine.get_summary_stats(results)

        assert "total_detections" in stats
        assert "adult_count" in stats
        assert "immature_count" in stats
        assert "population_stage" in stats


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_mode_a_inference_speed(self):
        """Test Mode A inference completes within target time."""
        model_path = find_test_model("a")
        if model_path is None:
            pytest.skip("No Mode A model available for testing")

        engine = InferenceEngine(model_path=model_path, mode="a")
        engine.load_model()

        test_image = create_test_image(416, 416)

        # Warmup
        engine.predict(test_image)

        # Timed runs
        times = []
        for _ in range(5):
            start = time.perf_counter()
            engine.predict(test_image)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        mean_time = np.mean(times)
        print(f"\nMode A inference time: {mean_time:.2f}ms")

        # Target: <500ms per image
        assert mean_time < 500, f"Mode A inference too slow: {mean_time:.2f}ms"

    def test_mode_b_inference_speed(self):
        """Test Mode B inference completes within target time (without SAHI)."""
        model_path = find_test_model("b")
        if model_path is None:
            pytest.skip("No Mode B model available for testing")

        engine = InferenceEngine(model_path=model_path, mode="b")
        engine.load_model()

        test_image = create_test_image(640, 640)

        # Warmup
        engine.predict(test_image)

        # Timed runs
        times = []
        for _ in range(5):
            start = time.perf_counter()
            engine.predict(test_image)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        mean_time = np.mean(times)
        print(f"\nMode B inference time (no SAHI): {mean_time:.2f}ms")

        # Target: <1000ms per image (without SAHI overhead)
        assert mean_time < 1000, f"Mode B inference too slow: {mean_time:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
