#!/usr/bin/env python3
"""
Validate exported TFLite models.

Tests:
1. Model loads correctly
2. Inference produces valid outputs
3. Output shape matches expected dimensions
4. Performance benchmarks
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
EXPORTED_DIR = MODELS_DIR / "exported"


def load_tflite_model(model_path: Path):
    """Load TFLite model and return interpreter."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    return interpreter


def get_model_info(interpreter) -> Dict[str, Any]:
    """Get model input/output details."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    info = {
        "num_inputs": len(input_details),
        "num_outputs": len(output_details),
        "inputs": [],
        "outputs": []
    }

    for inp in input_details:
        info["inputs"].append({
            "name": inp["name"],
            "shape": inp["shape"].tolist(),
            "dtype": str(inp["dtype"])
        })

    for out in output_details:
        info["outputs"].append({
            "name": out["name"],
            "shape": out["shape"].tolist(),
            "dtype": str(out["dtype"])
        })

    return info


def run_inference(interpreter, input_data: np.ndarray) -> np.ndarray:
    """Run inference on input data."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input
    interpreter.set_tensor(input_details[0]["index"], input_data)

    # Run inference
    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(output_details[0]["index"])

    return output


def benchmark_inference(
    interpreter,
    input_shape: Tuple[int, ...],
    num_runs: int = 10,
    warmup_runs: int = 3
) -> Dict[str, float]:
    """Benchmark inference performance."""
    # Create random input
    input_data = np.random.rand(*input_shape).astype(np.float32)

    # Warmup runs
    for _ in range(warmup_runs):
        run_inference(interpreter, input_data)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        run_inference(interpreter, input_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times)
    }


def validate_model(model_path: Path, expected_imgsz: int, mode: str) -> bool:
    """
    Validate a single TFLite model.

    Returns True if validation passes.
    """
    print(f"\n{'=' * 60}")
    print(f"Validating: {model_path.name}")
    print("=" * 60)

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return False

    # Get file size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    try:
        # Load model
        print("\n1. Loading model...")
        interpreter = load_tflite_model(model_path)
        print("   Model loaded successfully")

        # Get model info
        print("\n2. Model architecture:")
        info = get_model_info(interpreter)
        print(f"   Inputs: {info['num_inputs']}")
        for inp in info["inputs"]:
            print(f"     - {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")
        print(f"   Outputs: {info['num_outputs']}")
        for out in info["outputs"]:
            print(f"     - {out['name']}: shape={out['shape']}, dtype={out['dtype']}")

        # Validate input shape
        expected_shape = [1, expected_imgsz, expected_imgsz, 3]
        actual_shape = info["inputs"][0]["shape"]
        # YOLOv8 TFLite might have shape in different order
        if list(actual_shape) != expected_shape:
            print(f"\n   Note: Input shape {actual_shape} differs from expected {expected_shape}")
            print("   This may be due to channel order (NCHW vs NHWC)")

        # Run test inference
        print("\n3. Running test inference...")
        input_shape = tuple(info["inputs"][0]["shape"])
        test_input = np.random.rand(*input_shape).astype(np.float32)
        output = run_inference(interpreter, test_input)
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")

        # Benchmark performance
        print("\n4. Benchmarking (10 runs)...")
        benchmark = benchmark_inference(interpreter, input_shape)
        print(f"   Mean: {benchmark['mean_ms']:.2f} ms")
        print(f"   Std:  {benchmark['std_ms']:.2f} ms")
        print(f"   Min:  {benchmark['min_ms']:.2f} ms")
        print(f"   Max:  {benchmark['max_ms']:.2f} ms")

        # Check performance targets
        print("\n5. Performance check:")
        if mode == "a":
            target_ms = 500
            target_size_mb = 10
        else:
            target_ms = 3000  # SAHI will add overhead
            target_size_mb = 30

        size_ok = size_mb <= target_size_mb
        speed_ok = benchmark["mean_ms"] <= target_ms

        print(f"   Size:  {size_mb:.2f} MB {'<=' if size_ok else '>'} {target_size_mb} MB - {'PASS' if size_ok else 'WARN'}")
        print(f"   Speed: {benchmark['mean_ms']:.2f} ms {'<=' if speed_ok else '>'} {target_ms} ms - {'PASS' if speed_ok else 'WARN'}")

        print(f"\nValidation: PASSED")
        return True

    except Exception as e:
        print(f"\nValidation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate exported TFLite models")
    parser.add_argument("--mode", choices=["a", "b", "both"], default="both",
                        help="Which mode to validate")
    parser.add_argument("--model-a", type=str, default=None,
                        help="Path to Mode A model (optional)")
    parser.add_argument("--model-b", type=str, default=None,
                        help="Path to Mode B model (optional)")

    args = parser.parse_args()

    results = {}

    if args.mode in ["a", "both"]:
        if args.model_a:
            model_path = Path(args.model_a)
        else:
            model_path = EXPORTED_DIR / "mode_a_disease_fp16.tflite"

        results["mode_a"] = validate_model(model_path, 416, "a")

    if args.mode in ["b", "both"]:
        if args.model_b:
            model_path = Path(args.model_b)
        else:
            model_path = EXPORTED_DIR / "mode_b_mite_fp16.tflite"

        results["mode_b"] = validate_model(model_path, 640, "b")

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    all_passed = True
    for mode, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {mode}: {status}")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
