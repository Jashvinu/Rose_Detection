#!/usr/bin/env python3
"""
Export trained YOLOv8 models to TFLite FP16 format.

Expected output sizes:
- Mode A (YOLOv8-Nano FP16): ~6-8 MB
- Mode B (YOLOv8-Small FP16): ~22-28 MB
"""

import os
import sys
from pathlib import Path
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTED_DIR = MODELS_DIR / "exported"


def find_best_checkpoint(mode: str) -> Path:
    """Find the best checkpoint for a given mode."""
    # Look for mode_a or mode_b directories in checkpoints
    pattern = f"mode_{mode}_*"

    matching_dirs = sorted(
        CHECKPOINTS_DIR.glob(pattern),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    for exp_dir in matching_dirs:
        best_path = exp_dir / "weights" / "best.pt"
        if best_path.exists():
            return best_path

    return None


def export_to_tflite(
    model_path: Path,
    output_name: str,
    imgsz: int,
    half: bool = True,
    int8: bool = False
) -> Path:
    """
    Export YOLOv8 model to TFLite format.

    Args:
        model_path: Path to trained model weights
        output_name: Name for exported model
        imgsz: Image size for export
        half: Use FP16 quantization
        int8: Use INT8 quantization (overrides half)

    Returns:
        Path to exported TFLite model
    """
    from ultralytics import YOLO

    print(f"\nExporting: {model_path}")
    print(f"Output name: {output_name}")
    print(f"Image size: {imgsz}")
    print(f"FP16: {half}, INT8: {int8}")

    # Load model
    model = YOLO(str(model_path))

    # Create output directory
    EXPORTED_DIR.mkdir(parents=True, exist_ok=True)

    # Export to TFLite
    export_path = model.export(
        format="tflite",
        imgsz=imgsz,
        half=half,
        int8=int8,
        simplify=True,
    )

    # Move to exported directory with proper name
    if export_path:
        export_path = Path(export_path)
        if export_path.exists():
            dest_path = EXPORTED_DIR / f"{output_name}.tflite"
            shutil.move(str(export_path), str(dest_path))

            # Get file size
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            print(f"Exported to: {dest_path}")
            print(f"Size: {size_mb:.2f} MB")

            return dest_path

    print("Export failed!")
    return None


def export_mode_a(model_path: Path = None):
    """Export Mode A model to TFLite FP16."""
    print("=" * 60)
    print("Exporting Mode A (Disease Detection) Model")
    print("=" * 60)

    if model_path is None:
        model_path = find_best_checkpoint("a")

    if model_path is None or not model_path.exists():
        print("Error: No Mode A checkpoint found.")
        print(f"Expected location: {CHECKPOINTS_DIR}/mode_a_*/weights/best.pt")
        print("Train the model first using train_mode_a.py")
        return None

    return export_to_tflite(
        model_path=model_path,
        output_name="mode_a_disease_fp16",
        imgsz=416,
        half=True,
        int8=False
    )


def export_mode_b(model_path: Path = None):
    """Export Mode B model to TFLite FP16."""
    print("\n" + "=" * 60)
    print("Exporting Mode B (Mite Counting) Model")
    print("=" * 60)

    if model_path is None:
        model_path = find_best_checkpoint("b")

    if model_path is None or not model_path.exists():
        print("Error: No Mode B checkpoint found.")
        print(f"Expected location: {CHECKPOINTS_DIR}/mode_b_*/weights/best.pt")
        print("Train the model first using train_mode_b.py")
        return None

    return export_to_tflite(
        model_path=model_path,
        output_name="mode_b_mite_fp16",
        imgsz=640,
        half=True,
        int8=False
    )


def main():
    """Main export function with CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Export YOLOv8 models to TFLite")
    parser.add_argument("--mode", choices=["a", "b", "both"], default="both",
                        help="Which mode to export")
    parser.add_argument("--model-a", type=str, default=None,
                        help="Path to Mode A model (optional)")
    parser.add_argument("--model-b", type=str, default=None,
                        help="Path to Mode B model (optional)")
    parser.add_argument("--int8", action="store_true",
                        help="Use INT8 quantization instead of FP16")

    args = parser.parse_args()

    results = {}

    if args.mode in ["a", "both"]:
        model_path = Path(args.model_a) if args.model_a else None
        results["mode_a"] = export_mode_a(model_path)

    if args.mode in ["b", "both"]:
        model_path = Path(args.model_b) if args.model_b else None
        results["mode_b"] = export_mode_b(model_path)

    # Summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)

    for mode, path in results.items():
        if path:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {mode}: {path.name} ({size_mb:.2f} MB)")
        else:
            print(f"  {mode}: FAILED")

    print(f"\nExported models location: {EXPORTED_DIR}")


if __name__ == "__main__":
    main()
