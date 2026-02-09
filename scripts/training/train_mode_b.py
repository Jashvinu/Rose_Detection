#!/usr/bin/env python3
"""
Train YOLOv8-Small model for Mode B (TSSM Mite Counting).

CPU-optimized training configuration for small object detection:
- Smaller batch size (2) for larger model
- Higher image resolution (640) for small objects
- Extended patience for longer convergence
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
PRETRAINED_DIR = MODELS_DIR / "pretrained"


def download_pretrained_model():
    """Download YOLOv8-Small pretrained weights if not present."""
    from ultralytics import YOLO

    pretrained_path = PRETRAINED_DIR / "yolov8s.pt"
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    if not pretrained_path.exists():
        print("Downloading YOLOv8-Small pretrained weights...")
        model = YOLO("yolov8s.pt")
        # Save to our pretrained directory
        import shutil
        default_path = Path.home() / ".cache" / "ultralytics" / "yolov8s.pt"
        if default_path.exists():
            shutil.copy2(default_path, pretrained_path)
        print(f"Saved to: {pretrained_path}")
    else:
        print(f"Using cached pretrained model: {pretrained_path}")

    return pretrained_path


def train_mode_b(
    epochs: int = 150,
    batch_size: int = 2,
    img_size: int = 640,
    patience: int = 30,
    resume: bool = False,
    resume_path: str = None
):
    """
    Train YOLOv8-Small for TSSM mite counting.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size (smaller for larger model)
        img_size: Image size (larger for small object detection)
        patience: Early stopping patience
        resume: Resume from last checkpoint
        resume_path: Path to checkpoint to resume from
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("Mode B Training: TSSM Mite Counting")
    print("=" * 60)
    print(f"Model: YOLOv8-Small")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: CPU")
    print("=" * 60)

    # Setup paths
    data_config = CONFIGS_DIR / "mode_b_mite.yaml"
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if data config exists
    if not data_config.exists():
        print(f"Error: Dataset config not found: {data_config}")
        print("Run the data pipeline scripts first.")
        return None

    # Load model
    if resume and resume_path:
        print(f"Resuming from: {resume_path}")
        model = YOLO(resume_path)
    else:
        pretrained_path = download_pretrained_model()
        model = YOLO(str(pretrained_path))

    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"mode_b_mite_{timestamp}"

    # Training configuration optimized for CPU and small objects
    train_args = {
        "data": str(data_config),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "device": "cpu",
        "workers": 2,
        "patience": patience,
        "optimizer": "AdamW",
        "lr0": 0.0005,  # Lower LR for larger model
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,  # Longer warmup
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "project": str(CHECKPOINTS_DIR),
        "name": experiment_name,
        "exist_ok": True,
        "pretrained": True,
        "verbose": True,
        "save": True,
        "save_period": 10,
        "val": True,
        "plots": True,
        # Data augmentation - tuned for small objects
        "hsv_h": 0.01,  # Less color variation
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 15,  # More rotation for mites
        "translate": 0.15,
        "scale": 0.4,
        "shear": 5,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 0.8,  # Slightly less mosaic
        "mixup": 0.05,
        # Small object specific
        "copy_paste": 0.1,  # Copy-paste augmentation for small objects
        "close_mosaic": 10,  # Disable mosaic for last 10 epochs
    }

    print("\nStarting training...")
    print("WARNING: Mode B training on CPU will be very slow.")
    print("Strongly recommend using Colab GPU fallback.")
    print("-" * 60)

    try:
        results = model.train(**train_args)

        # Print final results
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        best_model_path = CHECKPOINTS_DIR / experiment_name / "weights" / "best.pt"
        last_model_path = CHECKPOINTS_DIR / experiment_name / "weights" / "last.pt"

        print(f"Best model: {best_model_path}")
        print(f"Last model: {last_model_path}")

        # Validation metrics
        if hasattr(results, "results_dict"):
            print("\nValidation Metrics:")
            for key, value in results.results_dict.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")

        return best_model_path

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return None
    except Exception as e:
        print(f"\nTraining error: {e}")
        raise


def validate_model(model_path: str, data_config: str = None):
    """Validate trained model on test set."""
    from ultralytics import YOLO

    if data_config is None:
        data_config = str(CONFIGS_DIR / "mode_b_mite.yaml")

    print(f"\nValidating model: {model_path}")

    model = YOLO(model_path)
    results = model.val(data=data_config, split="test", device="cpu")

    print("\nTest Set Metrics:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")

    # Per-class metrics
    print("\nPer-Class AP50:")
    class_names = ["egg", "larva", "nymph", "adult_female", "adult_male"]
    for i, ap in enumerate(results.box.ap50):
        print(f"  {class_names[i]}: {ap:.4f}")

    return results


def main():
    """Main training function with CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Mode B Mite Counting Model")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint")
    parser.add_argument("--validate-only", type=str, default=None, help="Only validate this model")

    args = parser.parse_args()

    if args.validate_only:
        validate_model(args.validate_only)
    else:
        train_mode_b(
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            patience=args.patience,
            resume=args.resume is not None,
            resume_path=args.resume
        )


if __name__ == "__main__":
    main()
