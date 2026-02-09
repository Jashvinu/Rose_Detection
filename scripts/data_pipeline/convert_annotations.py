#!/usr/bin/env python3
"""
Convert classification datasets to YOLO detection format.

This script handles:
1. Converting folder-based classification datasets to YOLO format
2. Generating bounding boxes for full-image annotations
3. Mapping class names to standardized labels
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Mode A: Disease classification mapping
DISEASE_CLASS_MAPPING = {
    # Black Spot variants
    "black_spot": 0,
    "blackspot": 0,
    "black spot": 0,
    "diplocarpon_rosae": 0,
    # Rust variants
    "rust": 1,
    "rose_rust": 1,
    "phragmidium": 1,
    # Downy Mildew variants
    "downy_mildew": 2,
    "downy mildew": 2,
    "peronospora": 2,
    # Stippling (spider mite damage) variants
    "stippling": 3,
    "spider_mite_damage": 3,
    "mite_damage": 3,
    # Healthy variants
    "healthy": 4,
    "fresh_leaf": 4,
    "normal": 4,
}

DISEASE_CLASSES = ["black_spot", "rust", "downy_mildew", "stippling", "healthy"]

# Mode B: TSSM life stage mapping
MITE_CLASS_MAPPING = {
    "egg": 0,
    "eggs": 0,
    "larva": 1,
    "larvae": 1,
    "nymph": 2,
    "nymphs": 2,
    "adult_female": 3,
    "female": 3,
    "adult_male": 4,
    "male": 4,
    "adult": 3,  # Default adults to female (more common)
}

MITE_CLASSES = ["egg", "larva", "nymph", "adult_female", "adult_male"]


def normalize_class_name(name: str) -> str:
    """Normalize class name for matching."""
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def get_disease_class_id(folder_name: str) -> Optional[int]:
    """Get disease class ID from folder name."""
    normalized = normalize_class_name(folder_name)
    return DISEASE_CLASS_MAPPING.get(normalized)


def get_mite_class_id(label: str) -> Optional[int]:
    """Get mite class ID from label."""
    normalized = normalize_class_name(label)
    return MITE_CLASS_MAPPING.get(normalized)


def create_full_image_bbox(img_width: int, img_height: int, margin: float = 0.05) -> str:
    """
    Create YOLO format bbox for full image with small margin.

    YOLO format: class_id x_center y_center width height (normalized 0-1)
    """
    # Center is always 0.5, 0.5 for full image
    x_center = 0.5
    y_center = 0.5
    width = 1.0 - (2 * margin)
    height = 1.0 - (2 * margin)

    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_classification_to_yolo(
    source_dir: Path,
    output_dir: Path,
    class_mapping_func,
    mode: str = "disease"
) -> Tuple[int, int]:
    """
    Convert classification dataset (folder per class) to YOLO format.

    Returns: (success_count, skip_count)
    """
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0

    # Process each class folder
    for class_folder in source_dir.iterdir():
        if not class_folder.is_dir():
            continue

        class_id = class_mapping_func(class_folder.name)
        if class_id is None:
            print(f"  Skipping unknown class: {class_folder.name}")
            skip_count += 1
            continue

        print(f"  Processing class '{class_folder.name}' -> ID {class_id}")

        # Process images in class folder
        for img_path in class_folder.rglob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            try:
                # Open image to get dimensions
                with Image.open(img_path) as img:
                    width, height = img.size

                # Create unique filename
                new_name = f"{mode}_{class_id}_{success_count:06d}"
                new_img_path = output_images / f"{new_name}{img_path.suffix.lower()}"
                new_label_path = output_labels / f"{new_name}.txt"

                # Copy image
                shutil.copy2(img_path, new_img_path)

                # Create YOLO label (full image bbox)
                bbox = create_full_image_bbox(width, height)
                new_label_path.write_text(f"{class_id} {bbox}\n")

                success_count += 1

            except Exception as e:
                print(f"    Error processing {img_path}: {e}")
                skip_count += 1

    return success_count, skip_count


def convert_yolo_dataset(
    source_dir: Path,
    output_dir: Path,
    class_mapping: Dict[str, int],
    source_classes: List[str]
) -> Tuple[int, int]:
    """
    Convert existing YOLO dataset with different class IDs.

    Returns: (success_count, skip_count)
    """
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0

    # Find images and labels directories
    source_images = source_dir / "images"
    source_labels = source_dir / "labels"

    if not source_images.exists():
        # Try train/valid/test structure
        for split in ["train", "valid", "test"]:
            split_images = source_dir / split / "images"
            split_labels = source_dir / split / "labels"

            if split_images.exists():
                s, sk = _convert_yolo_split(
                    split_images, split_labels,
                    output_images, output_labels,
                    class_mapping, source_classes, success_count
                )
                success_count += s
                skip_count += sk

        return success_count, skip_count

    return _convert_yolo_split(
        source_images, source_labels,
        output_images, output_labels,
        class_mapping, source_classes, 0
    )


def _convert_yolo_split(
    source_images: Path,
    source_labels: Path,
    output_images: Path,
    output_labels: Path,
    class_mapping: Dict[str, int],
    source_classes: List[str],
    start_idx: int
) -> Tuple[int, int]:
    """Convert a single YOLO split."""
    success_count = 0
    skip_count = 0

    for img_path in source_images.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_path = source_labels / f"{img_path.stem}.txt"

        if not label_path.exists():
            skip_count += 1
            continue

        try:
            # Read and convert labels
            new_lines = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    old_class_id = int(parts[0])
                    if old_class_id < len(source_classes):
                        old_class_name = source_classes[old_class_id]
                        new_class_id = class_mapping.get(
                            normalize_class_name(old_class_name)
                        )

                        if new_class_id is not None:
                            new_lines.append(
                                f"{new_class_id} {' '.join(parts[1:])}"
                            )

            if new_lines:
                # Copy image and save converted labels
                idx = start_idx + success_count
                new_name = f"mite_{idx:06d}"
                new_img_path = output_images / f"{new_name}{img_path.suffix.lower()}"
                new_label_path = output_labels / f"{new_name}.txt"

                shutil.copy2(img_path, new_img_path)
                new_label_path.write_text("\n".join(new_lines) + "\n")

                success_count += 1
            else:
                skip_count += 1

        except Exception as e:
            print(f"    Error processing {img_path}: {e}")
            skip_count += 1

    return success_count, skip_count


def convert_mode_a_datasets():
    """Convert all Mode A (disease) datasets to YOLO format."""
    print("\n" + "=" * 60)
    print("Converting Mode A (Disease) Datasets")
    print("=" * 60)

    output_dir = PROCESSED_DATA_DIR / "mode_a"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_skip = 0

    # Convert Mendeley Rose dataset
    mendeley_dir = RAW_DATA_DIR / "mendeley_rose"
    if mendeley_dir.exists():
        print(f"\nProcessing Mendeley Rose: {mendeley_dir}")
        s, sk = convert_classification_to_yolo(
            mendeley_dir, output_dir, get_disease_class_id, "mendeley"
        )
        total_success += s
        total_skip += sk
        print(f"  Converted: {s}, Skipped: {sk}")

    # Convert Kaggle Rose dataset
    kaggle_dir = RAW_DATA_DIR / "kaggle_rose"
    if kaggle_dir.exists():
        print(f"\nProcessing Kaggle Rose: {kaggle_dir}")
        s, sk = convert_classification_to_yolo(
            kaggle_dir, output_dir, get_disease_class_id, "kaggle"
        )
        total_success += s
        total_skip += sk
        print(f"  Converted: {s}, Skipped: {sk}")

    print(f"\nMode A Total - Converted: {total_success}, Skipped: {total_skip}")

    # Save class names
    classes_file = output_dir / "classes.txt"
    classes_file.write_text("\n".join(DISEASE_CLASSES))
    print(f"Class names saved to: {classes_file}")

    return total_success


def convert_mode_b_datasets():
    """Convert all Mode B (mite) datasets to YOLO format."""
    print("\n" + "=" * 60)
    print("Converting Mode B (Mite) Datasets")
    print("=" * 60)

    output_dir = PROCESSED_DATA_DIR / "mode_b"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_skip = 0

    # Convert PlantVillage TSSM (classification format)
    pv_dir = RAW_DATA_DIR / "plantvillage_tssm"
    if pv_dir.exists():
        print(f"\nProcessing PlantVillage TSSM: {pv_dir}")
        s, sk = convert_classification_to_yolo(
            pv_dir, output_dir, get_mite_class_id, "pv"
        )
        total_success += s
        total_skip += sk
        print(f"  Converted: {s}, Skipped: {sk}")

    # Convert Roboflow TSSM (already YOLO format)
    rf_dir = RAW_DATA_DIR / "roboflow_tssm"
    if rf_dir.exists():
        print(f"\nProcessing Roboflow TSSM: {rf_dir}")

        # Try to read original class names from data.yaml
        data_yaml = rf_dir / "data.yaml"
        source_classes = MITE_CLASSES  # Default

        if data_yaml.exists():
            try:
                import yaml
                with open(data_yaml) as f:
                    config = yaml.safe_load(f)
                    if "names" in config:
                        source_classes = config["names"]
                        if isinstance(source_classes, dict):
                            source_classes = [source_classes[i] for i in sorted(source_classes.keys())]
            except Exception:
                pass

        s, sk = convert_yolo_dataset(
            rf_dir, output_dir, MITE_CLASS_MAPPING, source_classes
        )
        total_success += s
        total_skip += sk
        print(f"  Converted: {s}, Skipped: {sk}")

    print(f"\nMode B Total - Converted: {total_success}, Skipped: {total_skip}")

    # Save class names
    classes_file = output_dir / "classes.txt"
    classes_file.write_text("\n".join(MITE_CLASSES))
    print(f"Class names saved to: {classes_file}")

    return total_success


def main():
    """Main conversion function."""
    print("PlantVillage Rose Edition - Annotation Converter")
    print("=" * 60)
    print(f"Raw data: {RAW_DATA_DIR}")
    print(f"Processed data: {PROCESSED_DATA_DIR}")

    if not RAW_DATA_DIR.exists():
        print(f"\nError: Raw data directory not found: {RAW_DATA_DIR}")
        print("Run download_datasets.py first.")
        return 1

    mode_a_count = convert_mode_a_datasets()
    mode_b_count = convert_mode_b_datasets()

    print("\n" + "=" * 60)
    print("Conversion Complete")
    print("=" * 60)
    print(f"Mode A images: {mode_a_count}")
    print(f"Mode B images: {mode_b_count}")
    print(f"\nProcessed data saved to: {PROCESSED_DATA_DIR}")

    if mode_a_count == 0 and mode_b_count == 0:
        print("\nWarning: No images converted. Check raw data directory.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
