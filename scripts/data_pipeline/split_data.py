#!/usr/bin/env python3
"""
Split processed datasets into train/val/test sets.

Split ratio: 70% train, 20% validation, 10% test
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# Random seed for reproducibility
RANDOM_SEED = 42


def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Get matching image and label file pairs."""
    pairs = []

    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))

    return pairs


def get_class_distribution(pairs: List[Tuple[Path, Path]]) -> dict:
    """Get class distribution from label files."""
    class_counts = defaultdict(int)

    for _, label_path in pairs:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

    return dict(class_counts)


def stratified_split(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42
) -> Tuple[List, List, List]:
    """
    Perform stratified split to maintain class distribution.

    Returns: (train_pairs, val_pairs, test_pairs)
    """
    random.seed(seed)

    # Group by primary class (first class in label file)
    class_groups = defaultdict(list)
    for pair in pairs:
        img_path, label_path = pair
        with open(label_path, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                class_id = int(first_line.split()[0])
                class_groups[class_id].append(pair)

    train_pairs = []
    val_pairs = []
    test_pairs = []

    # Split each class proportionally
    for class_id, class_pairs in class_groups.items():
        random.shuffle(class_pairs)
        n = len(class_pairs)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_pairs.extend(class_pairs[:train_end])
        val_pairs.extend(class_pairs[train_end:val_end])
        test_pairs.extend(class_pairs[val_end:])

    # Shuffle final lists
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)

    return train_pairs, val_pairs, test_pairs


def copy_pairs_to_split(
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    split_name: str
) -> int:
    """Copy image/label pairs to split directory."""
    images_out = output_dir / split_name / "images"
    labels_out = output_dir / split_name / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in pairs:
        shutil.copy2(img_path, images_out / img_path.name)
        shutil.copy2(label_path, labels_out / label_path.name)

    return len(pairs)


def split_dataset(mode: str, classes: List[str]) -> bool:
    """Split a mode's dataset into train/val/test."""
    print(f"\n{'=' * 60}")
    print(f"Splitting Mode {mode.upper()} Dataset")
    print("=" * 60)

    source_dir = PROCESSED_DATA_DIR / f"mode_{mode}"
    output_dir = PROCESSED_DATA_DIR / f"mode_{mode}_split"

    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        print("Run convert_annotations.py first.")
        return False

    # Get all pairs
    pairs = get_image_label_pairs(images_dir, labels_dir)
    print(f"Total image-label pairs: {len(pairs)}")

    if len(pairs) == 0:
        print("Error: No image-label pairs found.")
        return False

    # Show class distribution before split
    print("\nClass distribution (before split):")
    dist = get_class_distribution(pairs)
    for class_id, count in sorted(dist.items()):
        class_name = classes[class_id] if class_id < len(classes) else f"unknown_{class_id}"
        print(f"  {class_id}: {class_name} = {count}")

    # Perform stratified split
    train_pairs, val_pairs, test_pairs = stratified_split(
        pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_pairs)} ({len(train_pairs)/len(pairs)*100:.1f}%)")
    print(f"  Val:   {len(val_pairs)} ({len(val_pairs)/len(pairs)*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} ({len(test_pairs)/len(pairs)*100:.1f}%)")

    # Clear output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Copy to split directories
    copy_pairs_to_split(train_pairs, output_dir, "train")
    copy_pairs_to_split(val_pairs, output_dir, "val")
    copy_pairs_to_split(test_pairs, output_dir, "test")

    # Copy classes file
    classes_file = source_dir / "classes.txt"
    if classes_file.exists():
        shutil.copy2(classes_file, output_dir / "classes.txt")

    # Show class distribution after split
    print("\nClass distribution per split:")
    for split_name in ["train", "val", "test"]:
        split_pairs = get_image_label_pairs(
            output_dir / split_name / "images",
            output_dir / split_name / "labels"
        )
        dist = get_class_distribution(split_pairs)
        print(f"\n  {split_name.upper()}:")
        for class_id, count in sorted(dist.items()):
            class_name = classes[class_id] if class_id < len(classes) else f"unknown_{class_id}"
            print(f"    {class_id}: {class_name} = {count}")

    print(f"\nSplit data saved to: {output_dir}")
    return True


def main():
    """Main split function."""
    print("PlantVillage Rose Edition - Data Splitter")
    print("=" * 60)
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print(f"Random seed: {RANDOM_SEED}")

    # Mode A classes
    mode_a_classes = ["black_spot", "rust", "downy_mildew", "stippling", "healthy"]

    # Mode B classes
    mode_b_classes = ["egg", "larva", "nymph", "adult_female", "adult_male"]

    results = {}

    # Split Mode A
    results["mode_a"] = split_dataset("a", mode_a_classes)

    # Split Mode B
    results["mode_b"] = split_dataset("b", mode_b_classes)

    # Summary
    print("\n" + "=" * 60)
    print("Split Summary")
    print("=" * 60)
    for mode, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {mode}: {status}")

    if all(results.values()):
        print("\nAll datasets split successfully!")
        return 0
    else:
        print("\nSome splits failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
