#!/usr/bin/env python3
"""
Download datasets for PlantVillage Rose Edition.

Datasets:
- Layer 1: Mendeley Rose Disease (4,725 images)
- Layer 1: Kaggle Rose Leaf Disease (14,910 images)
- Layer 2: PlantVillage Tomato TSSM (~1,600 images)
- Layer 3: TSSM Roboflow (32K instances)
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_mendeley_rose_disease(output_dir: Path) -> bool:
    """
    Download Mendeley Rose Disease Dataset.
    Source: https://data.mendeley.com/datasets/
    Contains: 4,725 images across disease categories
    """
    print("=" * 60)
    print("Downloading Mendeley Rose Disease Dataset...")
    print("=" * 60)

    try:
        from datasets import load_dataset

        # Try loading from HuggingFace mirror or Mendeley
        dataset_dir = output_dir / "mendeley_rose"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Note: This is a placeholder - actual Mendeley datasets require manual download
        # or specific API access. Users should download from:
        # https://data.mendeley.com/datasets/... (search for "rose disease")

        print(f"Dataset directory created: {dataset_dir}")
        print("\nNOTE: Mendeley datasets require manual download.")
        print("Please download rose disease dataset from Mendeley Data and place in:")
        print(f"  {dataset_dir}")
        print("\nAlternatively, the script will try HuggingFace datasets...")

        # Try HuggingFace alternative
        try:
            dataset = load_dataset("nirmalkumar7/rose-diseases", split="train")

            # Save images by class
            for i, item in enumerate(dataset):
                label = item.get("label", "unknown")
                label_dir = dataset_dir / str(label)
                label_dir.mkdir(parents=True, exist_ok=True)

                img = item["image"]
                img.save(label_dir / f"img_{i:05d}.jpg")

                if (i + 1) % 500 == 0:
                    print(f"  Saved {i + 1} images...")

            print(f"Successfully downloaded {len(dataset)} images")
            return True

        except Exception as e:
            print(f"HuggingFace download failed: {e}")
            print("Please download manually from Mendeley Data")
            return False

    except ImportError:
        print("Error: 'datasets' package not installed. Run: pip install datasets")
        return False


def download_kaggle_rose_disease(output_dir: Path) -> bool:
    """
    Download Kaggle Rose Leaf Disease Dataset.
    Source: https://www.kaggle.com/datasets/
    Contains: 14,910 images
    """
    print("\n" + "=" * 60)
    print("Downloading Kaggle Rose Leaf Disease Dataset...")
    print("=" * 60)

    try:
        import kagglehub

        dataset_dir = output_dir / "kaggle_rose"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Download rose leaf disease dataset
        # Common dataset names to try
        dataset_names = [
            "rashikrahmanpritom/rose-leaf-disease-dataset",
            "vipoooool/rose-diseases-dataset",
            "plant-village/rose-leaf-diseases",
        ]

        for dataset_name in dataset_names:
            try:
                print(f"Trying to download: {dataset_name}")
                path = kagglehub.dataset_download(dataset_name)

                # Move to our directory
                if path and Path(path).exists():
                    for item in Path(path).rglob("*"):
                        if item.is_file() and item.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            dest = dataset_dir / item.relative_to(path)
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest)

                    print(f"Successfully downloaded from {dataset_name}")
                    return True

            except Exception as e:
                print(f"  Failed: {e}")
                continue

        print("\nNOTE: Kaggle download requires authentication.")
        print("Set up Kaggle API credentials:")
        print("  1. Create ~/.kaggle/kaggle.json with your API token")
        print("  2. Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        print(f"\nManually download rose disease dataset to: {dataset_dir}")
        return False

    except ImportError:
        print("Error: 'kagglehub' package not installed. Run: pip install kagglehub")
        return False


def download_plantvillage_tssm(output_dir: Path) -> bool:
    """
    Download PlantVillage Tomato Spider Mite Dataset.
    Source: PlantVillage / HuggingFace
    Contains: ~1,600 images for transfer learning
    """
    print("\n" + "=" * 60)
    print("Downloading PlantVillage Tomato TSSM Dataset...")
    print("=" * 60)

    try:
        from datasets import load_dataset

        dataset_dir = output_dir / "plantvillage_tssm"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Load PlantVillage dataset and filter for spider mite
        try:
            dataset = load_dataset("gianlab/plantvillage", split="train")

            # Filter for tomato spider mite images
            spider_mite_keywords = ["spider", "mite", "tssm"]
            count = 0

            for i, item in enumerate(dataset):
                label = str(item.get("label", "")).lower()

                # Check if this is a spider mite related image
                if any(kw in label for kw in spider_mite_keywords) or "tomato" in label:
                    label_dir = dataset_dir / label.replace(" ", "_")
                    label_dir.mkdir(parents=True, exist_ok=True)

                    img = item["image"]
                    img.save(label_dir / f"img_{count:05d}.jpg")
                    count += 1

                    if count % 200 == 0:
                        print(f"  Saved {count} images...")

            print(f"Successfully downloaded {count} images")
            return count > 0

        except Exception as e:
            print(f"Download failed: {e}")
            print(f"Please download PlantVillage tomato dataset manually to: {dataset_dir}")
            return False

    except ImportError:
        print("Error: 'datasets' package not installed. Run: pip install datasets")
        return False


def download_roboflow_tssm(output_dir: Path, api_key: Optional[str] = None) -> bool:
    """
    Download TSSM Roboflow Dataset.
    Source: Roboflow Universe
    Contains: 32K instances with bounding box annotations
    """
    print("\n" + "=" * 60)
    print("Downloading Roboflow TSSM Dataset...")
    print("=" * 60)

    try:
        from roboflow import Roboflow

        dataset_dir = output_dir / "roboflow_tssm"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Get API key from environment or parameter
        api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")

        if not api_key:
            print("\nNOTE: Roboflow requires API key for download.")
            print("Get your API key from: https://app.roboflow.com/settings/api")
            print("Then set ROBOFLOW_API_KEY environment variable or pass api_key parameter")
            print(f"\nManually download TSSM dataset to: {dataset_dir}")

            # Create a placeholder info file
            info_file = dataset_dir / "DOWNLOAD_INSTRUCTIONS.txt"
            info_file.write_text("""
TSSM Roboflow Dataset Download Instructions
============================================

1. Go to https://universe.roboflow.com/
2. Search for "spider mite" or "TSSM"
3. Look for datasets with bounding box annotations
4. Download in YOLOv8 format
5. Extract to this directory

Recommended datasets:
- Two-spotted spider mite detection
- TSSM life stage classification

Expected structure after download:
roboflow_tssm/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
""")
            return False

        try:
            rf = Roboflow(api_key=api_key)

            # Try to find TSSM dataset
            # This is an example - actual workspace/project names vary
            project = rf.workspace().project("tssm-detection")
            dataset = project.version(1).download("yolov8", location=str(dataset_dir))

            print(f"Successfully downloaded to: {dataset_dir}")
            return True

        except Exception as e:
            print(f"Download failed: {e}")
            print("Please download manually from Roboflow Universe")
            return False

    except ImportError:
        print("Error: 'roboflow' package not installed. Run: pip install roboflow")
        return False


def main():
    """Main download function."""
    print("PlantVillage Rose Edition - Dataset Downloader")
    print("=" * 60)
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # Download all datasets
    results["mendeley_rose"] = download_mendeley_rose_disease(RAW_DATA_DIR)
    results["kaggle_rose"] = download_kaggle_rose_disease(RAW_DATA_DIR)
    results["plantvillage_tssm"] = download_plantvillage_tssm(RAW_DATA_DIR)
    results["roboflow_tssm"] = download_roboflow_tssm(RAW_DATA_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    for name, success in results.items():
        status = "SUCCESS" if success else "MANUAL DOWNLOAD REQUIRED"
        print(f"  {name}: {status}")

    # Count total images
    total_images = 0
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        total_images += len(list(RAW_DATA_DIR.rglob(ext)))

    print(f"\nTotal images downloaded: {total_images}")
    print(f"Data location: {RAW_DATA_DIR}")

    if not all(results.values()):
        print("\nSome datasets require manual download. See instructions above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
