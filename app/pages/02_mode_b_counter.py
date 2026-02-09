#!/usr/bin/env python3
"""
Mode B: Mite Counter

YOLOv8-Small + SAHI based detection for counting individual spider mites
in high-resolution macro photographs.
"""

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import json
import sys

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.components.sahi_processor import SAHIProcessor
from app.components.inference_engine import InferenceEngine

# Page config
st.set_page_config(
    page_title="Mode B - Mite Counter",
    page_icon="🔍",
    layout="wide"
)

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
EXPORTED_DIR = MODELS_DIR / "exported"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Mite life stage info
MITE_INFO = {
    "egg": {
        "name": "Eggs",
        "color": "#FFFF00",
        "size": "~0.14mm diameter",
        "description": "Spherical, translucent to opaque eggs laid on leaf undersides.",
        "lifecycle": "Hatch in 2-3 days"
    },
    "larva": {
        "name": "Larvae",
        "color": "#00FFFF",
        "size": "~0.15mm",
        "description": "Six-legged stage, pale colored, actively feeding.",
        "lifecycle": "Molts to nymph in 1-2 days"
    },
    "nymph": {
        "name": "Nymphs",
        "color": "#00FF00",
        "size": "~0.2-0.3mm",
        "description": "Eight-legged immature stage, developing color spots.",
        "lifecycle": "Two nymphal stages, 2-3 days total"
    },
    "adult_female": {
        "name": "Adult Females",
        "color": "#FF00FF",
        "size": "~0.4mm",
        "description": "Oval body, two characteristic dark spots visible on back.",
        "lifecycle": "Can lay 100+ eggs over 2-4 weeks"
    },
    "adult_male": {
        "name": "Adult Males",
        "color": "#FF0000",
        "size": "~0.3mm",
        "description": "Smaller wedge-shaped body, more active than females.",
        "lifecycle": "Lives ~2 weeks, mates with multiple females"
    }
}


def find_mode_b_models():
    """Find available Mode B models."""
    models = []

    # TFLite models
    if EXPORTED_DIR.exists():
        for f in EXPORTED_DIR.glob("*mode_b*.tflite"):
            models.append(str(f))
        for f in EXPORTED_DIR.glob("*mite*.tflite"):
            models.append(str(f))

    # Checkpoint models
    if CHECKPOINTS_DIR.exists():
        for exp_dir in CHECKPOINTS_DIR.iterdir():
            if "mode_b" in exp_dir.name.lower():
                best_pt = exp_dir / "weights" / "best.pt"
                if best_pt.exists():
                    models.append(str(best_pt))

    return list(set(models))


@st.cache_resource
def load_sahi_processor(
    model_path: str,
    slice_size: int,
    overlap: float,
    confidence: float
):
    """Load and cache the SAHI processor."""
    processor = SAHIProcessor(
        model_path=model_path,
        slice_size=slice_size,
        overlap_ratio=overlap,
        confidence_threshold=confidence
    )
    if processor.load_model():
        return processor
    return None


def display_population_analysis(results: dict):
    """Display population analysis from mite counts."""
    class_counts = results.get("class_counts", {})
    total = results.get("total_count", 0)

    # Population breakdown
    eggs = class_counts.get("egg", 0)
    larvae = class_counts.get("larva", 0)
    nymphs = class_counts.get("nymph", 0)
    adult_f = class_counts.get("adult_female", 0)
    adult_m = class_counts.get("adult_male", 0)

    adults = adult_f + adult_m
    immature = eggs + larvae + nymphs

    st.subheader("Population Analysis")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Mites", total)

    with col2:
        st.metric("Adults", adults)

    with col3:
        st.metric("Immature", immature)

    with col4:
        if total > 0:
            ratio = immature / total * 100
            st.metric("Immature Ratio", f"{ratio:.1f}%")

    with col5:
        if adults > 0:
            sex_ratio = adult_f / adults * 100
            st.metric("Female Ratio", f"{sex_ratio:.1f}%")

    # Population stage assessment
    st.divider()

    if total == 0:
        st.success("✓ No mites detected - Leaf appears clean")
    elif immature > adults * 2:
        st.warning("⚠️ **Growing Population** - High proportion of immature stages indicates population expansion")
        st.markdown("""
        **Recommended Actions:**
        - Apply miticide treatment immediately
        - Target eggs and larvae specifically
        - Plan follow-up application in 5-7 days
        """)
    elif adults > immature:
        st.info("ℹ️ **Mature Population** - Mostly adult mites present")
        st.markdown("""
        **Recommended Actions:**
        - Apply adulticide treatment
        - Monitor for egg laying
        - Consider beneficial predators
        """)
    else:
        st.warning("⚠️ **Active Infestation** - Mixed life stages detected")
        st.markdown("""
        **Recommended Actions:**
        - Combined treatment approach recommended
        - Multiple applications may be needed
        - Increase monitoring frequency
        """)


def display_life_stage_breakdown(results: dict):
    """Display detailed life stage breakdown."""
    class_counts = results.get("class_counts", {})

    st.subheader("Life Stage Details")

    for stage_key, count in class_counts.items():
        info = MITE_INFO.get(stage_key, {})

        with st.expander(
            f"**{info.get('name', stage_key)}** - {count} detected",
            expanded=(count > 0)
        ):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"""
                **Size**: {info.get('size', 'Unknown')}

                **Count**: {count}

                **Lifecycle**: {info.get('lifecycle', 'Unknown')}
                """)

            with col2:
                st.markdown(f"""
                **Description**: {info.get('description', 'No description')}
                """)


def process_image(processor: SAHIProcessor, image: np.ndarray, image_name: str):
    """Process image with SAHI and display results."""
    with st.spinner("Running sliced inference (this may take a moment)..."):
        results = processor.predict(image)

    # Display images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Overlay")
        annotated = processor.draw_detections(image, results["detections"])
        st.image(annotated, use_container_width=True)

    # Population analysis
    display_population_analysis(results)

    # Life stage breakdown
    display_life_stage_breakdown(results)

    # Detailed detection table
    st.subheader("Detection Details")

    if results["detections"]:
        import pandas as pd

        det_data = []
        for i, det in enumerate(results["detections"]):
            det_data.append({
                "ID": i + 1,
                "Class": det["class_name"],
                "Confidence": f"{det['confidence']:.1%}",
                "X1": int(det["bbox"][0]),
                "Y1": int(det["bbox"][1]),
                "X2": int(det["bbox"][2]),
                "Y2": int(det["bbox"][3])
            })

        df = pd.DataFrame(det_data)
        st.dataframe(df, use_container_width=True)

    # Export options
    st.subheader("Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Download annotated image
        from PIL import Image as PILImage
        import io

        annotated_pil = PILImage.fromarray(annotated)
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")

        st.download_button(
            "📷 Download Annotated Image",
            buf.getvalue(),
            "mite_detection.png",
            "image/png"
        )

    with col2:
        # Download JSON results
        results_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            "📋 Download JSON Results",
            results_json,
            "mite_counts.json",
            "application/json"
        )

    with col3:
        # Download COCO format
        coco_export = processor.export_to_coco(
            results["detections"],
            image_name,
            results["image_size"]
        )
        coco_json = json.dumps(coco_export, indent=2)
        st.download_button(
            "📦 Download COCO Format",
            coco_json,
            "coco_annotations.json",
            "application/json"
        )


def main():
    """Main Mode B page."""
    st.title("🔍 Mode B: Mite Counter")
    st.markdown("""
    Upload high-resolution macro photos of rose leaves for spider mite detection and counting.
    Uses SAHI (Slicing Aided Hyper Inference) for accurate small object detection.
    """)

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        # Model selection
        models = find_mode_b_models()
        if not models:
            st.error("No Mode B models found. Train a model first.")
            st.stop()

        model_path = st.selectbox(
            "Select Model",
            models,
            format_func=lambda x: Path(x).name
        )

        st.divider()

        # SAHI parameters
        st.subheader("SAHI Parameters")

        slice_size = st.slider(
            "Slice Size (px)",
            min_value=128,
            max_value=512,
            value=256,
            step=32,
            help="Size of each analysis tile"
        )

        overlap = st.slider(
            "Overlap Ratio",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Overlap between tiles to catch edge objects"
        )

        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detections"
        )

        st.divider()

        # Info panel
        st.subheader("About SAHI")
        st.markdown("""
        **Sliced Inference** breaks large images into
        overlapping tiles for better small object detection.

        **Tips:**
        - Smaller slices = better for tiny objects
        - Higher overlap = fewer missed detections
        - Lower confidence = more detections (may include false positives)
        """)

    # Load processor
    processor = load_sahi_processor(model_path, slice_size, overlap, confidence)
    if processor is None:
        st.error("Failed to load model. Check the model file.")
        st.stop()

    st.success(f"Model loaded: {Path(model_path).name}")

    # File upload
    st.subheader("Upload Macro Photo")

    uploaded_file = st.file_uploader(
        "Choose a high-resolution image",
        type=["jpg", "jpeg", "png", "tiff"],
        help="For best results, use macro photos at 10x magnification or higher"
    )

    if uploaded_file:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        # Show image info
        width, height = image.size
        st.info(f"Image size: {width} x {height} pixels | "
                f"Tiles: ~{((width // slice_size) + 1) * ((height // slice_size) + 1)}")

        # Process button
        if st.button("🔬 Analyze Image", type="primary"):
            process_image(processor, image_array, uploaded_file.name)

    else:
        # Show example/demo
        st.markdown("""
        ---
        ### How to Use

        1. **Capture macro photo** of rose leaf (10x+ magnification recommended)
        2. **Upload image** using the file uploader above
        3. **Adjust settings** in sidebar if needed
        4. **Click Analyze** to run detection
        5. **Export results** in your preferred format

        ### Life Stages Detected

        | Stage | Size | Appearance |
        |-------|------|------------|
        | Egg | 0.14mm | Spherical, translucent |
        | Larva | 0.15mm | 6 legs, pale |
        | Nymph | 0.2-0.3mm | 8 legs, developing spots |
        | Adult ♀ | 0.4mm | Oval, two dark spots |
        | Adult ♂ | 0.3mm | Wedge-shaped, active |
        """)


if __name__ == "__main__":
    main()
