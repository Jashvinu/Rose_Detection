#!/usr/bin/env python3
"""
Mode A: Rose Disease Scanner

YOLOv8-Nano based disease detection for field-level rose health monitoring.
Supports single image upload and video processing.
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

from app.components.inference_engine import InferenceEngine

# Page config
st.set_page_config(
    page_title="Mode A - Disease Scanner",
    page_icon="🔬",
    layout="wide"
)

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
EXPORTED_DIR = MODELS_DIR / "exported"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Disease classes with descriptions
DISEASE_INFO = {
    "black_spot": {
        "name": "Black Spot",
        "color": "#000000",
        "description": "Fungal disease (Diplocarpon rosae) causing black circular spots with fringed edges.",
        "severity": "High",
        "action": "Remove infected leaves, apply fungicide, improve air circulation."
    },
    "rust": {
        "name": "Rust",
        "color": "#FF8C00",
        "description": "Fungal infection (Phragmidium) showing orange/rust-colored pustules on leaves.",
        "severity": "Medium",
        "action": "Remove infected leaves, apply sulfur-based fungicide."
    },
    "downy_mildew": {
        "name": "Downy Mildew",
        "color": "#808080",
        "description": "Oomycete infection (Peronospora sparsa) with grayish-white downy growth.",
        "severity": "High",
        "action": "Improve ventilation, reduce humidity, apply copper-based fungicide."
    },
    "stippling": {
        "name": "Stippling",
        "color": "#FFD700",
        "description": "Damage from spider mite feeding - tiny yellow/white dots on leaves.",
        "severity": "Medium",
        "action": "Use miticides, increase humidity, introduce predatory mites."
    },
    "healthy": {
        "name": "Healthy",
        "color": "#00FF00",
        "description": "No visible disease symptoms detected.",
        "severity": "None",
        "action": "Maintain current care regimen."
    }
}


def find_mode_a_models():
    """Find available Mode A models."""
    models = []

    # TFLite models
    if EXPORTED_DIR.exists():
        for f in EXPORTED_DIR.glob("*mode_a*.tflite"):
            models.append(str(f))
        for f in EXPORTED_DIR.glob("*disease*.tflite"):
            models.append(str(f))

    # Checkpoint models
    if CHECKPOINTS_DIR.exists():
        for exp_dir in CHECKPOINTS_DIR.iterdir():
            if "mode_a" in exp_dir.name.lower():
                best_pt = exp_dir / "weights" / "best.pt"
                if best_pt.exists():
                    models.append(str(best_pt))

    return list(set(models))


@st.cache_resource
def load_model(model_path: str, confidence: float):
    """Load and cache the inference engine."""
    engine = InferenceEngine(
        model_path=model_path,
        mode="a",
        confidence_threshold=confidence
    )
    if engine.load_model():
        return engine
    return None


def display_results(results: dict):
    """Display detection results with disease information."""
    st.subheader("Detection Results")

    class_counts = results.get("class_counts", {})
    detections = results.get("detections", [])

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Detections", len(detections))

    with col2:
        diseases = sum(v for k, v in class_counts.items() if k != "healthy")
        st.metric("Diseases Found", diseases)

    with col3:
        healthy = class_counts.get("healthy", 0)
        st.metric("Healthy Areas", healthy)

    with col4:
        if detections:
            avg_conf = np.mean([d["confidence"] for d in detections])
            st.metric("Avg Confidence", f"{avg_conf:.1%}")

    st.divider()

    # Disease breakdown
    st.subheader("Disease Breakdown")

    for class_name, count in class_counts.items():
        if count > 0:
            info = DISEASE_INFO.get(class_name, {})

            with st.expander(f"**{info.get('name', class_name)}** - {count} detection(s)", expanded=True):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown(f"""
                    **Severity**: {info.get('severity', 'Unknown')}

                    **Count**: {count}
                    """)

                with col2:
                    st.markdown(f"""
                    **Description**: {info.get('description', 'No description')}

                    **Recommended Action**: {info.get('action', 'Consult expert')}
                    """)


def process_image(engine: InferenceEngine, image: np.ndarray):
    """Process a single image and display results."""
    with st.spinner("Analyzing image..."):
        results = engine.predict(image)

    # Display annotated image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Overlay")
        annotated = engine.draw_detections(image, results["detections"])
        st.image(annotated, use_container_width=True)

    # Display results
    display_results(results)

    # Export options
    st.subheader("Export Results")
    col1, col2 = st.columns(2)

    with col1:
        # Download annotated image
        from PIL import Image as PILImage
        import io

        annotated_pil = PILImage.fromarray(annotated)
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")

        st.download_button(
            "Download Annotated Image",
            buf.getvalue(),
            "annotated_result.png",
            "image/png"
        )

    with col2:
        # Download JSON results
        results_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            "Download Results JSON",
            results_json,
            "detection_results.json",
            "application/json"
        )


def process_video(engine: InferenceEngine, video_file, frame_skip: int):
    """Process uploaded video file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress):
        progress_bar.progress(progress)
        status_text.text(f"Processing: {progress:.1%}")

    # Process video
    with st.spinner("Processing video..."):
        results = engine.predict_video(
            tmp_path,
            frame_skip=frame_skip,
            progress_callback=update_progress
        )

    progress_bar.empty()
    status_text.empty()

    # Display summary
    st.subheader("Video Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Frames", results["total_frames"])

    with col2:
        st.metric("Processed Frames", results["processed_frames"])

    with col3:
        st.metric("FPS", f"{results['fps']:.1f}")

    with col4:
        total_detections = sum(
            len(fr["detections"]) for fr in results["frame_results"]
        )
        st.metric("Total Detections", total_detections)

    # Aggregate counts
    st.subheader("Aggregate Disease Counts")

    total_counts = results["total_counts"]
    for class_name, count in total_counts.items():
        if count > 0:
            info = DISEASE_INFO.get(class_name, {})
            st.markdown(f"**{info.get('name', class_name)}**: {count}")

    # Timeline chart
    st.subheader("Detection Timeline")

    import pandas as pd

    timeline_data = []
    for fr in results["frame_results"]:
        frame_num = fr["frame"]
        for cls, count in fr["class_counts"].items():
            if count > 0:
                timeline_data.append({
                    "Frame": frame_num,
                    "Class": cls,
                    "Count": count
                })

    if timeline_data:
        df = pd.DataFrame(timeline_data)
        st.line_chart(df.pivot(index="Frame", columns="Class", values="Count").fillna(0))

    # Cleanup temp file
    Path(tmp_path).unlink(missing_ok=True)


def main():
    """Main Mode A page."""
    st.title("🔬 Mode A: Rose Disease Scanner")
    st.markdown("""
    Upload rose leaf images or video for disease detection.
    The model identifies Black Spot, Rust, Downy Mildew, Stippling damage, and healthy leaves.
    """)

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        # Model selection
        models = find_mode_a_models()
        if not models:
            st.error("No Mode A models found. Train a model first.")
            st.stop()

        model_path = st.selectbox(
            "Select Model",
            models,
            format_func=lambda x: Path(x).name
        )

        # Confidence threshold
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detections"
        )

        # Video settings
        st.subheader("Video Settings")
        frame_skip = st.slider(
            "Frame Skip",
            min_value=1,
            max_value=30,
            value=5,
            help="Process every Nth frame"
        )

    # Load model
    engine = load_model(model_path, confidence)
    if engine is None:
        st.error("Failed to load model. Check the model file.")
        st.stop()

    st.success(f"Model loaded: {Path(model_path).name}")

    # Input tabs
    tab1, tab2 = st.tabs(["📷 Image Upload", "🎥 Video Upload"])

    with tab1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            key="image_upload"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image)
            process_image(engine, image_array)

    with tab2:
        st.subheader("Upload Video")
        video_file = st.file_uploader(
            "Choose a video",
            type=["mp4", "avi", "mov"],
            key="video_upload"
        )

        if video_file:
            st.video(video_file)

            if st.button("Analyze Video"):
                video_file.seek(0)
                process_video(engine, video_file, frame_skip)


if __name__ == "__main__":
    main()
