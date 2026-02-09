#!/usr/bin/env python3
"""
PlantVillage Rose Edition - Main Streamlit Application

Two-mode ML system for rose disease detection and spider mite counting:
- Mode A (Row Scanner): YOLOv8-Nano for disease/stippling detection
- Mode B (Macro Counter): YOLOv8-Small + SAHI for individual mite counting
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="PlantVillage Rose Edition",
    page_icon="🌹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EXPORTED_DIR = MODELS_DIR / "exported"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"


def get_available_models():
    """Find available models for both modes."""
    models = {
        "mode_a": [],
        "mode_b": []
    }

    # Check exported TFLite models
    if EXPORTED_DIR.exists():
        for f in EXPORTED_DIR.glob("*.tflite"):
            if "mode_a" in f.name.lower() or "disease" in f.name.lower():
                models["mode_a"].append(str(f))
            elif "mode_b" in f.name.lower() or "mite" in f.name.lower():
                models["mode_b"].append(str(f))

    # Check checkpoint models
    if CHECKPOINTS_DIR.exists():
        for exp_dir in CHECKPOINTS_DIR.iterdir():
            best_pt = exp_dir / "weights" / "best.pt"
            if best_pt.exists():
                if "mode_a" in exp_dir.name.lower():
                    models["mode_a"].append(str(best_pt))
                elif "mode_b" in exp_dir.name.lower():
                    models["mode_b"].append(str(best_pt))

    return models


def main():
    """Main application entry point."""
    # Header
    st.title("🌹 PlantVillage Rose Edition")
    st.markdown("""
    **AI-powered rose health analysis system**

    Select a mode from the sidebar to get started:
    - **Mode A (Row Scanner)**: Detect diseases in rose leaves from field images
    - **Mode B (Macro Counter)**: Count spider mites in high-resolution macro photos
    """)

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        st.markdown("""
        📄 **Pages**
        - [Mode A - Disease Scanner](/Mode_A_Scanner)
        - [Mode B - Mite Counter](/Mode_B_Counter)
        """)

        st.divider()

        # Model status
        st.header("Model Status")
        models = get_available_models()

        st.subheader("Mode A (Disease)")
        if models["mode_a"]:
            for m in models["mode_a"]:
                st.success(f"✓ {Path(m).name}")
        else:
            st.warning("No models found. Train a model first.")

        st.subheader("Mode B (Mites)")
        if models["mode_b"]:
            for m in models["mode_b"]:
                st.success(f"✓ {Path(m).name}")
        else:
            st.warning("No models found. Train a model first.")

        st.divider()

        st.markdown("""
        **Quick Start**
        1. Train models using scripts in `scripts/training/`
        2. Export to TFLite using `scripts/export/`
        3. Use the mode pages above for inference
        """)

    # Main content - overview cards
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔬 Mode A: Disease Scanner")
        st.markdown("""
        **Purpose**: Row-level disease detection in rose fields

        **Model**: YOLOv8-Nano (FP16)

        **Detects**:
        - Black Spot
        - Rust
        - Downy Mildew
        - Stippling (mite damage)
        - Healthy leaves

        **Input**: Field photos or video

        **Speed**: <500ms per image
        """)

        if st.button("Open Mode A", key="btn_mode_a"):
            st.switch_page("pages/01_mode_a_scanner.py")

    with col2:
        st.subheader("🔍 Mode B: Mite Counter")
        st.markdown("""
        **Purpose**: Count individual mites in macro photos

        **Model**: YOLOv8-Small + SAHI (FP16)

        **Detects**:
        - Eggs
        - Larvae
        - Nymphs
        - Adult Females
        - Adult Males

        **Input**: High-resolution macro photos

        **Speed**: ~3s per image (with SAHI)
        """)

        if st.button("Open Mode B", key="btn_mode_b"):
            st.switch_page("pages/02_mode_b_counter.py")

    # System info
    st.divider()
    st.subheader("System Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Project Root", str(PROJECT_ROOT.name))

    with col2:
        model_count = len(models["mode_a"]) + len(models["mode_b"])
        st.metric("Available Models", model_count)

    with col3:
        import sys
        st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}")

    # Footer
    st.divider()
    st.markdown("""
    ---
    **PlantVillage Rose Edition** | Built with Streamlit, YOLOv8, and SAHI

    For issues or contributions, see the project documentation.
    """)


if __name__ == "__main__":
    main()
