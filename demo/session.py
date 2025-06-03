"""Session state management for the Streamlit app."""

import streamlit as st

from assurity_demo.config import AllowedModelsOCR, AllowedModelsClaim
from assurity_demo.pipeline import Pipeline


def initialize_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "processing_results" not in st.session_state:
        st.session_state.processing_results = []
    if "available_claims" not in st.session_state:
        st.session_state.available_claims = []
    if "selected_claim" not in st.session_state:
        st.session_state.selected_claim = None
    if "selected_ocr_model" not in st.session_state:
        st.session_state.selected_ocr_model = AllowedModelsOCR.GPT_41
    if "selected_claim_model" not in st.session_state:
        st.session_state.selected_claim_model = AllowedModelsClaim.GPT_41


def load_pipeline(silent: bool = False):
    """Load the Pipeline with caching."""
    if st.session_state.pipeline is None:
        if silent:
            # Silent initialization - just show a simple spinner
            with st.spinner("Initializing pipeline..."):
                try:
                    pipeline = Pipeline(
                        ocr_model=st.session_state.selected_ocr_model, claim_model=st.session_state.selected_claim_model
                    )
                    st.session_state.pipeline = pipeline
                except Exception as e:
                    st.error(f"Failed to initialize Pipeline: {str(e)}")
                    st.error("Please check your environment variables and API keys in the .env file.")
                    st.stop()
        else:
            # Detailed initialization with progress UI
            with st.spinner("Initializing Pipeline..."):
                try:
                    # Create a logging container
                    log_container = st.empty()
                    progress_bar = st.progress(0)

                    with log_container.container():
                        st.write("🔧 **Pipeline Initialization Steps:**")

                        # Step 1: Show model configuration
                        st.write("📋 **Selected Models:**")
                        st.write(f"   • OCR Model: `{st.session_state.selected_ocr_model.value}`")
                        st.write(f"   • Claim Model: `{st.session_state.selected_claim_model.value}`")
                        progress_bar.progress(20)

                        st.write("🏗️ **Creating Pipeline Instance...**")
                        progress_bar.progress(40)

                        # Initialize pipeline
                        pipeline = Pipeline(
                            ocr_model=st.session_state.selected_ocr_model,
                            claim_model=st.session_state.selected_claim_model,
                        )
                        progress_bar.progress(60)

                        st.write("🤖 **Initializing OCR Processor...**")
                        progress_bar.progress(70)

                        st.write("🧠 **Initializing Claim Processor...**")
                        progress_bar.progress(80)

                        st.write("📊 **Loading Benefit Databases...**")
                        progress_bar.progress(90)

                        st.write("✅ **Pipeline Ready!**")
                        progress_bar.progress(100)

                    st.session_state.pipeline = pipeline

                    # Clear the logging container and show success
                    log_container.empty()
                    progress_bar.empty()
                    st.success("Pipeline initialized successfully!")

                except Exception as e:
                    st.error(f"Failed to initialize Pipeline: {str(e)}")
                    st.error("Please check your environment variables and API keys in the .env file.")
                    st.error(f"**Debug Info:** {type(e).__name__}: {str(e)}")
                    st.stop()
    return st.session_state.pipeline
