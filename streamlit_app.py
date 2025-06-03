"""
Streamlit app for Insurance Claims Adjudication Pipeline.

This is the main entry point for the Streamlit application.
The app is organized into modular components in the app/ directory.
"""

import time
import logging
import warnings

from datetime import datetime

import streamlit as st

# Suppress annoying torch.classes warnings in Streamlit
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

from demo.utils import format_duration
from demo.claims import display_claim_browser, display_claim_details
from demo.results import export_results, display_processing_history, display_adjudication_results
from demo.session import load_pipeline, initialize_session_state
from assurity_demo.config import AllowedModelsOCR, AllowedModelsClaim
from demo.pipeline_runner import run_pipeline_on_existing_claim


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model Selection
        st.subheader("🤖 Model Selection")

        # OCR Model Selection
        ocr_model_names = {
            AllowedModelsOCR.GPT_4O: "GPT-4o (OpenAI) - Older multimodal model",
            AllowedModelsOCR.GPT_41: "GPT-4.1 (OpenAI) - NEWEST, best coding & instruction following",
            AllowedModelsOCR.GEMINI_2_0_FLASH: "Gemini 2.0 Flash (Google) - Fast, efficient",
        }

        selected_ocr_model = st.selectbox(
            "OCR Model:",
            options=list(AllowedModelsOCR),
            format_func=lambda x: ocr_model_names[x],
            index=list(AllowedModelsOCR).index(st.session_state.selected_ocr_model),
            help="Model used for OCR text extraction from PDFs",
        )

        # Claim Model Selection
        claim_model_names = {
            AllowedModelsClaim.GPT_4O: "GPT-4o (OpenAI) - Older multimodal model",
            AllowedModelsClaim.GPT_41: "GPT-4.1 (OpenAI) - NEWEST, best coding & instruction following",
            AllowedModelsClaim.GEMINI_2_0_FLASH: "Gemini 2.0 Flash (Google) - Fast, efficient",
            AllowedModelsClaim.GEMINI_2_5_FLASH_PREVIEW_05_20: "Gemini 2.5 Flash Preview (Google) - Experimental",
        }

        selected_claim_model = st.selectbox(
            "Claim Processing Model:",
            options=list(AllowedModelsClaim),
            format_func=lambda x: claim_model_names[x],
            index=list(AllowedModelsClaim).index(st.session_state.selected_claim_model),
            help="Model used for claim analysis (dates, exclusions, benefits, payments)",
        )

        # Check if models have changed and update session state
        models_changed = (
            selected_ocr_model != st.session_state.selected_ocr_model
            or selected_claim_model != st.session_state.selected_claim_model
        )

        if models_changed:
            st.session_state.selected_ocr_model = selected_ocr_model
            st.session_state.selected_claim_model = selected_claim_model
            st.session_state.pipeline = None  # Reset pipeline when models change
            st.info("🔄 Models updated - pipeline will reinitialize")

        st.divider()

        # Pipeline status
        if st.session_state.pipeline is None:
            st.warning("Pipeline not initialized")
            if st.button("Initialize Pipeline"):
                load_pipeline(silent=False)  # Detailed UI for explicit user action
                if st.session_state.pipeline is not None:
                    st.rerun()
        else:
            st.success("Pipeline ready")

            # Show current model configuration
            with st.expander("📋 Current Models"):
                st.write(f"**OCR:** {ocr_model_names[st.session_state.selected_ocr_model]}")
                st.write(f"**Claim:** {claim_model_names[st.session_state.selected_claim_model]}")

            if st.button("Reload Pipeline"):
                st.session_state.pipeline = None
                load_pipeline(silent=False)  # Detailed UI for explicit user action
                if st.session_state.pipeline is not None:
                    st.rerun()

        st.divider()

        # Export results
        if st.session_state.processing_results:
            st.header("📥 Export Results")
            latest_result_entry = st.session_state.processing_results[-1]

            # Extract the actual result for filename generation
            if isinstance(latest_result_entry, dict):
                latest_result = latest_result_entry["result"]
            else:
                latest_result = latest_result_entry

            if st.download_button(
                label="Download Latest Results (JSON)",
                data=export_results(latest_result_entry),
                file_name=f"{latest_result.policy_id}_{latest_result.claim_id}_results.json",
                mime="application/json",
            ):
                st.success("Results downloaded!")


def render_main_content():
    """Render the main content area."""
    st.title("🏥 Insurance Claims Adjudication Pipeline")
    st.markdown(
        "Process insurance claims using AI-powered analysis - browse and analyze existing claims from the demo dataset."
    )

    # Show pipeline status prominently
    if st.session_state.pipeline is None:
        st.error(
            "🚨 **Pipeline Not Initialized** - Please initialize the pipeline in the sidebar before processing any claims."
        )
    else:
        st.success("✅ **Pipeline Ready** - You can now process claims using the options below.")

    # Main content - Browse Existing Claims only
    st.markdown("---")

    selected_claim = display_claim_browser()

    if selected_claim:
        st.markdown("---")
        display_claim_details(selected_claim)

        # Process button for selected claim
        pipeline_ready = st.session_state.pipeline is not None

        if not pipeline_ready:
            st.warning("⚠️ **Please initialize the pipeline in the sidebar before processing claims.**")

        if st.button("🚀 Run Pipeline on Selected Claim", type="primary", disabled=not pipeline_ready):
            processing_start = time.time()

            result = run_pipeline_on_existing_claim(selected_claim)

            if result:
                processing_time = time.time() - processing_start

                # Store result and timing in session state
                result_with_timing = {
                    "result": result,
                    "processing_time": processing_time,
                    "timestamp": datetime.now(),
                    "mode": "existing_claim",
                }
                st.session_state.processing_results.append(result_with_timing)

                # Display results
                st.markdown("---")

                # Show timing summary first
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Total Time", format_duration(processing_time))
                with col2:
                    st.metric("🚀 Processing Mode", "Fast (Existing OCR)")
                with col3:
                    st.metric("📊 Decision", result.decision.decision_recommendation.upper())

                display_adjudication_results(result)

                st.success("🎉 Processing completed successfully!")
            else:
                st.error("❌ Pipeline execution failed")

    # Display processing history
    display_processing_history()


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="Insurance Claims Adjudication", page_icon="🏥", layout="wide")

    initialize_session_state()

    # Render sidebar and main content
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
