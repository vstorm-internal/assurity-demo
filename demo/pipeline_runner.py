"""Pipeline execution functionality."""

import json
import time
import tempfile

from typing import Any, Dict
from datetime import datetime

import streamlit as st

from logzero import logger

from assurity_demo.models import Claim, Document, AdjudicationOutput

from .utils import format_duration


def run_pipeline_on_existing_claim(claim_info: Dict[str, Any]) -> AdjudicationOutput | None:
    """Run the pipeline on an existing claim, using pre-processed OCR data if available."""
    # Check if pipeline is initialized
    if st.session_state.pipeline is None:
        st.error("❌ **Pipeline not initialized!** Please initialize the pipeline in the sidebar first.")
        return None

    pipeline = st.session_state.pipeline

    # Start timing
    start_time = time.time()

    # Create main logging containers
    main_log = st.empty()
    progress_container = st.empty()
    phase_log = st.empty()
    timing_log = st.empty()

    try:
        # Determine processing method based on OCR availability
        if claim_info["has_ocr_output"]:
            return _run_fast_processing(
                claim_info, pipeline, start_time, main_log, progress_container, phase_log, timing_log
            )
        else:
            return _run_full_processing(
                claim_info, pipeline, start_time, main_log, progress_container, phase_log, timing_log
            )

    except Exception as e:
        error_msg = f"Pipeline execution failed: {str(e)}"
        st.error(f"❌ {error_msg}")
        st.error(f"**Error Type:** {type(e).__name__}")
        st.error(f"**Error Details:** {str(e)}")
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return None
    finally:
        # Clean up logging containers
        main_log.empty()
        progress_container.empty()
        phase_log.empty()
        timing_log.empty()


def _run_fast_processing(claim_info, pipeline, start_time, main_log, progress_container, phase_log, timing_log):
    """Run fast processing using existing OCR data."""
    with main_log.container():
        st.write("🚀 **Fast Processing Mode - Using Existing OCR Data**")
        st.write(f"📁 **Claim:** {claim_info['policy_id']} - {claim_info['claim_id']}")
        st.write(f"⏱️ **Started at:** {datetime.now().strftime('%H:%M:%S')}")

    with st.spinner(""):
        progress_bar = progress_container.progress(0)

        with phase_log.container():
            step_start = time.time()

            with open(claim_info["ocr_json_file"], "r") as f:
                claim_data = json.load(f)

            documents = [Document(**doc_data) for doc_data in claim_data.get("documents", [])]

            target_claim = Claim(
                policy_id=claim_data.get("policy_id", claim_info["policy_id"]),
                claim_id=claim_data.get("claim_id", claim_info["claim_id"]),
                documents=documents,
            )

            if target_claim.policy_id != claim_info["policy_id"] or target_claim.claim_id != claim_info["claim_id"]:
                st.warning(f"⚠️ OCR file contains different IDs: {target_claim.policy_id}/{target_claim.claim_id}")

            st.write("   ✅ Loaded claim data")
            progress_bar.progress(20)

            # Step 4: Run pipeline phases
            pipeline_start = time.time()
            progress_bar.progress(50)

            with tempfile.TemporaryDirectory() as tmp_dir:
                adjudication_output = _run_pipeline_phases(pipeline, target_claim, progress_bar, phase_log)

                pipeline_duration = time.time() - pipeline_start
                total_duration = time.time() - start_time

                st.write(f"   ✅ **Pipeline completed successfully!** ({format_duration(pipeline_duration)})")

                # Show timing summary
                with timing_log.container():
                    st.write("⏱️ **Timing Summary:**")
                    st.write(f"   📊 **Total Processing Time:** {format_duration(total_duration)}")
                    st.write(f"   🚀 **Pipeline Execution:** {format_duration(pipeline_duration)}")
                    st.write(f"   💾 **Data Loading:** {format_duration(total_duration - pipeline_duration)}")

                return adjudication_output


def _run_full_processing(claim_info, pipeline, start_time, main_log, progress_container, phase_log, timing_log):
    """Run full processing including OCR."""
    with main_log.container():
        st.write("🐌 **Full Processing Mode - Running OCR + Pipeline**")
        st.write(f"📁 **Claim:** {claim_info['policy_id']} - {claim_info['claim_id']}")
        st.write(f"📄 **Files to process:** {claim_info['file_count']} PDF files")
        st.warning("⚠️ **This will take several minutes due to OCR processing**")

    with st.spinner("Running OCR and adjudication pipeline (this may take several minutes)..."):
        progress_bar = progress_container.progress(0)

        with phase_log.container():
            st.write("**🔄 Processing Steps:**")

            # OCR Phase
            st.write("1️⃣ **Running OCR on PDF files...**")
            st.write("   ⚠️ This is the slowest step - processing each PDF file")
            progress_bar.progress(10)

            policy_id = claim_info["policy_id"]
            # Derive claim path from the first PDF file's parent directory
            if claim_info["pdf_files"]:
                claim_path = claim_info["pdf_files"][0].parent.parent  # Go up to policy directory
            else:
                st.error("❌ No PDF files found to process.")
                return None

            st.write(f"   📂 Processing policy directory: `{claim_path}`")
            progress_bar.progress(20)

            claims = pipeline.run_ocr_on_claims_in_directory(claim_path, policy_id)
            st.write(f"   ✅ OCR completed on {len(claims)} claims")
            progress_bar.progress(60)

            # Find target claim
            st.write("2️⃣ **Finding target claim...**")
            target_claim = None
            for claim in claims:
                if claim.claim_id == claim_info["claim_id"]:
                    target_claim = claim
                    break

            if not target_claim:
                st.error(f"❌ Could not find claim {claim_info['claim_id']} in the processed results.")
                return None

            st.write(f"   ✅ Found target claim with {len(target_claim.documents)} documents")
            progress_bar.progress(70)

            # Pipeline phases
            st.write("3️⃣ **Running Adjudication Pipeline...**")
            with tempfile.TemporaryDirectory():
                adjudication_output = _run_pipeline_phases(
                    pipeline, target_claim, progress_bar, phase_log, start_progress=75
                )
                return adjudication_output


def _run_pipeline_phases(pipeline, target_claim, progress_bar, phase_log, start_progress=55):
    """Run the individual pipeline phases with dynamic UI updates."""
    # Create a dynamic status container for pipeline phases
    pipeline_status = st.empty()

    # Phase 1: Dates Analysis
    with pipeline_status.container():
        st.write("🧠 **Analyzing dates...** ⏳")
    progress_bar.progress(start_progress)

    dates_output = pipeline.check_dates(target_claim.documents, "dates")

    with pipeline_status.container():
        st.write("🧠 **Analyzing dates...** ✅")
        st.write("🚫 **Checking exclusions...** ⏳")
    progress_bar.progress(start_progress + 10)

    # Phase 2: Exclusions Analysis
    exclusions_output = pipeline.check_exclusions(target_claim.documents, "exclusions")

    with pipeline_status.container():
        st.write("🧠 **Analyzing dates...** ✅")
        st.write("🚫 **Checking exclusions...** ✅")
        st.write("💰 **Mapping benefits...** ⏳")
    progress_bar.progress(start_progress + 20)

    # Phase 3: Benefits Mapping
    benefits_output = pipeline.map_benefits(target_claim.documents, "benefit_mapping")

    with pipeline_status.container():
        st.write("🧠 **Analyzing dates...** ✅")
        st.write("🚫 **Checking exclusions...** ✅")
        st.write("💰 **Mapping benefits...** ✅")
        st.write("💵 **Calculating payments...** ⏳")
    progress_bar.progress(start_progress + 30)

    # Phase 4: Payment Calculation
    from assurity_demo.models import Benefit

    benefits_to_pay = [Benefit(name=medical_proc.name) for medical_proc in benefits_output.covered]
    benefit_payment_output = pipeline.calculate_benefit_payments(
        target_claim.documents, benefits_to_pay, "benefit_payment"
    )

    with pipeline_status.container():
        st.write("🧠 **Analyzing dates...** ✅")
        st.write("🚫 **Checking exclusions...** ✅")
        st.write("💰 **Mapping benefits...** ✅")
        st.write("💵 **Calculating payments...** ✅")
        st.write("⚖️ **Final decision...** ⏳")
    progress_bar.progress(start_progress + 40)

    # Phase 5: Final Decision
    recommendation_output = pipeline.get_claim_recommendation(
        target_claim.documents, dates_output, exclusions_output, benefits_output, "claim_recommendation"
    )

    # Create final adjudication output
    adjudication_output = AdjudicationOutput(
        policy_id=target_claim.policy_id,
        claim_id=target_claim.claim_id,
        claim_documents=target_claim.documents,
        dates=dates_output,
        exclusions=exclusions_output,
        benefits=benefits_output,
        benefit_payment=benefit_payment_output,
        decision=recommendation_output,
    )

    progress_bar.progress(100)

    # Final status - all complete
    with pipeline_status.container():
        st.write("🎉 **All phases completed successfully!**")

    return adjudication_output
