"""Results display and export functionality."""

import json

import pandas as pd
import streamlit as st

from assurity_demo.models import AdjudicationOutput

from .utils import format_duration


def display_adjudication_results(result: AdjudicationOutput):
    """Display the adjudication results in a user-friendly format."""
    st.header("📋 Adjudication Results")

    # Basic claim info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Policy ID", result.policy_id)
    with col2:
        st.metric("Claim ID", result.claim_id)

    # Final decision
    decision = result.decision
    decision_color = {"accept": "🟢", "deny": "🔴", "requires_review": "🟡"}.get(decision.decision_recommendation, "⚪")

    st.subheader(f"{decision_color} Final Decision: {decision.decision_recommendation.upper()}")
    st.write(f"**Claimant:** {decision.claimant}")
    st.write(f"**Justification:** {decision.decision_justification}")

    # Detailed breakdown
    st.subheader("📊 Detailed Analysis")

    # Dates analysis
    with st.expander("📅 Dates Analysis"):
        _display_dates_analysis(result.dates)

    # Exclusions analysis
    with st.expander("🚫 Exclusions Analysis"):
        _display_exclusions_analysis(result.exclusions)

    # Benefits analysis
    with st.expander("💰 Benefits Analysis"):
        _display_benefits_analysis(result.benefits)

    # Payment information
    with st.expander("💵 Payment Information"):
        _display_payment_information(result.benefit_payment)


def _display_dates_analysis(dates):
    """Display dates analysis section."""
    col1, col2 = st.columns(2)
    with col1:
        if hasattr(dates, "policy_start_date") and dates.policy_start_date:
            st.write(f"**Policy Start Date:** {dates.policy_start_date}")
        if hasattr(dates, "accident_date") and dates.accident_date:
            st.write(f"**Accident Date:** {dates.accident_date}")
        if hasattr(dates, "treatment_date") and dates.treatment_date:
            st.write(f"**Treatment Date:** {dates.treatment_date}")
    with col2:
        st.write(f"**Recommendation:** {dates.decision_recommendation}")
        if hasattr(dates, "details_for_decision") and dates.details_for_decision:
            st.write(f"**Details:** {dates.details_for_decision}")


def _display_exclusions_analysis(exclusions):
    """Display exclusions analysis section."""
    st.write(f"**Exclusions Apply:** {'Yes' if exclusions.do_any_exclusions_apply else 'No'}")
    st.write(f"**Recommendation:** {exclusions.decision_recommendation}")

    if exclusions.exclusions:
        st.write("**Applicable Exclusions:**")
        for exclusion in exclusions.exclusions:
            st.write(f"• {exclusion}")

    if exclusions.trigger_files:
        st.write("**Evidence Files:**")
        for file in exclusions.trigger_files:
            st.write(f"• {file}")

    st.write(f"**Details:** {exclusions.details}")


def _display_benefits_analysis(benefits):
    """Display benefits analysis section."""
    st.write(f"**Policy Type:** {benefits.policy_type}")

    if benefits.covered:
        st.write("**✅ Covered Procedures:**")
        covered_df = pd.DataFrame(
            [
                {
                    "Procedure": proc.name or "Unnamed",
                    "CPT Codes": ", ".join(map(str, proc.cpt_codes)) if proc.cpt_codes else "None",
                    "HCPCS Codes": ", ".join(proc.hcpcs_codes) if proc.hcpcs_codes else "None",
                    "ICD-10 PCS": ", ".join(proc.icd10_pcs_codes) if proc.icd10_pcs_codes else "None",
                }
                for proc in benefits.covered
            ]
        )
        st.dataframe(covered_df, use_container_width=True)

    if benefits.not_covered:
        st.write("**❌ Not Covered Procedures:**")
        not_covered_df = pd.DataFrame(
            [
                {
                    "Procedure": proc.name or "Unnamed",
                    "CPT Codes": ", ".join(map(str, proc.cpt_codes)) if proc.cpt_codes else "None",
                    "HCPCS Codes": ", ".join(proc.hcpcs_codes) if proc.hcpcs_codes else "None",
                    "ICD-10 PCS": ", ".join(proc.icd10_pcs_codes) if proc.icd10_pcs_codes else "None",
                }
                for proc in benefits.not_covered
            ]
        )
        st.dataframe(not_covered_df, use_container_width=True)

    if benefits.document_quality_issues:
        st.write("**⚠️ Document Quality Issues:**")
        for issue in benefits.document_quality_issues:
            st.write(f"• {issue}")


def _display_payment_information(payment):
    """Display payment information section."""
    # Calculate total payment amount from benefit_payments list
    total_payment = 0
    if hasattr(payment, "benefit_payments") and payment.benefit_payments:
        for benefit_payment in payment.benefit_payments:
            total_payment += benefit_payment.payment_amount

        st.metric("Recommended Payment Amount", f"${total_payment:,.2f}")

        # Show breakdown by benefit
        if len(payment.benefit_payments) > 1:
            st.write("**Payment Breakdown:**")
            payment_df = pd.DataFrame(
                [
                    {"Benefit": bp.benefit.name, "Payment Amount": f"${bp.payment_amount:,.2f}"}
                    for bp in payment.benefit_payments
                ]
            )
            st.dataframe(payment_df, use_container_width=True)
    else:
        st.metric("Recommended Payment Amount", "$0.00")
        st.info("No payment information available")


def _display_historical_result_summary(result: AdjudicationOutput):
    """Display a clean summary of historical adjudication results."""
    # Basic info metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Policy ID", result.policy_id)
    with col2:
        st.metric("Claim ID", result.claim_id)
    with col3:
        st.metric("Documents", len(result.claim_documents))
    with col4:
        # Calculate total payment
        total_payment = 0
        if hasattr(result.benefit_payment, "benefit_payments") and result.benefit_payment.benefit_payments:
            for bp in result.benefit_payment.benefit_payments:
                total_payment += bp.payment_amount
        st.metric("Payment", f"${total_payment:,.2f}")

    # Final decision prominently displayed
    decision = result.decision
    decision_color = {"accept": "🟢", "deny": "🔴", "requires_review": "🟡"}.get(decision.decision_recommendation, "⚪")

    st.markdown(f"### {decision_color} Decision: **{decision.decision_recommendation.upper()}**")

    # Key details in a clean format
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**👤 Claimant:** {decision.claimant}")
    with col2:
        st.write(f"**📊 Exclusions Apply:** {'Yes' if result.exclusions.do_any_exclusions_apply else 'No'}")

    # Brief justification
    st.write("**💭 Decision Justification:**")
    st.write(decision.decision_justification)


def export_results(result_entry) -> str:
    """Export results to JSON string."""
    # Handle both old format (direct result) and new format (dict with timing)
    if isinstance(result_entry, dict):
        result = result_entry["result"]
    else:
        # Old format - direct result object
        result = result_entry

    return json.dumps(result.model_dump(), indent=2, default=str)


def display_processing_history():
    """Display processing history section."""
    if not st.session_state.processing_results:
        return

    st.markdown("---")
    st.header("📚 Processing History")

    history_data = []
    for i, result_entry in enumerate(st.session_state.processing_results):
        # Handle both old format (direct result) and new format (dict with timing)
        if isinstance(result_entry, dict):
            result = result_entry["result"]
            processing_time = result_entry.get("processing_time", 0)
            mode = result_entry.get("mode", "unknown")
        else:
            # Old format - direct result object
            result = result_entry
            processing_time = 0
            mode = "legacy"

        # Calculate total payment amount
        total_payment = 0
        if hasattr(result.benefit_payment, "benefit_payments") and result.benefit_payment.benefit_payments:
            for bp in result.benefit_payment.benefit_payments:
                total_payment += bp.payment_amount

        history_data.append(
            {
                "Index": i + 1,
                "Policy ID": result.policy_id,
                "Claim ID": result.claim_id,
                "Decision": result.decision.decision_recommendation.upper(),
                "Payment Amount": f"${total_payment:,.2f}",
                "Processing Time": format_duration(processing_time) if processing_time > 0 else "N/A",
                "Mode": "🚀 Fast" if mode == "existing_claim" else "❓ Legacy",
            }
        )

    history_df = pd.DataFrame(history_data)

    # Allow selection of previous results
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_index = st.selectbox(
            "Select a previous result to view:",
            range(len(history_data)),
            format_func=lambda x: f"Result {x + 1}: {history_data[x]['Policy ID']} - {history_data[x]['Claim ID']}",
        )
    with col2:
        st.write("")  # spacing
        if st.button("👁️ View Details"):
            st.session_state["show_historical_result"] = selected_index

    # Show historical result in a clean container
    if "show_historical_result" in st.session_state and st.session_state["show_historical_result"] is not None:
        selected_entry = st.session_state.processing_results[st.session_state["show_historical_result"]]

        # Handle both old and new formats
        if isinstance(selected_entry, dict):
            result_to_display = selected_entry["result"]
            processing_time = selected_entry.get("processing_time", 0)
            mode = selected_entry.get("mode", "unknown")
        else:
            result_to_display = selected_entry
            processing_time = 0
            mode = "legacy"

        # Create a clean modal-like container
        st.markdown("---")

        # Header with close button
        header_col1, header_col2 = st.columns([4, 1])
        with header_col1:
            st.subheader(f"📋 Historical Result: {result_to_display.policy_id} - {result_to_display.claim_id}")
        with header_col2:
            if st.button("✖️ Close", key="close_historical"):
                del st.session_state["show_historical_result"]
                st.rerun()

        # Show timing info if available
        if processing_time > 0:
            st.info(f"⏱️ **Processing Time:** {format_duration(processing_time)} ({mode} mode)")

        # Display the result in a contained area
        with st.container():
            _display_historical_result_summary(result_to_display)

            # Detailed sections in expanders
            with st.expander("📅 Dates Analysis", expanded=False):
                _display_dates_analysis(result_to_display.dates)

            with st.expander("🚫 Exclusions Analysis", expanded=False):
                _display_exclusions_analysis(result_to_display.exclusions)

            with st.expander("💰 Benefits Analysis", expanded=False):
                _display_benefits_analysis(result_to_display.benefits)

            with st.expander("💵 Payment Information", expanded=False):
                _display_payment_information(result_to_display.benefit_payment)

        st.markdown("---")

    st.dataframe(history_df, use_container_width=True)

    if st.button("Clear History"):
        st.session_state.processing_results = []
        st.rerun()
