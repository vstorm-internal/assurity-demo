"""Claims loading and browsing functionality."""

from typing import Any, Dict
from pathlib import Path

import pandas as pd
import streamlit as st

from logzero import logger


def load_available_claims():
    """Load available claims from the demo directory structure and any existing OCR outputs."""
    demo_path = Path("res/demo")

    # Check multiple possible OCR output locations
    ocr_path = Path("res/ocr")

    claims = []

    if not demo_path.exists():
        st.warning(f"Demo directory not found: {demo_path}")
        return claims

    try:
        # Iterate through policy directories
        for policy_dir in demo_path.iterdir():
            if policy_dir.is_dir():
                policy_id = policy_dir.name

                # Iterate through claim subdirectories
                for claim_dir in policy_dir.iterdir():
                    if claim_dir.is_dir():
                        claim_id = claim_dir.name

                        # Count PDF files in the claim directory
                        pdf_files = list(claim_dir.glob("*.pdf"))

                        # Check if there's existing OCR output for this claim
                        ocr_json_file = None
                        has_ocr_output = False

                        if ocr_path.exists():
                            potential_file = ocr_path / f"{policy_id}_{claim_id}.json"
                            if potential_file.exists():
                                ocr_json_file = potential_file
                                has_ocr_output = True

                        claim_info = {
                            "policy_id": policy_id,
                            "claim_id": claim_id,
                            "pdf_files": pdf_files,
                            "file_count": len(pdf_files),
                            "has_ocr_output": has_ocr_output,
                            "ocr_json_file": ocr_json_file,
                        }
                        claims.append(claim_info)

    except Exception as e:
        logger.error(f"Error loading claims: {e}")
        st.error(f"Error loading claims: {e}")

    return claims


def display_claim_browser():
    """Display the claim browser interface."""
    st.subheader("🗂️ Browse Available Claims")

    # Show information about OCR discovery
    with st.expander("ℹ️ About OCR Data Discovery"):
        st.markdown("""
        The app searches for existing OCR output files in the following location `res/ocr/`
        
        OCR files are expected to be in JSON format with name `{policy_id}_{claim_id}.json`
        """)

    # Load available claims
    if not st.session_state.available_claims or st.button("🔄 Refresh Claims List"):
        with st.spinner("Loading available claims..."):
            st.session_state.available_claims = load_available_claims()

    if not st.session_state.available_claims:
        st.warning("No claims found in res/demo directory. Please check the directory structure.")
        return None

    # Create a dataframe for display
    claims_df = pd.DataFrame(
        [
            {
                "Policy ID": claim["policy_id"],
                "Claim ID": claim["claim_id"],
                "PDF Files": claim["file_count"],
                "OCR Status": "✅ Available" if claim["has_ocr_output"] else "❌ Not Found",
            }
            for claim in st.session_state.available_claims
        ]
    )

    st.write(f"Found **{len(st.session_state.available_claims)}** available claims:")
    st.dataframe(claims_df, use_container_width=True)

    # Show OCR status summary
    ocr_available_count = sum(1 for claim in st.session_state.available_claims if claim["has_ocr_output"])
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Claims with OCR Data", ocr_available_count)
    with col2:
        st.metric("Claims requiring OCR", len(st.session_state.available_claims) - ocr_available_count)

    if ocr_available_count > 0:
        st.info("💡 Claims with existing OCR data will process much faster!")
    elif len(st.session_state.available_claims) > 0:
        st.warning("⚠️ No existing OCR data found. All processing will require full OCR (slow).")

    # Claim selection
    claim_options = [
        f"{claim['policy_id']} - {claim['claim_id']} ({claim['file_count']} files) {'🚀' if claim['has_ocr_output'] else '🐌'}"
        for claim in st.session_state.available_claims
    ]

    selected_index = st.selectbox(
        "Select a claim to process:",
        range(len(claim_options)),
        format_func=lambda x: claim_options[x],
        key="claim_selector",
        help="🚀 = Fast (OCR available), 🐌 = Slow (OCR required)",
    )

    if selected_index is not None:
        selected_claim = st.session_state.available_claims[selected_index]
        st.session_state.selected_claim = selected_claim
        return selected_claim

    return None


def display_claim_details(claim_info: Dict[str, Any]):
    """Display detailed information about the selected claim."""
    st.subheader(f"📋 Claim Details: {claim_info['policy_id']} - {claim_info['claim_id']}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Policy ID", claim_info["policy_id"])
    with col2:
        st.metric("Claim ID", claim_info["claim_id"])
    with col3:
        st.metric("PDF Files", claim_info["file_count"])
    with col4:
        ocr_status = "✅ Available" if claim_info["has_ocr_output"] else "❌ Missing"
        processing_time = "Fast" if claim_info["has_ocr_output"] else "Slow"
        st.metric("OCR Status", ocr_status, processing_time)

    # OCR information
    if claim_info["has_ocr_output"]:
        st.success("🚀 **Fast Processing Available**")
    else:
        st.warning("🐌 **Slow Processing Required** - OCR will be run on PDF files (this may take several minutes)")

    # Show file list
    with st.expander("📄 Files in this Claim"):
        if claim_info["pdf_files"]:
            st.write(f"**Total Files:** {len(claim_info['pdf_files'])}")

            # Create simple file list with preview
            for i, pdf_file in enumerate(claim_info["pdf_files"]):
                try:
                    file_size = pdf_file.stat().st_size
                    file_type = "Policy Document" if "PolicyPages" in pdf_file.name else "Claim Document"

                    # Create a container for each file
                    with st.container():
                        col1, col2 = st.columns([4, 1])

                        with col1:
                            st.write(f"**{pdf_file.name}**")
                            st.caption(f"{file_type} • {file_size / 1024:.1f} KB")

                        with col2:
                            # Preview button
                            if st.button(
                                "👁️ Preview", key=f"preview_{claim_info['policy_id']}_{claim_info['claim_id']}_{i}"
                            ):
                                st.session_state[f"show_preview_{i}"] = not st.session_state.get(
                                    f"show_preview_{i}", False
                                )

                    # Show preview if requested
                    if st.session_state.get(f"show_preview_{i}", False):
                        _display_pdf_preview(pdf_file, i)

                    # Add spacing between files
                    if i < len(claim_info["pdf_files"]) - 1:
                        st.markdown("<br>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error processing file {pdf_file.name}: {e}")
        else:
            st.warning("No PDF files found in this claim directory.")


def _display_pdf_preview(pdf_file, file_index):
    """Display PDF preview for a given file."""
    with st.container():
        st.markdown("---")

        # Try to show PDF visual preview
        try:
            from pdf2image import convert_from_path

            # Convert first page to image
            images = convert_from_path(
                pdf_file,
                first_page=1,
                last_page=1,
                dpi=150,  # Good quality for preview
            )

            if images:
                st.write(f"**📄 {pdf_file.name}**")

                # Display the first page image
                st.image(images[0], caption=f"Page 1 of {pdf_file.name}", use_container_width=True)

                # Get total page count
                try:
                    import PyPDF2

                    with open(pdf_file, "rb") as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        page_count = len(pdf_reader.pages)
                        if page_count > 1:
                            st.caption(f"📑 This document has {page_count} pages total (showing page 1)")
                except:
                    st.caption("📑 First page preview")

            else:
                st.warning("Could not generate preview image from PDF.")

        except ImportError:
            st.warning("pdf2image not available for preview. Install with: pip install pdf2image")
            st.info("Note: pdf2image also requires poppler-utils system dependency")
        except Exception as e:
            st.error(f"Error generating PDF preview: {str(e)[:100]}...")

        st.markdown("---")
