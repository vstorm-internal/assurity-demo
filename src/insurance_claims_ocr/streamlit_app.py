import streamlit as st
import json
from pathlib import Path
import pandas as pd


def load_ocr_results():
    """Load OCR results from Results_OCR2.json file."""
    results_file = Path("data/results/results.json")
    if not results_file.exists():
        st.error(f"Results file {results_file} does not exist!")
        return None

    try:
        with open(results_file, "r") as f:
            results = json.load(f)
        return results
    except Exception as e:
        st.error(f"Error loading results file: {str(e)}")
        return None


def main():
    st.title("OCR Results Comparison")
    st.write("Compare OCR results from different methods for your documents")

    # Load OCR results
    results = load_ocr_results()
    if not results:
        return

    # Get list of files from results
    files = list(results.keys())
    if not files:
        st.error("No results found in the file!")
        return

    # File selection
    selected_file = st.selectbox(
        "Select a file to view results", options=files, format_func=lambda x: x
    )

    if selected_file:
        file_results = results[selected_file]
        st.subheader(f"Results for: {selected_file}")

        # Get all OCR models from the first file's results
        ocr_models = list(file_results.keys())

        # Create columns for results (2 columns per row)
        num_columns = 2
        for i in range(0, len(ocr_models), num_columns):
            cols = st.columns(num_columns)
            for j, model in enumerate(ocr_models[i : i + num_columns]):
                with cols[j]:
                    st.subheader(f"{model} OCR")
                    try:
                        model_text = file_results[model]
                        st.text_area(f"{model} Result", model_text, height=300)
                    except Exception as e:
                        st.error(f"Error displaying {model} results: {str(e)}")

        # Add a section for comparing results
        st.subheader("Comparison")

        # Create a DataFrame for comparison
        comparison_data = {
            "Method": ocr_models,
            "Text Length": [len(file_results.get(model, "")) for model in ocr_models],
        }

        df = pd.DataFrame(comparison_data)
        st.dataframe(df)

        # Display additional information
        # st.subheader("Additional Information")

        # Document type information
        # doc_type_info = file_results.get('document_type', {})
        # st.write("Document Type:", doc_type_info.get('type', 'Unknown'))
        # st.write("Classification Confidence:", f"{doc_type_info.get('confidence', 0):.1f}%")
        # st.write("Classification Method:", doc_type_info.get('method', 'Unknown'))

        # Layout features
        # layout_features = file_results.get('layout_features', {})
        # if layout_features:
        #     st.write("Layout Features:")
        #     for feature, value in layout_features.items():
        #         st.write(f"- {feature}: {value}")


if __name__ == "__main__":
    main()
