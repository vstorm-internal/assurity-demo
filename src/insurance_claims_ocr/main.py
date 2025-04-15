import os

from dotenv import load_dotenv

from insurance_claims_ocr.ocr_pipeline import OCRPipeline


def main():
    # Load environment variables
    load_dotenv()

    # Initialize the OCR pipeline
    pipeline = OCRPipeline()

    # Example image path (replace with your actual image path)
    image_path = "res/data/inputs/jpeg/UB04 Matlock.jpeg"

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Process the image
    try:
        results = pipeline.process_image(image_path)

        # Print the results
        print("\n=== OCR Results ===")
        for document in results.documents:
            print(f"Document {document.structured_data.document_type.document_type}:")

            print("\nExtracted data:")
            print(document.structured_data.model_dump_json(indent=2))
            print("\n")

            print("\nRaw text:")
            print(document.raw_text)

    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    main()
