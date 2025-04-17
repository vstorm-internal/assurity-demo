import os
from pathlib import Path

from dotenv import load_dotenv

from assurity_poc.config import get_settings
from assurity_poc.image_matching import ImageMatcher
from assurity_poc.ocr_processor import OCRProcessor

UB04_BLANK_PATH = Path("./res/image_match/blanks/UB04_blank.png")
HCFA1500_BLANK_PATH = Path("./res/image_match/blanks/HCFA1500_blank.png")


settings = get_settings()


def main():
    # Load environment variables
    load_dotenv()

    # Initialize the OCR pipeline
    ocr_processor = OCRProcessor()
    image_matcher = ImageMatcher()
    # Example image path (replace with your actual image path)
    image_path = "res/data/inputs/jpeg/UB04 Matlock.jpeg"

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    # Process the image
    try:
        results = ocr_processor.process_image(image_path)

        ub04_similarity = image_matcher(image_path, UB04_BLANK_PATH)
        hcfa1500_similarity = image_matcher(image_path, HCFA1500_BLANK_PATH)

        is_ub04 = ub04_similarity["hash_diff"] <= settings.hash_threshold
        is_hcfa1500 = hcfa1500_similarity["hash_diff"] <= settings.hash_threshold

        if is_ub04:
            print("UB04")
        elif is_hcfa1500:
            print("HCFA1500")
        elif is_ub04 and is_hcfa1500:
            print("Unknown")
        else:
            print("Unknown")

        # TODO: use structured data
        # # Print the results
        # print("\n=== OCR Results ===")
        # for document in results.documents:
        #     print(f"Document {document.structured_data.document_type.document_type}:")

        #     print("\nExtracted data:")
        #     print(document.structured_data.model_dump_json(indent=2))
        #     print("\n")

        #     print("\nRaw text:")
        #     print(document.raw_text)

        print(f"GPT: {results['gpt_text']}")
        print(f"Gemini: {results['gemini_text']}")
        print(f"Similarity: {results['similarity']}")

    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    main()
