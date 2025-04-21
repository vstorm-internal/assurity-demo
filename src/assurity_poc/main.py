import os

from dotenv import load_dotenv
from fire import Fire
from logzero import logger

from assurity_poc.config import get_settings
from assurity_poc.processors.ocr_processor import OCRProcessor
from assurity_poc.utils.claim_utils import is_hcfa1500, is_ub04

settings = get_settings()


def main(file_path: str):
    # Load environment variables
    load_dotenv()

    # Initialize the OCR pipeline
    ocr_processor = OCRProcessor()

    # Check if the image exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Process the image
    try:
        results = ocr_processor.process_image(file_path)

        is_ub04 = is_ub04(file_path)
        is_hcfa1500 = is_hcfa1500(file_path)

        if is_ub04:
            logger.info("UB04")
        elif is_hcfa1500:
            logger.info("HCFA1500")
        elif is_ub04 and is_hcfa1500:
            logger.info("Unknown")
        else:
            logger.info("Unknown")

        # TODO: use structured data
        logger.info("\n=== OCR Results ===")
        for document in results.documents:
            logger.info(
                f"Document {document.structured_data.document_type.document_type}:"
            )

            logger.info("\nExtracted data:")
            logger.info(document.structured_data.model_dump_json(indent=2))
            logger.info("\n")

            logger.info("\nRaw text:")
            logger.info(document.raw_text)

        logger.info(f"GPT: {results['gpt_text']}")
        logger.info(f"Gemini: {results['gemini_text']}")
        logger.info(f"Similarity: {results['similarity']}")

        if results["similarity"]["overall"] > settings.similarity_threshold:
            logger.info("Similarity threshold met")
        else:
            logger.info("Similarity threshold not met")

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    Fire(main)
