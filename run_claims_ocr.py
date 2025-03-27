import logging
import sys
import argparse
from datetime import datetime
from claims_ocr_processor import ClaimsOCRProcessor

def setup_logger():
    """Set up and configure logger for the application."""
    # Create logger
    logger = logging.getLogger('claims_ocr')
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(f'logs/claims_ocr_{current_time}.log')

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process claims documents with OCR')
    parser.add_argument('--disable-donut', action='store_true', 
                      help='Disable Donut processor (useful for machines without GPU)')
    parser.add_argument('--disable-llm', action='store_true',
                      help='Disable LLM processor (useful for machines without GPU)')
    parser.add_argument('--input-dir', type=str, default='data/claims/input',
                      help='Directory containing input files')
    parser.add_argument('--output-dir', type=str, default='data/claims/output',
                      help='Directory for output files')
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger()
    logger.info("Starting Claims OCR Processing")

    try:
        # Initialize processor with custom paths
        processor = ClaimsOCRProcessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
        logger.info(f"Initialized processor with input_dir: {processor.input_dir}, output_dir: {processor.output_dir}")
        
        # Process all claims with specified options
        disabled_processors = []
        if args.disable_donut:
            disabled_processors.append("donut")
            logger.info("Donut processor disabled")
        if args.disable_llm:
            disabled_processors.append("llm")
            logger.info("LLM processor disabled")
            
        results = processor.process_all_claims(
            run_all_classifiers=True, 
            run_all_ocr=True,
            disabled_processors=disabled_processors
        )
        logger.info(f"Processed {len(results)} claims")
        
        # Calculate average confidence
        if results:
            # Access the confidence score correctly from the nested dictionary
            results_with_confidence = [r for r in results if 'ocr_results' in r and 'confidence' in r['ocr_results']]
            
            if results_with_confidence:
                avg_confidence = sum(r['ocr_results']['confidence'] for r in results_with_confidence) / len(results_with_confidence)
                logger.info(f"Average confidence score: {avg_confidence:.2f}%")
                
                # Show files with low confidence that might need manual review
                threshold = 70.0  # Set threshold for low confidence
                low_confidence = [r['filename'] for r in results_with_confidence if r['ocr_results']['confidence'] < threshold]
                
                if low_confidence:
                    logger.warning(f"Found {len(low_confidence)} files that may need manual review:")
                    for filename in low_confidence:
                        logger.warning(f"Low confidence file: {filename}")
            else:
                logger.warning("No OCR confidence scores were found in the results")
        else:
            logger.warning("No results were processed")

    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}", exc_info=True)
        raise

    logger.info("Claims OCR Processing completed")

if __name__ == "__main__":
    main() 