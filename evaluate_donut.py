import os
import argparse
import json
from donut_processor import DonutDocumentProcessor

def main():
    parser = argparse.ArgumentParser(description='Evaluate Donut document processor')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing test documents')
    parser.add_argument('--output', type=str, default='donut_evaluation_results.json', help='Path to save results')
    args = parser.parse_args()
    
    # Initialize the processor
    try:
        processor = DonutDocumentProcessor()
        print("Initialized Donut processor successfully")
    except Exception as e:
        print(f"Failed to initialize Donut processor: {str(e)}")
        return
    
    # Get a list of image files
    image_files = []
    for ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'pdf']:
        image_files.extend([os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                          if f.lower().endswith(f'.{ext}')])
    
    # Process each image
    results = []
    for image_path in image_files:
        try:
            print(f"\nProcessing {os.path.basename(image_path)}...")
            
            # Classify document
            doc_type, confidence = processor.classify_document(image_path)
            print(f"  Document type: {doc_type} (confidence: {confidence:.2f}%)")
            
            # Extract information
            extraction_results = processor.extract_information(image_path, doc_type)
            
            # Store results
            results.append({
                'file': os.path.basename(image_path),
                'document_type': doc_type,
                'confidence': confidence,
                'extraction_results': extraction_results
            })
            
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"\nProcessed {len(results)} documents with average confidence {avg_confidence:.2f}%")
        
        # Count by document type
        type_counts = {}
        for r in results:
            doc_type = r['document_type']
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        print("\nDocument type distribution:")
        for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {doc_type}: {count} ({count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main() 