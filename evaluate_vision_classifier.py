import os
import argparse
from document_vision_classifier import DocumentVisionClassifier

def main():
    parser = argparse.ArgumentParser(description='Evaluate document vision classifier')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing test documents')
    args = parser.parse_args()
    
    # Initialize the classifier
    try:
        classifier = DocumentVisionClassifier()
        print("Initialized vision classifier successfully")
    except Exception as e:
        print(f"Failed to initialize vision classifier: {str(e)}")
        return
    
    # Get a list of image files
    image_files = []
    for ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'pdf']:
        image_files.extend([os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                          if f.lower().endswith(f'.{ext}')])
    
    # Evaluate each image
    results = []
    for image_path in image_files:
        try:
            doc_type, confidence = classifier.classify(image_path)
            results.append({
                'file': os.path.basename(image_path),
                'type': doc_type,
                'confidence': confidence
            })
            print(f"{os.path.basename(image_path)}: {doc_type} (confidence: {confidence:.2f}%)")
        except Exception as e:
            print(f"Error classifying {os.path.basename(image_path)}: {str(e)}")
    
    # Print summary
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        print(f"\nProcessed {len(results)} documents with average confidence {avg_confidence:.2f}%")
        
        # Count by document type
        type_counts = {}
        for r in results:
            doc_type = r['type']
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        print("\nDocument type distribution:")
        for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {doc_type}: {count} ({count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main() 