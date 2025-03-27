import os
import argparse
import pandas as pd
from document_classifier import DocumentClassifier

def main():
    parser = argparse.ArgumentParser(description='Train the document classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV file')
    parser.add_argument('--output', type=str, default='models/doc_classifier.joblib', help='Path to save model')
    args = parser.parse_args()
    
    # Load training data
    df = pd.read_csv(args.data)
    
    # Check if required columns exist
    if 'text' not in df.columns or 'document_type' not in df.columns:
        print("Error: Training data must contain 'text' and 'document_type' columns.")
        return
    
    # Initialize and train the classifier
    classifier = DocumentClassifier(model_path=args.output)
    classifier.train(df['text'].tolist(), df['document_type'].tolist())
    
    print(f"Model trained and saved to {args.output}")

if __name__ == "__main__":
    main() 