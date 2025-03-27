import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

class DocumentClassifier:
    def __init__(self, model_path="models/doc_classifier.joblib"):
        """Initialize the document classifier."""
        self.model_path = model_path
        self.model_dir = os.path.dirname(model_path)
        self.pipeline = None
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # Load model if it exists
        if os.path.exists(model_path):
            self.pipeline = joblib.load(model_path)
    
    def train(self, texts, labels):
        """
        Train a document classifier on text samples.
        
        Args:
            texts: List of document text strings
            labels: List of corresponding document type labels
        """
        # Create a pipeline with TF-IDF features and a Random Forest classifier
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train the model
        self.pipeline.fit(texts, labels)
        
        # Save the model
        joblib.dump(self.pipeline, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def predict(self, text):
        """
        Predict document type from document text.
        
        Args:
            text: Document text string
            
        Returns:
            predicted_class: String representing document class
            confidence: Prediction confidence score (0-100)
        """
        if self.pipeline is None:
            return "unknown", 0.0
        
        # Get prediction
        predicted_class = self.pipeline.predict([text])[0]
        
        # Get prediction probability
        probabilities = self.pipeline.predict_proba([text])[0]
        confidence = float(np.max(probabilities) * 100)
        
        return predicted_class, confidence 