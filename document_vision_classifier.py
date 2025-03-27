from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image
import logging

logger = logging.getLogger(__name__)
class DocumentVisionClassifier:
    def __init__(self, model_name="microsoft/dit-base-finetuned-rvlcdip"):
        """Initialize document vision classifier with a pre-trained model."""
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # Map RVLCDIP classes to insurance document types
        self.class_mapping = {
            "letter": "correspondence",
            "form": "claim_form",
            "email": "correspondence",
            "handwritten": "handwritten_note",
            "advertisement": "marketing_material",
            "scientific_report": "medical_report",
            "scientific_publication": "medical_literature",
            "specification": "policy_document",
            "file_folder": "file_folder",
            "news_article": "newsletter",
            "budget": "financial_statement",
            "invoice": "invoice",
            "presentation": "presentation",
            "questionnaire": "questionnaire",
            "resume": "resume",
            "memo": "memo"
        }
    
    def classify(self, image_path):
        """
        Classify document image using vision transformer.
        
        Args:
            image_path: Path to image file
            
        Returns:
            document_type: Classified document type
            confidence: Classification confidence
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Extract features
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get predicted class and confidence
        predicted_class_idx = logits.argmax(-1).item()
        confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item() * 100
        
        # Get original model class
        original_class = self.model.config.id2label[predicted_class_idx]
        
        # Map to our document types
        document_type = self.class_mapping.get(original_class, "unknown")
        logger.info(f"Predicted class: {original_class}, Document type: {document_type}, Confidence: {confidence}")
        return document_type, confidence 