import os
import torch
import logging
from typing import Dict, Tuple, List, Any
from PIL import Image
from pdf2image import convert_from_path
from transformers import DonutProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)

class DonutDocumentProcessor:
    """Document processor using ClovaAI's Donut model for extraction and classification."""
    
    def __init__(self, 
                 model_name="naver-clova-ix/donut-base-finetuned-cord-v2",
                 classification_model_name="naver-clova-ix/donut-base-finetuned-rvlcdip"):
        """
        Initialize the Donut document processor.
        
        Args:
            model_name: Model checkpoint for extraction tasks
            classification_model_name: Model checkpoint for classification tasks
        """
        # Load extraction model and processor
        logger.info(f"Loading Donut extraction model: {model_name}")
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Load classification model
        logger.info(f"Loading Donut classification model: {classification_model_name}")
        self.classification_processor = DonutProcessor.from_pretrained(classification_model_name)
        self.classification_model = VisionEncoderDecoderModel.from_pretrained(classification_model_name)
        
        # Define document types mapping
        self.document_types = {
            "letter": "correspondence",
            "form": "claim_form", 
            "email": "correspondence",
            "handwritten": "handwritten_note",
            "advertisement": "marketing_material",
            "scientific_report": "medical_report",
            "scientific_publication": "medical_report",
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
        
        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.classification_model.to(self.device)
        
        # Log the device being used
        if self.device.type == "cuda":
            logger.info(f"Using GPU for Donut processing: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        else:
            logger.info("Using CPU for Donut processing (GPU not available)")
        
    def load_document_image(self, image_path: str) -> Image.Image:
        """
        Load document image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        if image_path.lower().endswith('.pdf'):
            # Convert PDF to image (first page)
            pages = convert_from_path(image_path, first_page=1, last_page=1)
            if pages:
                return pages[0]
            else:
                raise ValueError(f"Failed to convert PDF to image: {image_path}")
        else:
            # Load image directly
            return Image.open(image_path).convert("RGB")
    
    def classify_document(self, image_path: str) -> Tuple[str, float]:
        """
        Classify document type using Donut model.
        
        Args:
            image_path: Path to document image
            
        Returns:
            Tuple of (document_type, confidence)
        """
        try:
            # Load image
            image = self.load_document_image(image_path)
            
            # Prepare image for the model
            pixel_values = self.classification_processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Task token for classification
            task_prompt = "<s_rvlcdip>"
            decoder_input_ids = self.classification_processor.tokenizer(task_prompt, 
                                                         add_special_tokens=False, 
                                                         return_tensors="pt").input_ids.to(self.device)
            
            # Generate prediction
            outputs = self.classification_model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=20,
                early_stopping=True,
                pad_token_id=self.classification_processor.tokenizer.pad_token_id,
                eos_token_id=self.classification_processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=5,
                bad_words_ids=[[self.classification_processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Get prediction text
            prediction = self.classification_processor.tokenizer.batch_decode(outputs.sequences)[0]
            prediction = prediction.replace(task_prompt, "").replace("</s>", "").strip()
            
            # Calculate confidence score from beam scores
            confidence = torch.exp(outputs.sequences_scores[0]).item() * 100
            
            # Map to our document types
            document_type = self.document_types.get(prediction.lower(), "unknown")
            
            logger.info(f"Donut classification: {prediction} -> {document_type} (confidence: {confidence:.2f}%)")
            return document_type, confidence
            
        except Exception as e:
            logger.error(f"Error in Donut document classification: {str(e)}")
            return "unknown", 0.0
    
    def extract_information(self, image_path: str, doc_type: str = None) -> Dict[str, Any]:
        """
        Extract structured information from document using Donut.
        
        Args:
            image_path: Path to document image
            doc_type: Document type for task-specific extraction
            
        Returns:
            Dictionary with extracted information
        """
        try:
            # Convert TIFF to PNG if needed
            temp_png_path = None
            if image_path.lower().endswith(('.tiff', '.tif')):
                try:
                    # Create temp directory
                    import os
                    temp_dir = os.path.dirname(image_path)
                    os.makedirs(os.path.join(temp_dir, 'temp'), exist_ok=True)
                    
                    # Generate temp PNG path
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    temp_png_path = os.path.join(temp_dir, 'temp', f"{base_name}.png")
                    
                    # Convert TIFF to PNG
                    img = Image.open(image_path)
                    img.save(temp_png_path, 'PNG')
                    logger.info(f"Converted TIFF to PNG for Donut processing: {temp_png_path}")
                    
                    # Use the PNG path instead
                    image_path = temp_png_path
                except Exception as e:
                    logger.warning(f"Failed to convert TIFF to PNG: {str(e)}")
                    # Continue with original path if conversion fails
            
            # Load image
            image = self.load_document_image(image_path)
            
            # Prepare image for the model
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            # Set task token based on document type
            # Different tasks might use different prompts
            task_prompt = self._get_task_prompt(doc_type)
            
            # Generate prediction
            decoder_input_ids = self.processor.tokenizer(task_prompt, 
                                                 add_special_tokens=False, 
                                                 return_tensors="pt").input_ids.to(self.device)
            
            # outputs = self.model.generate(
            #     pixel_values,
            #     decoder_input_ids=decoder_input_ids,
            #     max_length=self.processor.tokenizer.model_max_length,
            #     early_stopping=True,
            #     pad_token_id=self.processor.tokenizer.pad_token_id,
            #     eos_token_id=self.processor.tokenizer.eos_token_id,
            #     use_cache=True,
            #     num_beams=5,
            #     bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            #     return_dict_in_generate=True
            # )
            
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,            
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True
            )
            # Decode output tokens to text
            prediction = self.processor.tokenizer.batch_decode(outputs.sequences)[0]
            prediction = prediction.replace(task_prompt, "").replace("</s>", "").strip()
            
            # Parse structured output based on the model format
            # The exact parsing will depend on the model's output format
            parsed_data = self._parse_model_output(prediction, doc_type)
            
            # Clean up temp file if created
            if temp_png_path and os.path.exists(temp_png_path):
                try:
                    os.remove(temp_png_path)
                except Exception as cleanup_err:
                    logger.warning(f"Failed to remove temporary PNG: {str(cleanup_err)}")
            
            return {
                "raw_prediction": prediction,
                "parsed_data": parsed_data
            }
            
        except Exception as e:
            logger.error(f"Error in Donut information extraction: {str(e)}")
            return {
                "error": str(e),
                "raw_prediction": "",
                "parsed_data": {}
            }
    
    def _get_task_prompt(self, doc_type: str) -> str:
        """
        Get the appropriate task prompt based on document type.
        
        Args:
            doc_type: Document type
            
        Returns:
            Task prompt string
        """
        # Map document types to appropriate task prompts
        # This will vary depending on the specific Donut models you're using
        if doc_type == "invoice":
            return "<s_cord>"  # For receipt/invoice parsing with CORD model
        elif doc_type == "form":
            return "<s_docvqa>"  # For document VQA tasks
        else:
            # Default to general document parsing
            return "<s_cord>"
    
    def _parse_model_output(self, output: str, doc_type: str) -> Dict[str, Any]:
        """
        Parse the model's output into structured data.
        
        Args:
            output: Raw model output text
            doc_type: Document type for type-specific parsing
            
        Returns:
            Structured data dictionary
        """
        # Donut typically outputs in a JSON-like format with special tokens
        # The exact parsing logic depends on the model's output format
        import json
        import re
        
        # Try to convert to proper JSON by fixing common issues
        # Replace OCR errors in JSON structure
        cleaned_output = output
        
        # Fix missing quotes around keys
        cleaned_output = re.sub(r'(\w+):', r'"\1":', cleaned_output)
        
        # Ensure quotes are consistent
        cleaned_output = cleaned_output.replace("'", '"')
        
        try:
            # Try to parse as JSON
            if cleaned_output.startswith("{") and cleaned_output.endswith("}"):
                data = json.loads(cleaned_output)
                return data
            else:
                # If not valid JSON, return as plain text
                return {"text": cleaned_output}
        except json.JSONDecodeError:
            # If JSON parsing fails, extract key-value pairs using regex
            results = {}
            pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
            matches = re.findall(pattern, cleaned_output)
            
            for key, value in matches:
                results[key] = value
            
            return results if results else {"text": cleaned_output}
    
    def process_document(self, image_path: str) -> Tuple[Dict[str, Any], float]:
        """
        Process document with both classification and information extraction.
        
        Args:
            image_path: Path to document image
            
        Returns:
            Tuple of (results_dict, confidence)
        """
        # First classify the document
        doc_type, confidence = self.classify_document(image_path)
        
        # Then extract information based on the document type
        extraction_results = self.extract_information(image_path, doc_type)
        
        # Combine results
        results = {
            "document_type": doc_type,
            "confidence": confidence,
            "extracted_data": extraction_results["parsed_data"],
            "raw_model_output": extraction_results.get("raw_prediction", "")
        }
        
        return results, confidence 