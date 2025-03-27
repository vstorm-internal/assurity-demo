import os
import json
import base64
from typing import Dict, Tuple, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

class LLMDocumentProcessor:
    """Processes documents using GPT-4o's vision capabilities through LangChain."""
    
    def __init__(self, model="gpt-4o"):
        """Initialize with model name."""
        self.model = model
        # Initialize LangChain OpenAI chat model
        self.chat = ChatOpenAI(
            model=model,
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image as base64 string for the vision model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        # Special handling for PDF files
        if image_path.lower().endswith('.pdf'):
            # Use first page of PDF
            from pdf2image import convert_from_path
            pages = convert_from_path(image_path, dpi=200, first_page=1, last_page=1)
            if pages:
                import io
                from PIL import Image
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                pages[0].save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                return base64.b64encode(img_byte_arr.read()).decode('utf-8')
            else:
                raise ValueError(f"Could not convert PDF to image: {image_path}")
        else:
            # For regular image files
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    
    def process_document(self, image_path: str) -> Tuple[Dict[str, Any], float]:
        """
        Process document image with GPT-4o vision capabilities.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (structured_data, confidence)
        """
        try:
            # Encode image to base64
            base64_image = self.encode_image(image_path)
            
            # Create system prompt
            system_message = SystemMessage(
                content=(
                    "You are an expert document classifier and information extractor for insurance claims. "
                    "You will be shown an insurance document image. Your tasks are:\n"
                    "1. Perform OCR to read all visible text in the document\n"
                    "2. Classify the document type (invoice, prescription, medical_report, etc.)\n"
                    "3. Extract all relevant structured information based on the document type\n"
                    "4. Return your analysis as valid JSON"
                )
            )
            
            # Create user message with image
            human_message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "This is a scanned insurance document. Please:\n"
                            "1. Read and extract all text from this image\n"
                            "2. Classify the document type\n"
                            "3. Extract structured information relevant to this document type\n\n"
                            "Return ONLY a valid JSON object with these fields:\n"
                            "- 'document_type': string classification of document type\n"
                            "- 'confidence': float between 0-100 indicating classification confidence\n"
                            "- 'extracted_text': the OCR'd text from the document\n"
                            "- 'extracted_data': object containing structured data extracted from the document"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            )
            
            # Call LangChain with GPT-4o
            response = self.chat.invoke([system_message, human_message])
            
            # Extract and parse JSON from response
            content = response.content
            
            # Find JSON in the response (may be wrapped in markdown code blocks)
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > 0:
                json_content = content[json_start:json_end]
                structured_data = json.loads(json_content)
                
                # Extract confidence
                confidence = float(structured_data.get("confidence", 0))
                
                return structured_data, confidence
            else:
                # Fallback if proper JSON not found
                return {
                    "document_type": "unknown",
                    "confidence": 0,
                    "extracted_data": {"error": "Failed to parse LLM response as JSON"},
                    "extracted_text": ""
                }, 0
                
        except Exception as e:
            print(f"LLM processing error: {str(e)}")
            return {
                "document_type": "unknown", 
                "confidence": 0,
                "extracted_data": {"error": f"LLM API error: {str(e)}"},
                "extracted_text": ""
            }, 0 