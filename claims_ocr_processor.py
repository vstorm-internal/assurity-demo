import os
import glob
import json
import cv2
import numpy as np
import pytesseract
from PIL import Image
from datetime import datetime
from pdf2image import convert_from_path
import csv
import torch
# Add Google Cloud Vision imports
from google.cloud import vision
from google.cloud.vision import AnnotateImageRequest
import io

class ClaimsOCRProcessor:
    def __init__(self, input_dir='data/claims/input', output_dir='data/claims/output'):
        """Initialize the OCR processor with input and output directories."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Google Vision client (will be lazy-loaded when needed)
        self._vision_client = None
        
    @property
    def vision_client(self):
        """Lazy-load the Google Vision client when first needed."""
        if self._vision_client is None:
            try:
                self._vision_client = vision.ImageAnnotatorClient()
            except Exception as e:
                print(f"Warning: Failed to initialize Google Vision client: {str(e)}")
                print("Make sure GOOGLE_APPLICATION_CREDENTIALS environment variable is set")
        return self._vision_client
        
    def get_input_files(self):
        """Get list of image files from input directory."""
        extensions = ['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'gif', 'pdf']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(self.input_dir, f'*.{ext}')))
            image_files.extend(glob.glob(os.path.join(self.input_dir, f'*.{ext.upper()}')))
            
        return image_files
    
    def preprocess_image(self, image_path):
        """Apply preprocessing to improve OCR quality."""
        if image_path.lower().endswith('.pdf'):
            # Convert PDF to image
            pages = convert_from_path(image_path)
            # Use first page for now - could be modified to process all pages
            img = np.array(pages[0])
        else:
            # Read image
            img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        threshold = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply dilation to connect broken characters
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(threshold, kernel, iterations=1)
        
        return dilated
    
    def perform_ocr(self, preprocessed_img):
        """Perform OCR on preprocessed image and return text with confidence."""
        # Get OCR data with detailed information including confidence
        ocr_data = pytesseract.image_to_data(preprocessed_img, output_type=pytesseract.Output.DICT)
        
        # Extract text and confidence values
        texts = []
        confidences = []
        
        # Process OCR results
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) > 0:  # Filter empty results
                text = ocr_data['text'][i].strip()
                confidence = int(ocr_data['conf'][i])
                
                if text:  # Only include non-empty text
                    texts.append(text)
                    confidences.append(confidence)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Get full text
        full_text = " ".join(texts)
        
        return full_text, avg_confidence
    
    def extract_structured_data(self, text, doc_type="unknown"):
        """
        Extract structured data from OCR text based on document type.
        This applies different extraction rules based on the document classification.
        """
        # Initialize structured data with common fields
        structured_data = {
            "common": {
                "patient_name": None,
                "service_date": None,
                "provider_name": None,
            },
            "document_specific": {}
        }
        
        # Extract common fields using regex patterns
        import re
        
        # Basic patient name pattern (very simplified)
        patient_match = re.search(r"patient(?:\s*name)?[\s:]+([A-Za-z\s]+)", text, re.IGNORECASE)
        if patient_match:
            structured_data["common"]["patient_name"] = patient_match.group(1).strip()
        
        # Date pattern
        date_match = re.search(r"(?:date|dos)[\s:]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2})", text, re.IGNORECASE)
        if date_match:
            structured_data["common"]["service_date"] = date_match.group(1).strip()
        
        # Provider name pattern
        provider_match = re.search(r"(?:provider|doctor|physician|hospital)[\s:]+([A-Za-z\s,.]+)", text, re.IGNORECASE)
        if provider_match:
            structured_data["common"]["provider_name"] = provider_match.group(1).strip()
        
        # Apply document-specific extraction logic
        if doc_type == "invoice":
            # Extract invoice-specific fields
            structured_data["document_specific"] = self._extract_invoice_data(text)
        
        elif doc_type == "prescription":
            # Extract prescription-specific fields
            structured_data["document_specific"] = self._extract_prescription_data(text)
        
        elif doc_type == "medical_report":
            # Extract medical report fields
            structured_data["document_specific"] = self._extract_medical_report_data(text)
        
        elif doc_type == "lab_result":
            # Extract lab result fields
            structured_data["document_specific"] = self._extract_lab_result_data(text)
        
        # Add more document types as needed
        
        return structured_data

    def _extract_invoice_data(self, text):
        """Extract fields specific to invoices."""
        import re
        
        result = {
            "total_amount": None,
            "invoice_number": None,
            "service_items": [],
            "insurance_paid": None,
            "patient_responsibility": None
        }
        
        # Extract total amount
        total_match = re.search(r"(?:total|amount\s*due|balance\s*due)[\s:]*[$]?(\d+(?:\.\d{2})?)", text, re.IGNORECASE)
        if total_match:
            result["total_amount"] = total_match.group(1).strip()
        
        # Extract invoice/claim number
        invoice_match = re.search(r"(?:invoice|claim)[\s#:]*(\w+[-]?\w+)", text, re.IGNORECASE)
        if invoice_match:
            result["invoice_number"] = invoice_match.group(1).strip()
        
        # Try to extract line items (this is complex and would need refinement)
        service_items = re.findall(r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+([A-Z0-9]+)\s+([^\n$]+)\s+[$]?(\d+\.\d{2})", text)
        for item in service_items:
            result["service_items"].append({
                "date": item[0],
                "code": item[1],
                "description": item[2].strip(),
                "amount": item[3]
            })
        
        return result

    def _extract_prescription_data(self, text):
        """Extract fields specific to prescriptions."""
        import re
        
        result = {
            "medication": None,
            "dosage": None,
            "quantity": None,
            "refills": None,
            "prescriber": None,
            "pharmacy": None
        }
        
        # Extract medication name
        med_match = re.search(r"(?:medication|drug|rx)[\s:]+([A-Za-z0-9\s]+)", text, re.IGNORECASE)
        if med_match:
            result["medication"] = med_match.group(1).strip()
        
        # Extract dosage
        dosage_match = re.search(r"(?:dosage|dose|sig)[\s:]+([^\n]+)", text, re.IGNORECASE)
        if dosage_match:
            result["dosage"] = dosage_match.group(1).strip()
        
        # Extract quantity
        qty_match = re.search(r"(?:quantity|qty)[\s:]+(\d+)", text, re.IGNORECASE)
        if qty_match:
            result["quantity"] = qty_match.group(1).strip()
        
        # Extract refills
        refill_match = re.search(r"(?:refills|refill)[\s:]+(\d+)", text, re.IGNORECASE)
        if refill_match:
            result["refills"] = refill_match.group(1).strip()
        
        return result

    def _extract_medical_report_data(self, text):
        """Extract fields specific to medical reports."""
        # Implementation for medical reports would go here
        return {
            "diagnosis": None,
            "symptoms": None,
            "treatment": None,
            "followup": None
        }

    def _extract_lab_result_data(self, text):
        """Extract fields specific to lab results."""
        # Implementation for lab results would go here
        return {
            "test_name": None,
            "result_value": None,
            "reference_range": None,
            "interpretation": None
        }
    
    def classify_document(self, text, image=None):
        """
        Classify document type based on OCR text and optionally the image.
        Returns document type and confidence score.
        """
        # Define keyword sets for different document types
        document_types = {
            "invoice": ["invoice", "bill", "payment", "amount due", "total due", "balance", "charge"],
            "prescription": ["prescription", "rx", "medication", "pharmacy", "dose", "refill"],
            "medical_report": ["diagnosis", "symptoms", "examination", "assessment", "medical history"],
            "lab_result": ["laboratory", "test results", "blood test", "analysis", "specimen", "reference range"],
            "insurance_card": ["member", "group", "plan", "coverage", "insurance card", "id number"],
            "explanation_of_benefits": ["eob", "explanation of benefits", "claim summary", "benefits", "not a bill"]
        }
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Count keyword matches for each document type
        match_counts = {}
        total_keywords = 0
        
        for doc_type, keywords in document_types.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            match_counts[doc_type] = count
            total_keywords += count
        
        # If no keywords matched, it's unknown
        if total_keywords == 0:
            return "unknown", 0.0
        
        # Find the document type with the most keyword matches
        best_match = max(match_counts.items(), key=lambda x: x[1])
        doc_type = best_match[0]
        
        # Calculate confidence based on proportion of matched keywords
        match_ratio = best_match[1] / sum(match_counts.values())
        confidence = min(match_ratio * 100, 100.0)
        
        return doc_type, confidence
    
    def analyze_document_layout(self, image):
        """
        Analyze document layout to help with classification.
        Returns features that can help determine document type.
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Apply edge detection to identify document structure
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours to identify text blocks and other elements
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count potential text regions/blocks
        text_block_count = len([c for c in contours if cv2.contourArea(c) > 500])
        
        # Check for table structures (common in invoices, EOBs, lab results)
        horizontal_lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=100, 
            minLineLength=width*0.3, maxLineGap=20
        )
        
        has_tables = horizontal_lines is not None and len(horizontal_lines) > 3
        
        # Check for logo-like elements (typically in the top portion)
        top_region = image[0:int(height*0.2), :]
        logo_candidate = len([c for c in contours 
                             if cv2.boundingRect(c)[1] < height*0.2 
                             and cv2.contourArea(c) > 1000]) > 0
        
        return {
            "height_width_ratio": height / width,
            "text_block_count": text_block_count,
            "has_tables": has_tables,
            "has_logo": logo_candidate
        }
    
    def perform_google_vision_ocr(self, image_path):
        """Perform OCR using Google Cloud Vision API."""
        # Ensure we have a valid client
        if not self.vision_client:
            return "", 0
            
        try:
            # Read image file
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
                
            # For PDF files, we need special handling
            if image_path.lower().endswith('.pdf'):
                # Convert first page to image
                pages = convert_from_path(image_path)
                # Use first page for now
                img = pages[0]
                # Save to a temporary buffer
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                content = buf.getvalue()
            
            # Create image object
            image = vision.Image(content=content)
            
            # Perform text detection
            response = self.vision_client.document_text_detection(image=image)
            
            # Extract full text
            full_text = response.full_text_annotation.text
            
            # Calculate confidence by averaging page confidences
            confidences = []
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    confidences.append(block.confidence)
            
            avg_confidence = sum(confidences) / len(confidences) * 100 if confidences else 0
            
            return full_text, avg_confidence
            
        except Exception as e:
            print(f"Google Vision OCR failed: {str(e)}")
            return "", 0
    
    def classify_with_google_vision(self, image_path):
        """
        Use Google Cloud Vision API to detect properties and classify the document.
        """
        # Ensure we have a valid client
        if not self.vision_client:
            return "unknown", 0
            
        try:
            # Read image file
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
                
            # For PDF files, we need special handling
            if image_path.lower().endswith('.pdf'):
                pages = convert_from_path(image_path)
                img = pages[0]
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                content = buf.getvalue()
            
            # Create image object
            image = vision.Image(content=content)
            
            # Perform document text detection and label detection
            text_response = self.vision_client.document_text_detection(image=image)
            label_response = self.vision_client.label_detection(image=image)
            
            # Extract text
            text = text_response.full_text_annotation.text.lower()
            
            # Extract labels
            labels = [label.description.lower() for label in label_response.label_annotations]
            
            # Define keyword maps for document types
            document_types = {
                "invoice": ["invoice", "bill", "payment", "amount due", "total due", "balance", "charge"],
                "prescription": ["prescription", "rx", "medication", "pharmacy", "dose", "refill"],
                "medical_report": ["diagnosis", "symptoms", "examination", "assessment", "medical history"],
                "lab_result": ["laboratory", "test results", "blood test", "analysis", "specimen", "reference range"],
                "insurance_card": ["member", "group", "plan", "coverage", "insurance card", "id number"],
                "explanation_of_benefits": ["eob", "explanation of benefits", "claim summary", "benefits", "not a bill"]
            }
            
            # Score each document type based on text content and labels
            scores = {}
            
            # Check text content against keywords
            for doc_type, keywords in document_types.items():
                score = sum(1 for keyword in keywords if keyword in text)
                
                # Check if any labels match document type keywords
                label_score = sum(1 for label in labels for keyword in keywords if keyword in label)
                
                # Combined score (text matches count more than label matches)
                scores[doc_type] = score * 2 + label_score
            
            # If no significant matches, it's unknown
            if max(scores.values(), default=0) == 0:
                return "unknown", 0
            
            # Find the document type with the highest score
            best_match = max(scores.items(), key=lambda x: x[1])
            doc_type = best_match[0]
            
            # Calculate confidence based on relative score
            total_score = sum(scores.values())
            confidence = (best_match[1] / total_score) * 100 if total_score > 0 else 0
            
            return doc_type, confidence
            
        except Exception as e:
            print(f"Google Vision classification failed: {str(e)}")
            return "unknown", 0
    
    def process_all_claims(self, run_all_classifiers=False, run_all_ocr=False, disabled_processors=None):
        """Process all claim images in the input directory.
        
        Args:
            run_all_classifiers (bool): If True, run all available classifiers for each image 
                                        regardless of confidence levels. If False, use the
                                        sequential pipeline approach (default behavior).
            run_all_ocr (bool): If True, run all available OCR mechanisms for each image and
                               combine their results. If False, use only the OCR mechanism
                               with highest confidence (default behavior).
            disabled_processors (list): List of processor names to disable (e.g., ["donut", "llm"])
        """
        if disabled_processors is None:
            disabled_processors = []
        
        image_files = self.get_input_files()
        results = []
        
        # Check if GPU is available
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("No GPU detected. Running on CPU only.")
        
        # Initialize classifiers
        classifiers_available = {
            "ml_text": False,
            "vision": False,
            "donut": False,
            "llm": False,
            "google_vision": self.vision_client is not None  # Check if Google Vision is available
        }
        
        # Try to initialize ML text classifier
        try:
            from document_classifier import DocumentClassifier
            text_classifier = DocumentClassifier()
            classifiers_available["ml_text"] = True
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Text classifier not available: {str(e)}")
        
        # Try to initialize vision classifier
        try:
            from document_vision_classifier import DocumentVisionClassifier
            vision_classifier = DocumentVisionClassifier()
            classifiers_available["vision"] = True
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Vision classifier not available: {str(e)}")
        
        # Try to initialize Donut processor if not disabled
        if "donut" not in disabled_processors:
            try:
                from donut_processor import DonutDocumentProcessor
                donut_processor = DonutDocumentProcessor()
                classifiers_available["donut"] = True
            except (ImportError, ModuleNotFoundError) as e:
                print(f"Donut processor not available: {str(e)}")
        else:
            print("Donut processor disabled by user")
        
        # Try to initialize LLM processor if not disabled
        if "llm" not in disabled_processors:
            try:
                from llm_processor import LLMDocumentProcessor
                llm_processor = LLMDocumentProcessor()
                classifiers_available["llm"] = True
            except (ImportError, ModuleNotFoundError) as e:
                print(f"LLM processor not available: {str(e)}")
        else:
            print("LLM processor disabled by user")
        
        for image_path in image_files:
            try:
                # Get filename for the result
                filename = os.path.basename(image_path)
                
                # Get original image for vision-based classification and layout analysis
                original_img = None
                if image_path.lower().endswith('.pdf'):
                    pages = convert_from_path(image_path)
                    original_img = np.array(pages[0])
                else:
                    original_img = cv2.imread(image_path)
                
                # Track classification attempts and extraction results
                classification_results = []
                extraction_sources = {}
                ocr_results = {}
                
                # Preprocess image for OCR
                preprocessed = self.preprocess_image(image_path)
                
                # Track OCR results from different methods
                ocr_texts = {}
                
                # Perform OCR using Tesseract
                tesseract_text, tesseract_confidence = self.perform_ocr(preprocessed)
                ocr_texts["tesseract"] = {
                    "text": tesseract_text,
                    "confidence": tesseract_confidence
                }
                
                # Default to Tesseract OCR
                text = tesseract_text
                ocr_confidence = tesseract_confidence
                ocr_method_used = "tesseract"
                
                # Add Google Vision OCR if available
                if classifiers_available["google_vision"]:
                    try:
                        google_vision_text, google_vision_ocr_confidence = self.perform_google_vision_ocr(image_path)
                        ocr_texts["google_vision"] = {
                            "text": google_vision_text,
                            "confidence": google_vision_ocr_confidence
                        }
                        
                        # If running all OCR, collect result, otherwise use the best one
                        if not run_all_ocr:
                            # If Google Vision OCR has better confidence, use it instead
                            if google_vision_ocr_confidence > ocr_confidence:
                                text = google_vision_text
                                ocr_confidence = google_vision_ocr_confidence
                                ocr_method_used = "google_vision"
                                print(f"Using Google Vision OCR for {filename} (confidence: {ocr_confidence:.2f}%)")
                    except Exception as e:
                        print(f"Google Vision OCR failed for {filename}: {str(e)}")
                
                # Add LLM-based OCR if available and we're running all OCR mechanisms
                llm_text = ""
                if classifiers_available["llm"] and (run_all_ocr or ocr_confidence < 50):
                    try:
                        # We'll utilize the text extraction from LLM processor if it's already going to be called
                        # This will be filled in later when we call the LLM processor
                        pass
                    except Exception as e:
                        print(f"LLM OCR failed for {filename}: {str(e)}")
                
                # If running all OCR mechanisms, combine the results
                if run_all_ocr and len(ocr_texts) > 1:
                    # Option 1: Use the text with highest confidence
                    best_ocr = max(ocr_texts.items(), key=lambda x: x[1]["confidence"])
                    text = best_ocr[1]["text"]
                    ocr_confidence = best_ocr[1]["confidence"]
                    ocr_method_used = best_ocr[0]
                    print(f"Using {ocr_method_used} OCR for {filename} (highest confidence: {ocr_confidence:.2f}%)")
                    
                    # Store all OCR results for reference
                    ocr_results = {
                        "method_used": ocr_method_used,
                        "confidence": ocr_confidence,
                        "all_results": ocr_texts
                    }
                else:
                    # Store the single OCR result that was used
                    ocr_results = {
                        "method_used": ocr_method_used,
                        "confidence": ocr_confidence,
                        "text": text
                    }
                
                # 1. Google Vision-based classification if available
                if classifiers_available["google_vision"]:
                    try:
                        doc_type, confidence = self.classify_with_google_vision(image_path)
                        classification_results.append({
                            "method": "google_vision",
                            "doc_type": doc_type,
                            "confidence": confidence
                        })
                    except Exception as e:
                        print(f"Google Vision classification failed for {filename}: {str(e)}")
                
                # 2. Vision-based classification if available
                if classifiers_available["vision"]:
                    try:
                        doc_type, confidence = vision_classifier.classify(image_path)
                        classification_results.append({
                            "method": "vision_model",
                            "doc_type": doc_type,
                            "confidence": confidence
                        })
                    except Exception as e:
                        print(f"Vision classification failed for {filename}: {str(e)}")
                
                # 3. Donut-based classification if available
                if classifiers_available["donut"]:
                    try:
                        donut_results, donut_confidence = donut_processor.classify_document(image_path)
                        classification_results.append({
                            "method": "donut_model",
                            "doc_type": donut_results,
                            "confidence": donut_confidence
                        })
                    except Exception as e:
                        print(f"Donut classification failed for {filename}: {str(e)}")
                
                # 4. ML text-based classification if available
                if classifiers_available["ml_text"] and text:
                    try:
                        doc_type, confidence = text_classifier.predict(text)
                        classification_results.append({
                            "method": "ml_text",
                            "doc_type": doc_type, 
                            "confidence": confidence
                        })
                    except Exception as e:
                        print(f"ML text classification failed for {filename}: {str(e)}")
                
                # 5. Rule-based classification
                rule_type, rule_confidence = self.classify_document(text)
                classification_results.append({
                    "method": "rule_based",
                    "doc_type": rule_type,
                    "confidence": rule_confidence
                })
                
                # Find the best classification so far
                best_classification = max(classification_results, key=lambda x: x["confidence"])
                best_confidence = best_classification["confidence"]
                
                # Prepare variables for extraction data
                donut_extraction_data = None
                llm_data = None
                
                # If running all classifiers or confidence is low, use Donut extraction
                ADVANCED_THRESHOLD = 70.0  # Use advanced extraction when confidence below this
                
                # Extract data using rule-based method
                rule_based_data = self.extract_structured_data(text, best_classification["doc_type"])
                extraction_sources["rule_based"] = rule_based_data
                
                # 5. Run advanced classifiers for extraction when confidence is low OR run_all_classifiers is True
                if run_all_classifiers or best_confidence < ADVANCED_THRESHOLD:
                    # 5.1 Try Donut extraction
                    if classifiers_available["donut"]:
                        try:
                            action_reason = "all classifiers mode" if run_all_classifiers else f"low confidence ({best_confidence:.2f}%)"
                            print(f"Using Donut extraction for {filename} ({action_reason})")
                            
                            # Use the best document type so far for specialized extraction
                            donut_results = donut_processor.extract_information(image_path, best_classification["doc_type"])
                            
                            # Store extraction results
                            donut_extraction_data = donut_results.get("parsed_data", {})
                            extraction_sources["donut"] = donut_extraction_data
                            
                        except Exception as e:
                            print(f"Donut extraction failed for {filename}: {str(e)}")
                    
                    # 5.2 Try LLM vision
                    if classifiers_available["llm"]:
                        try:
                            action_reason = "all classifiers mode" if run_all_classifiers else f"low confidence ({best_confidence:.2f}%)"
                            print(f"Using LLM vision for {filename} ({action_reason})")
                            
                            # Check if the file is a TIFF/TIF format and convert if needed
                            llm_image_path = image_path
                            temp_png_path = None
                            if image_path.lower().endswith(('.tiff', '.tif')):
                                # Create temp directory if it doesn't exist
                                temp_dir = os.path.join(self.output_dir, 'temp')
                                os.makedirs(temp_dir, exist_ok=True)
                                
                                # Generate temp PNG path
                                base_name = os.path.splitext(os.path.basename(image_path))[0]
                                temp_png_path = os.path.join(temp_dir, f"{base_name}.png")
                                
                                # Convert TIFF to PNG
                                try:
                                    img = Image.open(image_path)
                                    img.save(temp_png_path, 'PNG')
                                    llm_image_path = temp_png_path
                                    print(f"Converted TIFF to PNG for LLM processing: {temp_png_path}")
                                except Exception as convert_err:
                                    print(f"Failed to convert TIFF to PNG: {str(convert_err)}")
                            
                            # Call LLM processor with the image path (converted if needed)
                            llm_response, llm_confidence = llm_processor.process_document(llm_image_path)
                            
                            # Clean up temp file if created
                            if temp_png_path and os.path.exists(temp_png_path):
                                try:
                                    os.remove(temp_png_path)
                                except Exception as cleanup_err:
                                    print(f"Failed to remove temporary PNG: {str(cleanup_err)}")
                            
                            # Add to classification results
                            classification_results.append({
                                "method": "llm_vision",
                                "doc_type": llm_response.get("document_type", "unknown"),
                                "confidence": llm_confidence
                            })
                            
                            # Store full LLM response for later use
                            llm_data = llm_response.get("extracted_data", {})
                            extraction_sources["llm_vision"] = llm_data
                            
                            # Store the text extracted by the LLM and add to OCR results if running all OCR
                            llm_text = llm_response.get("extracted_text", "")
                            if run_all_ocr and llm_text:
                                # Estimate LLM OCR confidence (using a default if not provided)
                                llm_ocr_confidence = llm_response.get("text_confidence", 75.0)
                                ocr_texts["llm"] = {
                                    "text": llm_text,
                                    "confidence": llm_ocr_confidence
                                }
                                
                                # Reconsider the best OCR text
                                best_ocr = max(ocr_texts.items(), key=lambda x: x[1]["confidence"])
                                if best_ocr[0] == "llm":
                                    text = llm_text
                                    ocr_confidence = llm_ocr_confidence
                                    ocr_method_used = "llm"
                                    print(f"Using LLM OCR for {filename} (confidence: {llm_ocr_confidence:.2f}%)")
                            
                            # Re-evaluate best classification including LLM
                            best_classification = max(classification_results, key=lambda x: x["confidence"])
                            
                        except Exception as e:
                            print(f"LLM vision processing failed for {filename}: {str(e)}")
                
                # Use the result with highest confidence
                doc_type = best_classification["doc_type"]
                classification_confidence = best_classification["confidence"]
                classification_method = best_classification["method"]
                
                # Extract layout features
                layout_features = self.analyze_document_layout(original_img)
                
                # Initialize structured data
                structured_data = {}
                
                # If we have LLM text but used our OCR text previously, include the LLM's OCR text too
                if llm_text and not text.strip():
                    text = llm_text
                
                # Select the primary structured data based on priority
                # Default to using the best available data source
                if "llm_vision" in extraction_sources and extraction_sources["llm_vision"]:
                    structured_data = extraction_sources["llm_vision"]
                    primary_extraction = "llm_vision"
                elif "donut" in extraction_sources and extraction_sources["donut"]:
                    structured_data = extraction_sources["donut"]
                    primary_extraction = "donut"
                else:
                    structured_data = extraction_sources["rule_based"]
                    primary_extraction = "rule_based"
                
                # Create result object
                result = {
                    "filename": filename,
                    "timestamp": datetime.now().isoformat(),
                    "ocr_results": ocr_results,
                    "raw_text": text,
                    "document_type": {
                        "type": doc_type,
                        "confidence": classification_confidence,
                        "method": classification_method,
                        "all_attempts": classification_results
                    },
                    "layout_features": layout_features,
                    "structured_data": structured_data,
                    "primary_extraction_method": primary_extraction,
                    "all_extraction_results": extraction_sources
                }
                
                # Add to results list
                results.append(result)
                
                # Save individual result to JSON file
                output_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}.json")
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"Processed {filename}: {doc_type} ({classification_method}, confidence: {classification_confidence:.2f}%)")
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        # Save batch results
        batch_output_path = os.path.join(self.output_dir, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(batch_output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_suffix = "all_classifiers" if run_all_classifiers else "sequential_pipeline"
        if run_all_ocr:
            mode_suffix = f"{mode_suffix}_all_ocr" if mode_suffix else "all_ocr"
        csv_output_path = os.path.join(self.output_dir, f"classification_summary_{mode_suffix}_{timestamp}.csv")
        
        # Define possible classification methods
        methods = ["vision_model", "ml_text", "rule_based", "llm_vision", "donut_model"]
        
        with open(csv_output_path, 'w', newline='') as csvfile:
            # Create column headers
            fieldnames = ['filename']
            for method in methods:
                fieldnames.extend([f"{method}_doc_type", f"{method}_confidence"])
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data rows
            for result in results:
                row = {'filename': result['filename']}
                
                # Initialize all method columns with empty values
                for method in methods:
                    row[f"{method}_doc_type"] = ""
                    row[f"{method}_confidence"] = ""
                
                # Fill in actual values for methods that were used
                for classification in result['document_type']['all_attempts']:
                    method = classification['method']
                    if method in methods:
                        row[f"{method}_doc_type"] = classification['doc_type']
                        row[f"{method}_confidence"] = f"{classification['confidence']:.2f}"
                
                writer.writerow(row)
        
        print(f"Classification summary saved to {csv_output_path}")
            
        return results

if __name__ == "__main__":
    processor = ClaimsOCRProcessor()
    results = processor.process_all_claims()
    print(f"Processed {len(results)} claim images") 