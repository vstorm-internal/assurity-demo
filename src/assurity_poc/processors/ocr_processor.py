import asyncio
import os

import nest_asyncio
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from llama_parse import LlamaParse

from assurity_poc.models.claim import Output
from assurity_poc.utils import compute_text_similarity, preprocess_text

# Load environment variables
load_dotenv()

# Apply nest_asyncio for async operations
nest_asyncio.apply()


class OCRProcessor:
    def __init__(self):
        # Initialize LlamaParse with Gemini
        self.gemini_parser = LlamaParse(
            api_key=os.getenv("LLAMAPARSE_API_KEY"),
            result_type="text",
            verbose=True,
            language="en",
            take_screenshot=True,
            premium_mode=True,
            auto_mode_trigger_on_table_in_page=True,
            use_vendor_multimodal_model=True,
            vendor_multimodal_api_key=os.getenv("GEMINI_API_KEY"),
            vendor_multimodal_model_name="gemini-2.0-flash-001",
        )

        # Initialize LlamaParse with GPT-4o
        self.gpt_parser = LlamaParse(
            api_key=os.getenv("LLAMAPARSE_API_KEY"),
            result_type="text",
            verbose=True,
            language="en",
            premium_mode=True,
            take_screenshot=True,
            auto_mode_trigger_on_table_in_page=True,
            use_vendor_multimodal_model=True,
            vendor_multimodal_api_key=os.getenv("OPENAI_API_KEY"),
            vendor_multimodal_model_name="openai-gpt4o",
        )

        # Initialize models for text analysis
        self.gemini_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY")
        )

        self.gpt_model = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize Pydantic output parser
        self.parser = PydanticOutputParser(pydantic_object=Output)

        # Define analysis prompt with format instructions
        self.analysis_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert in processing insurance claim forms. 
            Extract and structure the information from the text according to the following format:
            {format_instructions}
            
            Note: The text can sometimes contain multiple documents. Return a list of documents in the specified format.""",
                ),
                ("user", "{text}"),
            ]
        ).partial(format_instructions=self.parser.get_format_instructions())

    async def extract_text_gemini(self, image_path: str) -> str:
        """Extract text using LlamaParse with Gemini."""
        try:
            result = await asyncio.to_thread(self.gemini_parser.load_data, image_path)
            return result[0].get_content()
        except Exception as e:
            print(f"Error in Gemini extraction: {str(e)}")
            return ""

    async def extract_text_gpt(self, image_path: str) -> str:
        """Extract text using LlamaParse with GPT-4o."""
        try:
            result = await asyncio.to_thread(self.gpt_parser.load_data, image_path)
            return result[0].get_content()
        except Exception as e:
            print(f"Error in GPT extraction: {str(e)}")
            return ""

    def analyze_text(self, text: str) -> Output:
        """Analyze the extracted text and return structured output."""
        try:
            # Create the chain using the pipe operator
            chain = self.analysis_prompt | self.gpt_model | self.parser
            # Invoke the chain with the text
            structured_output = chain.invoke({"text": text})

            return structured_output
        except Exception as e:
            print(f"Error in text analysis: {str(e)}")
            return Output(documents=[])

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate the similarity between two texts."""
        return compute_text_similarity(text1, text2)

    async def process_image_async(self, image_path: str) -> dict:
        """Process an image through the complete OCR pipeline asynchronously."""
        # Create and run both extractions in parallel
        gpt_text, gemini_text = await asyncio.gather(
            self.extract_text_gpt(image_path), self.extract_text_gemini(image_path)
        )

        # Preprocess the texts (only if they're not None)
        if gpt_text and gemini_text:
            gpt_text = preprocess_text(gpt_text)
            gemini_text = preprocess_text(gemini_text)

            # Calculate similarity between the texts
            similarity = self.calculate_similarity(gpt_text, gemini_text)
        else:
            similarity = {
                "levenshtein": 0.0,
                "jaccard": 0.0,
                "tfidf": 0.0,
                "embedding": 0.0,
                "overall": 0.0,
            }

        return {
            "gpt_text": gpt_text,
            "gemini_text": gemini_text,
            "similarity": similarity,
        }

    def process_image(self, image_path: str) -> dict:
        """Process an image through the complete OCR pipeline (synchronous wrapper)."""
        return asyncio.run(self.process_image_async(image_path))

    def test_two_different_texts(self, text1: str, text2: str):
        """Test the two texts OCR and return the similarity score."""
        gpt_text = self.extract_text_gpt(text1)
        gemini_text = self.extract_text_gemini(text2)
        similarity = self.calculate_similarity(gpt_text, gemini_text)
        return {
            "gpt_text": gpt_text,
            "gemini_text": gemini_text,
            "similarity": similarity,
        }
