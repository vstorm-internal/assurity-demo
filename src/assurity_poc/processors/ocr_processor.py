import asyncio

import nest_asyncio
from llama_parse import LlamaParse

from assurity_poc.config import get_settings
from assurity_poc.utils.text import compute_text_similarity, preprocess_text

# Apply nest_asyncio for async operations
nest_asyncio.apply()

settings = get_settings()


class OCRProcessor:
    def __init__(self):
        self.gemini_parser = LlamaParse(
            api_key=settings.llamaparse_api_key,
            result_type="markdown",
            verbose=True,
            language="en",
            take_screenshot=True,
            auto_mode_trigger_on_table_in_page=True,
            use_vendor_multimodal_model=True,
            vendor_multimodal_api_key=settings.gemini_api_key,
            vendor_multimodal_model_name="gemini-2.0-flash-001",
        )

        self.gpt_parser = LlamaParse(
            api_key=settings.llamaparse_api_key,
            result_type="text",
            verbose=True,
            language="en",
            take_screenshot=True,
            auto_mode_trigger_on_table_in_page=True,
            use_vendor_multimodal_model=True,
            azure_openai_deployment_name=settings.azure_openai_deployment_name,
            azure_openai_endpoint=settings.azure_openai_endpoint,
            azure_openai_api_version=settings.azure_openai_api_version,
            azure_openai_key=settings.azure_openai_api_key,
        )

    # TODO: split pages
    # def split_pages(self, image_path: str) -> list[str]:
    #     """Split an image into pages."""
    #     image = Image.open(image_path)

    async def extract_text_gemini(self, image_path: str) -> str:
        """Extract text using LlamaParse with Gemini."""
        try:
            result = await self.gemini_parser.aload_data(image_path)
            # Combine content from all pages
            return "\n".join(page.get_content() for page in result)
        except Exception as e:
            print(f"Error in Gemini extraction: {str(e)}")
            return ""

    async def extract_text_gpt(self, image_path: str) -> str:
        """Extract text using LlamaParse with GPT-4o."""
        try:
            result = await self.gpt_parser.aload_data(image_path)
            # Combine content from all pages
            return "\n".join(page.get_content() for page in result)
        except Exception as e:
            print(f"Error in GPT extraction: {str(e)}")
            return ""

    def calculate_similarity(self, text1: str, text2: str) -> dict[str, float]:
        """Calculate the similarity between two texts."""
        return compute_text_similarity(text1, text2)

    async def process_image_async(
        self, image_path: str
    ) -> dict[str, str | dict[str, float]]:
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
