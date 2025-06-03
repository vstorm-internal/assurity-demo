import os

from enum import Enum
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class Prompts(Enum):
    DATES = "dates"
    EXCLUSIONS = "exclusions"
    BENEFIT_MAPPING = "benefit_mapping"
    DECISION_MAKING = "decision_making"
    BENEFIT_PAYMENT = "benefit_payment"
    CLAIM_RECOMMENDATION = "claim_recommendation"


class AllowedModelsOCR(str, Enum):
    GPT_4O = "openai-gpt4o"
    GPT_41 = "openai-gpt-4-1"
    GEMINI_2_0_FLASH = "gemini-2.0-flash-001"


class AllowedModelsClaim(str, Enum):
    GPT_4O = "gpt-4o"
    GPT_41 = "gpt-4.1"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_FLASH_PREVIEW_05_20 = "gemini-2.5-flash-preview-05-20"


RES_DIR = Path("res")


class Settings(BaseSettings):
    """Application settings"""

    data_dir: Path = RES_DIR / "data"
    individual_benefits_csv: Path = RES_DIR / "individual_benefits.csv"
    group_benefits_csv: Path = RES_DIR / "group_benefits.csv"

    llamaparse_api_key: str = Field(description="API key for LlamaParse")
    promptlayer_api_key: str = Field(description="API key for PromptLayer")
    openai_api_key: str = Field(description="API key for OpenAI")
    gemini_api_key: str = Field(description="API key for Gemini")

    # OCR
    similarity_threshold: float = Field(description="Threshold for similarity", default=0.75)

    # Image matching
    ssim_threshold: float = Field(description="Threshold for SSIM", default=0.8)
    psnr_threshold: float = Field(description="Threshold for PSNR", default=30)
    hash_threshold: float = Field(description="Threshold for hash", default=22.5)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings"""
    load_dotenv()

    llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    promptlayer_api_key = os.getenv("PROMPTLAYER_API_KEY")

    return Settings(
        llamaparse_api_key=llamaparse_api_key,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        promptlayer_api_key=promptlayer_api_key,
    )
