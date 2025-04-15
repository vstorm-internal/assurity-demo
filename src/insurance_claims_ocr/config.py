import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    data_dir: Path = Path("res/data")

    # OCR
    llamaparse_api_key: str = Field(description="API key for LlamaParse")
    openai_api_key: str = Field(description="API key for OpenAI")
    gemini_api_key: str = Field(description="API key for Gemini")

    # Image matching
    ssim_threshold: float = Field(description="Threshold for SSIM", default=0.8)
    psnr_threshold: float = Field(description="Threshold for PSNR", default=30)
    hash_threshold: float = Field(description="Threshold for hash", default=10)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings"""
    load_dotenv()

    llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    return Settings(
        llamaparse_api_key=llamaparse_api_key,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
    )
