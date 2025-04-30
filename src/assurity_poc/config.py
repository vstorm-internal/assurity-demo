import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class AllowedModels(str, Enum):
    GPT = "gpt-4o"
    GEMINI = "gemini"
    AZURE_OPENAI = "azure-openai"


class Settings(BaseSettings):
    """Application settings"""

    data_dir: Path = Path("res/data")

    llamaparse_api_key: str = Field(description="API key for LlamaParse")
    promptlayer_api_key: str = Field(description="API key for PromptLayer")
    azure_openai_api_key: str = Field(description="API key for Azure OpenAI")
    azure_openai_endpoint: str = Field(description="Endpoint for Azure OpenAI")
    azure_openai_deployment_name: str = Field(
        description="Deployment name for Azure OpenAI"
    )
    azure_openai_api_version: str = Field(description="API version for Azure OpenAI")

    gemini_api_key: str = Field(description="API key for Gemini")
    promptlayer_prompt_name: str = Field(description="Prompt name for PromptLayer")
    promptlayer_prompt_version: str = Field(
        description="Prompt version for PromptLayer"
    )

    # OCR
    similarity_threshold: float = Field(
        description="Threshold for similarity", default=0.75
    )

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
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    promptlayer_api_key = os.getenv("PROMPTLAYER_API_KEY")
    promptlayer_prompt_name = os.getenv("PROMPTLAYER_PROMPT_NAME")
    promptlayer_prompt_version = os.getenv("PROMPTLAYER_PROMPT_VERSION")

    return Settings(
        llamaparse_api_key=llamaparse_api_key,
        azure_openai_api_key=azure_openai_api_key,
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_deployment_name=azure_openai_deployment_name,
        azure_openai_api_version=azure_openai_api_version,
        gemini_api_key=gemini_api_key,
        promptlayer_api_key=promptlayer_api_key,
        promptlayer_prompt_name=promptlayer_prompt_name,
        promptlayer_prompt_version=promptlayer_prompt_version,
    )
