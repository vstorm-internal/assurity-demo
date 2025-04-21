from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel  # Fixed import path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from assurity_poc.config import AllowedModels, get_settings

settings = get_settings()


class BaseParser(ABC):
    def __init__(self, llm_name: AllowedModels | str | None = "gpt-4o"):
        if isinstance(llm_name, str):
            llm_name = AllowedModels(llm_name)
        self.llm = self._create_llm(llm_name)

    def _create_llm(self, llm_name: AllowedModels) -> BaseChatModel:
        match llm_name:
            case AllowedModels.GPT:
                return ChatOpenAI(model=llm_name.value, api_key=settings.openai_api_key)
            case AllowedModels.GEMINI:
                return ChatGoogleGenerativeAI(
                    model=llm_name.value, google_api_key=settings.gemini_api_key
                )
            case _:
                raise ValueError(
                    f"Unsupported LLM: {llm_name}. Supported values are {[m.value for m in AllowedModels]}"
                )

    @abstractmethod
    def parse(self, text: str) -> BaseModel:
        raise NotImplementedError
