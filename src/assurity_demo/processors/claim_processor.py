from typing import Any
from datetime import datetime

from logzero import logger
from pydantic import BaseModel
from promptlayer import PromptLayer
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel

from assurity_demo.config import AllowedModelsClaim, get_settings
from assurity_demo.callbacks.promptlayer_callback import PromptLayerCallbackHandler

settings = get_settings()


class ClaimProcessor:
    def __init__(self, llm_name: AllowedModelsClaim = AllowedModelsClaim.GPT_41):
        if isinstance(llm_name, str):
            llm_name = AllowedModelsClaim(llm_name)
        self.llm = self._create_llm(llm_name)
        self.pl_client = PromptLayer(api_key=settings.promptlayer_api_key, enable_tracing=True)
        self._current_prompt_name = None

    def _create_llm(self, llm_name: AllowedModelsClaim) -> BaseChatModel:
        match llm_name:
            case AllowedModelsClaim.GPT_4O | AllowedModelsClaim.GPT_41:
                return ChatOpenAI(
                    model=llm_name.value,
                    api_key=settings.openai_api_key,
                    callbacks=[PromptLayerCallbackHandler(pl_id_callback=self._pl_id_callback)],
                )
            case AllowedModelsClaim.GEMINI_2_0_FLASH | AllowedModelsClaim.GEMINI_2_5_FLASH_PREVIEW_05_20:
                return ChatGoogleGenerativeAI(
                    model=llm_name.value,
                    google_api_key=settings.gemini_api_key,
                    callbacks=[PromptLayerCallbackHandler(pl_id_callback=self._pl_id_callback)],
                )
            case _:
                raise ValueError(
                    f"Unsupported LLM: {llm_name}. Supported values are {[m.value for m in AllowedModelsClaim]}"
                )

    # Helper functions
    @staticmethod
    def convert_role(role: str) -> str:
        """Convert 'user' role to 'human', keep others as is."""
        return "human" if role == "user" else role

    @staticmethod
    def convert_content(content: list[dict]) -> str | Any:
        """Extract text content from list or return as is."""
        return content[0]["text"] if isinstance(content, list) else content

    def _pl_id_callback(self, promptlayer_request_id: str) -> None:
        logger.debug("prompt layer id: %s", promptlayer_request_id)
        self.pl_client.track.metadata(
            request_id=promptlayer_request_id,
            metadata={"timestamp": datetime.now().isoformat()},
        )
        self.pl_client.track.prompt(
            request_id=promptlayer_request_id,
            prompt_name=self._current_prompt_name,
            prompt_input_variables={"output_format": self.output_parser.get_format_instructions()},
        )

    def run(
        self,
        input: BaseModel | Any,
        prompt_name: str,
        output_class: type = BaseModel,
    ) -> BaseModel:
        self._current_prompt_name = prompt_name

        # Create base parser
        base_parser = PydanticOutputParser(pydantic_object=output_class)

        # Use OutputFixingParser that will retry on JSON errors
        self.output_parser = OutputFixingParser.from_llm(parser=base_parser, llm=self.llm)

        input_variables = {
            "output_format": self.output_parser.get_format_instructions(),
            "input": input.model_dump_json(),
        }

        template = self.pl_client.templates.get(
            self._current_prompt_name,
            {
                "input_variables": input_variables,
            },
        )
        pl_messages = template["llm_kwargs"]["messages"]
        messages = [
            (
                self.convert_role(pl_message["role"]),
                self.convert_content(pl_message["content"]),
            )
            for pl_message in pl_messages
        ]
        response = self.llm.invoke(messages)

        try:
            # Use the fixing parser directly - it will handle JSON errors and retry
            parsed_result = self.output_parser.parse(response.content)
            return parsed_result
        except Exception as e:
            logger.error(f"OutputFixingParser failed to parse response: {e}")
            logger.error(f"Response content (first 1000 chars): {response.content[:1000]}")
