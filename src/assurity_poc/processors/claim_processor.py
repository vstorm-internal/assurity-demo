import json
from datetime import datetime

from langchain_core.language_models import BaseChatModel  # Fixed import path
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from logzero import logger
from promptlayer import PromptLayer

from assurity_poc.config import AllowedModels, get_settings
from assurity_poc.models import Input, Output, ParsedOutput
from src.assurity_poc.callbacks.promptlayer_callback import PromptLayerCallbackHandler

settings = get_settings()


class ClaimProcessor:
    def __init__(self, llm_name: AllowedModels = AllowedModels.AZURE_OPENAI):
        if isinstance(llm_name, str):
            llm_name = AllowedModels(llm_name)
        self.llm = self._create_llm(llm_name)
        self.output_parser = PydanticOutputParser(pydantic_object=Output)
        self.pl_client = PromptLayer(
            api_key=settings.promptlayer_api_key, enable_tracing=True
        )

    def _create_llm(self, llm_name: AllowedModels) -> BaseChatModel:
        match llm_name:
            case AllowedModels.AZURE_OPENAI:
                return AzureChatOpenAI(
                    model=llm_name.value,
                    api_key=settings.azure_openai_api_key,
                    azure_endpoint=settings.azure_openai_endpoint,
                    azure_deployment=settings.azure_openai_deployment_name,
                    api_version=settings.azure_openai_api_version,
                    callbacks=[
                        PromptLayerCallbackHandler(pl_id_callback=self._pl_id_callback)
                    ],
                )
            case AllowedModels.GEMINI:
                return ChatGoogleGenerativeAI(
                    model=llm_name.value, google_api_key=settings.gemini_api_key
                )
            case _:
                raise ValueError(
                    f"Unsupported LLM: {llm_name}. Supported values are {[m.value for m in AllowedModels]}"
                )

    # Helper functions
    @staticmethod
    def convert_role(role):
        """Convert 'user' role to 'human', keep others as is."""
        return "human" if role == "user" else role

    @staticmethod
    def convert_content(content):
        """Extract text content from list or return as is."""
        return content[0]["text"] if isinstance(content, list) else content

    def track_prompt_usage(self, prompt_layer_id):
        """Track prompt usage in PromptLayer."""
        self.pl_client.track.prompt(
            prompt_name=settings.promptlayer_prompt_name,
            request_id=prompt_layer_id,
            prompt_input_variables={
                "output_format": self.output_parser.get_format_instructions()
            },
            version=settings.promptlayer_prompt_version,
        )

        # # Define analysis prompt with format instructions
        # self.prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             """You are an expert in extracting structured data from documents.
        #             Extract and structure the information from the text according to the following format:
        #             {format_instructions}

        #             The text is a document that contains information about a policy or a claim.
        #             You should identify the type of document and extract the information accordingly.
        #             Make sure to extract all required information, do not skip anything.
        #             """,
        #         ),
        #         ("user", "{text}"),
        #     ]
        # ).partial(format_instructions=self.parser.get_format_instructions())

    def _pl_id_callback(self, promptlayer_request_id):
        logger.info("prompt layer id: %s", promptlayer_request_id)
        self.pl_client.track.metadata(
            request_id=promptlayer_request_id,
            metadata={"timestamp": datetime.now().isoformat()},
        )
        self.pl_client.track.prompt(
            request_id=promptlayer_request_id,
            prompt_name=settings.promptlayer_prompt_name,
            prompt_input_variables={
                "output_format": self.output_parser.get_format_instructions()
            },
            version=settings.promptlayer_prompt_version,
        )

    def run(self, input_docs: Input) -> ParsedOutput:
        template = self.pl_client.templates.get(
            settings.promptlayer_prompt_name,
            {
                "input_variables": {
                    "output_format": self.output_parser.get_format_instructions(),
                    "input": input_docs.model_dump_json(),
                }
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
        return Output.model_validate(json.loads(response.content))
