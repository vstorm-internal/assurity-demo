from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from assurity_poc.config import AllowedModels, get_settings
from assurity_poc.models import ClaimBase, ClaimOutput
from assurity_poc.parsers.base import BaseParser

settings = get_settings()


class ClaimParser(BaseParser):
    def __init__(self, llm_name: AllowedModels | str | None = "gpt-4o"):
        super().__init__(llm_name)

        # Initialize Pydantic output parser
        self.parser = PydanticOutputParser(pydantic_object=ClaimBase)

        # Define analysis prompt with format instructions
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert in processing insurance claim documents.
                    Extract and structure the information from the text according to the following format:
                    {format_instructions}
                    """,
                ),
                ("user", "{text}"),
            ]
        ).partial(format_instructions=self.parser.get_format_instructions())

    def parse(self, text: str) -> ClaimOutput:
        """Analyze the extracted text and return structured output."""
        try:
            # Create the chain using the pipe operator
            chain = self.prompt | self.llm | self.parser
            # Invoke the chain with the text
            structured_output = chain.invoke({"text": text})

            return ClaimOutput(raw_text=text, claim=structured_output)
        except Exception as e:
            print(f"Error in text analysis: {str(e)}")
            return ClaimOutput(raw_text=text, claim=None)
