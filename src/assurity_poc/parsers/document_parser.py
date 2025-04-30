from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from assurity_poc.config import AllowedModels, get_settings
from assurity_poc.models import Document, ParsedOutput
from assurity_poc.parsers.base_parser import BaseParser

settings = get_settings()


class DocumentParser(BaseParser):
    def __init__(self, llm_name: AllowedModels = AllowedModels.AZURE_OPENAI):
        super().__init__(llm_name)

        # Initialize Pydantic output parser
        self.parser = PydanticOutputParser(pydantic_object=Document)

        # Define analysis prompt with format instructions
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert in extracting structured data from documents.
                    Extract and structure the information from the text according to the following format:
                    {format_instructions}
                    
                    The text is a document that contains information about a policy or a claim.
                    You should identify the type of document and extract the information accordingly.
                    Make sure to extract all required information, do not skip anything.
                    """,
                ),
                ("user", "{text}"),
            ]
        ).partial(format_instructions=self.parser.get_format_instructions())

    def parse(self, text: str) -> ParsedOutput:
        """Analyze the extracted text and return structured output."""
        try:
            # Create the chain using the pipe operator
            chain = self.prompt | self.llm | self.parser
            # Invoke the chain with the text
            structured_output = chain.invoke({"text": text})

            return ParsedOutput(raw_text=text, document=structured_output)
        except Exception as e:
            print(f"Error in text analysis: {str(e)}")
            return ParsedOutput(raw_text=text, document=None)
