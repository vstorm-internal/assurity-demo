from typing import Literal

from pydantic import Field, BaseModel

from .benefits import BenefitsOutput


class Document(BaseModel):
    text: str = Field(description="Original text extracted from the document")
    file_name: str = Field(description="Name of the file")


class Input(BaseModel):
    documents: list[Document] = Field(description="List of documents")


class DatesOutput(BaseModel):
    was_policy_active: bool = Field(description="Whether the policy was active at the time of the accident")
    was_treatment_within_policy_timeframe: bool = Field(
        description="Whether the treatment was completed within the policy timeframe"
    )
    status: Literal["pay", "deny", "refer"] = Field(description="Decision on the claim based on the dates")


class ExclusionsOutput(BaseModel):
    do_any_exclusions_apply: bool = Field(description="Whether any exclusions apply to the claim")
    exclusions: list[str] = Field(description="List of exclusions that apply to the claim")
    status: Literal["pay", "deny", "refer"] = Field(description="Decision on the claim based on the exclusions")
    trigger_files: list[str] = Field(
        description="List of claim files that show evidence of the exclusions (not the policy documents)"
    )
    details: str = Field(
        description="Concise explanation of the decision along with every evidence file that shows the exclusions apply."
    )


class FinalDecisionInput(BaseModel):
    dates: DatesOutput = Field(description="Dates output")
    exclusions: ExclusionsOutput = Field(description="Exclusions output")
    benefits: BenefitsOutput = Field(description="Benefits output")


class FinalDecision(BaseModel):
    status: Literal["pay", "deny", "refer"] = Field(description="Decision on the claim.")
    details: str = Field(
        description="Detailed explanation of the decision along with every piece of information that led to it."
    )


class Output(BaseModel):
    dates: DatesOutput = Field(description="Dates output")
    exclusions: ExclusionsOutput = Field(description="Exclusions output")
    benefits: BenefitsOutput = Field(description="Benefits output")
    decision: FinalDecision = Field(description="Decision on the claim")
