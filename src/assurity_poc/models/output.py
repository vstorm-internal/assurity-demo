from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    text: str = Field(description="Original text extracted from the document")


class Input(BaseModel):
    documents: list[Document] = Field(description="List of documents")


class Decision(BaseModel):
    decision: Literal["accept", "review"] = Field(description="Decision on the claim.")
    reason: str = Field(description="Reason for the decision.")
    details: str = Field(
        description="Details of the decision containing all the information that led to the decision. Try to be as specific as possible."
    )
    should_reject: bool | None = Field(
        description="Should the claim be rejected? As the decision for the claim can only be either `accept` or `review`, this field must be true if any conditions for rejection are met. Thanks to this we can differentiate between claims that require a human review for a more detailed analysis and claims that are clearly invalid."
    )


class Output(BaseModel):
    was_policy_active: bool = Field(
        description="Whether the policy was active at the time of the accident"
    )
    was_treatment_completed_within_policy_timeframe: bool = Field(
        description="Whether the treatment was completed within the policy timeframe"
    )
    do_any_exclusions_apply: bool = Field(
        description="Whether any exclusions apply to the claim"
    )
    decision: Decision = Field(description="Decision on the claim")
