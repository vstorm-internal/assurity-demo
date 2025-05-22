import json

from typing import Literal
from pathlib import Path

from pydantic import Field, BaseModel

from assurity_poc.models.common import Document
from assurity_poc.models.benefits import BenefitMappingOutput


class Claim(BaseModel):
    policy_id: str = Field(description="Policy ID")
    claim_id: str = Field(description="Claim ID")
    documents: list[Document] = Field(description="List of claim documents")

    @classmethod
    def load_from_path(cls, path: Path) -> list["Claim"]:
        """
        Scan a directory for JSON files and load valid Claim objects.

        Args:
            path: Path to the directory containing claim JSON files

        Returns:
            List of valid Claim objects found in the directory
        """
        claims = []

        # Check if path is a directory
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")

        # Iterate through all JSON files in the directory
        for file_path in path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                # Attempt to create a Claim object and validate it
                claim = cls(**data)
                claims.append(claim)

            except (json.JSONDecodeError, TypeError, ValueError):
                # Log the error but continue processing other files
                continue

        return claims


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
    benefits: BenefitMappingOutput = Field(description="Benefits output")


class FinalDecision(BaseModel):
    status: Literal["pay", "deny", "refer"] = Field(description="Decision on the claim.")
    details: str = Field(
        description="Detailed explanation of the decision along with every piece of information that led to it."
    )


class AdjudicationOutput(BaseModel):
    dates: DatesOutput = Field(description="Dates output")
    exclusions: ExclusionsOutput = Field(description="Exclusions output")
    benefits: BenefitMappingOutput = Field(description="Benefits output")
    decision: FinalDecision = Field(description="Decision on the claim")
    policy_id: str = Field(description="Policy ID")
    claim_id: str = Field(description="Claim ID")
    claim_documents: list[Document] = Field(description="List of claim documents")
