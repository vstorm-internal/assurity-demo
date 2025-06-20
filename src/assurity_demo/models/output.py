import json

from typing import Literal
from pathlib import Path

from pydantic import Field, BaseModel

from assurity_demo.models.common import Document
from assurity_demo.models.benefits import (
    BenefitPaymentOutput,
    EnhancedBenefitMappingOutput,
)


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
    policy_start_date: str | None = Field(
        default=None, description="The date the policy was issued or became effective."
    )
    accident_date: str | None = Field(default=None, description="The date the accident occurred.")
    treatment_date: str | None = Field(
        default=None,
        description="The earliest date of medical treatment related to the claim.",
    )
    decision_recommendation: Literal["accept", "deny", "requires_review"] = Field(
        description="The decision recommendation based on the dates analysis."
    )
    details_for_decision: str | None = Field(
        default=None,
        description="Explanation of findings, including dates and flags, leading to the recommendation.",
    )


class ExclusionsOutput(BaseModel):
    do_any_exclusions_apply: bool = Field(description="Whether any exclusions apply to the claim")
    exclusions: list[str] = Field(description="List of exclusions that apply to the claim")
    decision_recommendation: Literal["accept", "deny", "requires_review"] = Field(
        description="Decision recommendation based on the exclusions"
    )
    trigger_files: list[str] = Field(
        description="List of claim files that show evidence of the exclusions (not the policy documents)"
    )
    details: str = Field(
        description="Concise explanation of the decision along with every evidence file that shows the exclusions apply."
    )


class RecommendationInput(BaseModel):
    claim_documents: list[Document] = Field(description="List of claim documents")
    dates: DatesOutput = Field(description="Dates output")
    exclusions: ExclusionsOutput = Field(description="Exclusions output")
    benefits: EnhancedBenefitMappingOutput = Field(description="Benefits output")


class RecommendationOutput(BaseModel):
    claimant: str = Field(description="Name of the claimant (first name, middle name (if applicable), last name)")
    decision_recommendation: Literal["accept", "deny", "requires_review"] = Field(
        description="Decision recommendation on the claim"
    )
    decision_justification: str = Field(
        description="Concise, clear explanation for the decision recommendation, referencing specific input data and evidence.",
        max_length=1000,
    )


class ClaimRecommendation(BaseModel):
    policy_id: str = Field(description="Policy ID")
    claim_id: str = Field(description="Claim ID")
    claimant: str = Field(description="Name of the claimant (first name, middle name (if applicable), last name)")
    decision_recommendation: Literal["accept", "deny", "requires_review"] = Field(
        description="Decision recommendation on the claim"
    )
    decision_justification: str = Field(
        description="Concise, clear explanation for the decision recommendation, referencing specific input data and evidence.",
        max_length=1000,
    )
    recommended_benefit_payment_amount: float | int = Field(description="Recommended benefit payment amount")


class AdjudicationOutput(BaseModel):
    dates: DatesOutput = Field(description="Dates output")
    exclusions: ExclusionsOutput = Field(description="Exclusions output")
    benefits: EnhancedBenefitMappingOutput = Field(description="Benefit mapping output")
    benefit_payment: BenefitPaymentOutput = Field(description="Benefit payment output")
    decision: RecommendationOutput = Field(description="Decision recommendation on the claim")
    policy_id: str = Field(description="Policy ID")
    claim_id: str = Field(description="Claim ID")
    claim_documents: list[Document] = Field(description="List of claim documents")
