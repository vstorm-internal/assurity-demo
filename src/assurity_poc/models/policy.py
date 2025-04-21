from datetime import date
from enum import Enum

from pydantic import BaseModel, Field

from assurity_poc.models.base import Person


class PolicyType(str, Enum):
    """Type of insurance coverage provided by the policy."""

    LIFE = "LIFE"
    DISABILITY = "DISABILITY"
    ACCIDENT = "ACCIDENT"
    HEALTH = "HEALTH"


class PolicyStatus(str, Enum):
    """Current state of an insurance policy."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    CANCELLED = "CANCELLED"


class PremiumMode(str, Enum):
    """Frequency at which policy premiums are paid."""

    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"


class Policy(BaseModel):
    """Policy information parsed from a policy document."""

    # Basic information
    policy_number: str = Field(description="Unique identifier for the policy")
    company_code: str = Field(
        alias="companyCode", description="Code identifying the insurance company"
    )
    # type: PolicyType = Field(description="Type of insurance coverage")
    # status: PolicyStatus = Field(description="Current status of the policy")

    # People
    owner: Person = Field(description="Information about the policy owner")
    insured: Person = Field(description="Information about the primary insured")
    # beneficiary: Person | None = Field(
    #     default=None, description="Information about the beneficiary"
    # )

    # Financial
    # premium: float = Field(description="Premium amount in USD")
    # premium_mode: PremiumMode = Field(
    #     alias="premiumMode", description="Frequency of premium payments"
    # )
    # face_amount: float = Field(
    #     alias="faceAmount", description="Face value of the policy in USD"
    # )

    # Coverage dates
    issue_date: date = Field(
        alias="issueDate", description="Date when the policy was issued"
    )
    effective_date: date = Field(
        alias="effectiveDate",
        description="Date when coverage begins (if not specified, use issue date)",
    )
    expiration_date: date | None = Field(
        default=None, alias="expirationDate", description="Date when coverage ends"
    )


class PolicyOutput(BaseModel):
    raw_text: str = Field(
        description="Original text extracted from the policy document"
    )
    policy: Policy | None = Field(description="Parsed policy data")
