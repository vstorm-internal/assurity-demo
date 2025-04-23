from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from assurity_poc.models.base import Address, Person


class ClaimFormType(str, Enum):
    UB04 = "UB04"  # Hospital/institutional claim form
    HCFA1500 = "HCFA1500"  # Professional services claim form


class ServiceLine(BaseModel):
    """Represents a service line item in a claim."""

    service_date: date = Field(description="Date when service was provided")
    procedure: str = Field(description="Procedure name and/or description")
    procedure_code: str = Field(alias="procedureCode", description="Procedure code")
    charges: float = Field(description="Amount charged for this service")
    units: int = Field(description="Number of units or days of service")


class ClaimBase(BaseModel):
    form_type: ClaimFormType | str = Field(
        description="Type of claim form (UB04 or HCFA1500)"
    )

    # Core dates
    accident_date: date = Field(description="Date when accident occurred")
    treatment_date: date = Field(description="Date when treatment was provided")

    # People
    insured: Person = Field(description="Primary insurance holder's information")
    insured_id: str | None = Field(
        alias="insuredId", description="Insurance member ID number"
    )
    patient: Person = Field(description="Patient's personal information")
    patient_address: Address | None = Field(
        alias="patientAddress", description="Patient's mailing address"
    )
    provider_name: str | None = Field(
        alias="providerName", description="Name of the healthcare facility"
    )
    provider_address: Address | None = Field(
        alias="providerAddress", description="Facility's address"
    )
    # Clinical
    principal_diagnosis: str | None = Field(
        alias="principalDiagnosis", description="Primary ICD-10 diagnosis code"
    )
    service_lines: list[ServiceLine] = Field(
        alias="serviceLines", description="List of services provided"
    )
    total_charges: float = Field(
        alias="totalCharges", description="Sum of all service line charges"
    )


class UB04Claim(ClaimBase):
    form_type: Literal[ClaimFormType.UB04] = Field(
        default=ClaimFormType.UB04, description="Type of claim form (UB04)"
    )


class HCFA1500Claim(ClaimBase):
    form_type: Literal[ClaimFormType.HCFA1500] = Field(
        default=ClaimFormType.HCFA1500,
        description="Type of claim form (HCFA1500)",
    )


class ClaimOutput(BaseModel):
    raw_text: str = Field(description="Original text extracted from the claim form")
    claim: UB04Claim | HCFA1500Claim | ClaimBase = Field(
        description="Parsed claim data", default=None
    )
