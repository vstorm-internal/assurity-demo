from enum import Enum
from datetime import date

from pydantic import Field, BaseModel

from assurity_demo.models.common import Person, Address


class ClaimDocumentType(str, Enum):
    UB04 = "UB04"
    HCFA1500 = "HCFA1500"
    FORM = "FORM"
    BILL = "BILL"
    AFTER_VISIT_SUMMARY = "AFTER_VISIT_SUMMARY"
    LETTER = "LETTER"
    EMAIL = "EMAIL"
    INVOICE = "INVOICE"
    OTHER = "OTHER"


class ServiceLine(BaseModel):
    """Represents a service line item in a claim."""

    service_date: date = Field(
        alias="serviceDate",
        description="Date when service was provided",
    )
    procedure: str = Field(description="Procedure name and/or description")
    procedure_code: str = Field(alias="procedureCode", description="Procedure code")
    charges: float = Field(description="Amount charged for this service")
    units: int = Field(description="Number of units or days of service")


class ClaimDocument(BaseModel):
    claim_document_type: ClaimDocumentType = Field(alias="claimDocumentType")

    # Core dates
    accident_date: date = Field(
        alias="accidentDate",
        description="Date when accident occurred",
    )
    treatment_date: date = Field(
        alias="treatmentDate",
        description="Date when treatment was provided",
    )

    # People
    insured: Person = Field(description="Primary insurance holder's information")
    insured_id: str | None = Field(
        alias="insuredId",
        description="Insurance member ID number",
    )
    patient: Person = Field(description="Patient's personal information")
    patient_address: Address | None = Field(
        alias="patientAddress",
        description="Patient's mailing address",
    )
    provider_name: str | None = Field(
        alias="providerName",
        description="Name of the healthcare facility",
    )
    provider_address: Address | None = Field(
        alias="providerAddress",
        description="Facility's address",
    )
    # Clinical
    principal_diagnosis: str | None = Field(
        alias="principalDiagnosis",
        description="Primary ICD-10 diagnosis code",
    )
    service_lines: list[ServiceLine] = Field(
        alias="serviceLines",
        description="List of services provided",
    )
    total_charges: float = Field(
        alias="totalCharges",
        description="Sum of all service line charges",
    )
