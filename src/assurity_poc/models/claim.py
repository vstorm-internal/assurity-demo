from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PersonInfo(BaseModel):
    """Person information."""

    name: str | None = Field(description="Person's full name")
    date_of_birth: str | None = Field(description="Person's date of birth")
    address: str | None = Field(description="Person's address")


class PatientInfo(PersonInfo):
    """Patient information."""

    insurance_id: str | None = Field(description="Patient's insurance ID")


class ProviderInfo(PersonInfo):
    """Provider information."""

    npi: str | None = Field(description="Provider's NPI number")


class ClaimDetails(BaseModel):
    """Claim details."""

    date_of_service: str | None = Field(description="Date of service")
    diagnosis_codes: list[str] | None = Field(description="List of diagnosis codes")
    procedure_codes: list[str] | None = Field(description="List of procedure codes")
    total_charges: float | None = Field(description="Total charges amount")


class InsuranceInfo(BaseModel):
    """Insurance information."""

    policy_number: str | None = Field(description="Insurance policy number")
    group_number: str | None = Field(description="Insurance group number")
    payer_name: str | None = Field(description="Insurance payer name")


class InsuranceClaim(BaseModel):
    """Insurance claim."""

    patient_info: PatientInfo
    provider_info: ProviderInfo
    claim_details: ClaimDetails
    insurance_info: InsuranceInfo

    @model_validator(mode="before")
    @classmethod
    def validate_claim(cls, values: dict) -> dict:
        # Add any custom validation logic here
        # For example, check if at least one field is present in each section
        required_sections = [
            "patient_info",
            "provider_info",
            "claim_details",
            "insurance_info",
        ]
        for section in required_sections:
            if section not in values:
                raise ValueError(f"Missing required section: {section}")
        return values


class DocumentType(BaseModel):
    """Document type."""

    document_type: Literal["UB04", "HCFA1500", "UNKNOWN"]
    confidence: float = Field(
        description="Confidence score of the document type, 0.0-1.0 scale (percentage)."
    )
    confidence_explanation: str = Field(
        description="Explanation of the confidence score, max 100 characters.",
        max_length=100,
    )


class Document(BaseModel):
    """Document base."""

    patient_info: PatientInfo
    provider_info: ProviderInfo
    claim_details: ClaimDetails
    insurance_info: InsuranceInfo
    document_type: DocumentType


# class UB04Claim(Document):
#     """UB04 claim."""


# class HCFA1500Claim(Document):
#     """HCFA1500 claim."""


# class UnknownClaim(Document):
#     """Unknown claim type."""


class RawText(BaseModel):
    """Raw text."""

    raw_text: str = Field(
        description="Raw text of the document extracted by the OCR pipeline"
    )
    confidence: float = Field(
        description="""
        Confidence score of the OCR pipeline; 0.0-1.0 scale (percentage). 
        To assess the confidence of the OCR pipeline, use the raw text to extract structured data and decide if raw text provides enough information to extract structured data.
        Were you able to extract all information declared in output schema from the raw text? Were the fields filled correctly? Is the information complete?
        """
    )
    confidence_explanation: str = Field(
        description="Explanation of the confidence score, max 100 characters.",
        max_length=100,
    )


class OutputDocument(BaseModel):
    structured_data: Document = Field(
        description="Structured data of the document extracted by the OCR pipeline"
    )
    raw_text: RawText = Field(
        description="Raw text of the document extracted by the OCR pipeline"
    )


class Output(BaseModel):
    """Final output of the OCR pipeline."""

    documents: list[OutputDocument] = Field(
        description="List of documents extracted from the image"
    )


# class Name(BaseModel):
#     id: int
#     type: str
#     first: str
#     middle: Optional[str] = None
#     last: str
#     fullName: str


# class Person(BaseModel):
#     name: Name
#     dateOfBirth: datetime
#     emails: Optional[List[str]] = None


# class EnhancedStatus(BaseModel):
#     enhancedStatus: Optional[str] = None
#     enhancedStatusDescription: Optional[str] = None


# class Payment(BaseModel):
#     chargeAmount: float
#     paymentAmount: float
#     paymentDate: datetime
#     deliveryMethod: str
#     benefitDescription: str
#     payee: Person


# class ClaimDetail(BaseModel):
#     status: str
#     payment: Optional[Payment] = None
#     benefitCode: Optional[str] = None
#     diagnosisCode: Optional[str] = None
#     incurredDate: datetime
#     reportedDate: datetime
#     benefitStartDate: Optional[datetime] = None
#     benefitEndDate: Optional[datetime] = None
#     updatedDate: datetime


# class Claim(BaseModel):
#     companyCode: str
#     policyNumber: str
#     id: int
#     status: str
#     statusReason: str
#     enhancedStatus: EnhancedStatus
#     diagnosisCode: Optional[str] = None
#     blockId: str
#     blockType: str
#     claimant: Person
#     insured: Person
#     incurredDate: datetime
#     reportedDate: datetime
#     eliminationFromDate: Optional[datetime] = None
#     eliminationToDate: Optional[datetime] = None
#     updatedDate: datetime
#     claimDetails: Optional[List[ClaimDetail]] = None


# class ClaimHistory(BaseModel):
#     claims: List[Claim]


# class ServiceInfo(BaseModel):
#     date_of_service: datetime
#     procedure_code: Optional[str] = None
#     diagnosis_code: Optional[str] = None
#     charge_amount: Optional[float] = None
#     description: Optional[str] = None


# class ClaimForm(BaseModel):
#     patient: PatientInfo
#     provider: ProviderInfo
#     services: List[ServiceInfo]
#     insurance: InsuranceInfo
#     form_type: str = Field(..., description="Type of claim form (e.g., UB04, HCFA1500)")
#     total_charge: Optional[float] = None
#     date_filed: Optional[datetime] = None


# class CheckInfo(BaseModel):
#     check_number: str
#     amount: float
#     date: datetime
#     payee: str
#     payer: str
#     memo: Optional[str] = None
#     account_number: Optional[str] = None
