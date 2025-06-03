from typing import Literal

from pydantic import Field, BaseModel

from assurity_demo.models.common import Document


class MedicalProcedure(BaseModel):
    """
    Medical procedures are the procedures that are performed on the patient, and are
    represented either by name, CPT codes, HCPCS codes, or ICD-10 PCS codes.
    Medical procedures include:
    - surgical procedures
    - medical procedures
    - diagnostic procedures
    - other procedures
    """

    name: str | None = Field(description="Name of the medical procedure")
    cpt_codes: list[int | str] | None = Field(
        description="CPT codes. CPT codes are five-digits and can be either numeric or alphanumeric."
    )
    hcpcs_codes: list[str] | None = Field(
        description="HCPCS codes. HCPCS codes are formatted as a letter followed by four numbers."
    )
    icd10_pcs_codes: list[str] | None = Field(
        description="ICD-10 PCS codes. ICD-10 PCS codes are formatted as seven-character alphanumeric codes used for inpatient medical procedures."
    )


class Benefit(BaseModel):
    """
    Benefits are procedures or services that are covered by the policy.
    """

    name: str = Field(description="Name of the benefit")


class BenefitMappingOutput(BaseModel):
    medical_procedures_in_claim: list[MedicalProcedure] = Field(
        description="List of all medical procedures found in claim documents"
    )
    policy_benefits: list[Benefit] = Field(description="List of all benefits found in the policy documents")
    policy_type: Literal["INDIVIDUAL", "GROUP"] = Field(description="Type of policy. Either 'INDIVIDUAL' or 'GROUP'")
    document_quality_issues: list[str] = Field(
        default_factory=list,
        description="List of any issues that might affect processing (illegible codes, incomplete descriptions, etc.)",
    )


class EnhancedBenefitMappingOutput(BenefitMappingOutput):
    """Extended output that includes coverage determination results for pipeline compatibility."""

    covered: list[MedicalProcedure] = Field(
        default_factory=list,
        description="List of medical procedures that are covered by the policy (determined by code-based lookup)",
    )
    not_covered: list[MedicalProcedure] = Field(
        default_factory=list,
        description="List of medical procedures that are NOT covered by the policy",
    )


class BenefitPayment(BaseModel):
    """
    Benefit payment is the amount of money that is payable to the patient for a given benefit.
    """

    benefit: Benefit = Field(description="Benefit that is being paid for")
    payment_amount: float = Field(description="Amount of money that is payable to the patient")


class BenefitPaymentInput(BaseModel):
    """
    Input of the benefit payment model.
    """

    claim_documents: list[Document] = Field(description="List of claim documents")
    benefits: list[Benefit] = Field(description="List of benefits")


class BenefitPaymentOutput(BaseModel):
    """
    Output of the benefit payment model.
    """

    benefit_payments: list[BenefitPayment] = Field(
        description="List of benefits and the amount of money that is payable to the patient for each benefit."
    )
