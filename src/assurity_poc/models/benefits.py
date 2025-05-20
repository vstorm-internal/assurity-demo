from typing import Literal

from pydantic import Field, BaseModel


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
        description="ICD-10 PCS codes. ICD-10 PCS codes are formatted as seven-character alphanumeric codes."
    )


class Benefit(BaseModel):
    """
    Benefits are procedures or services that are covered by the policy.
    """

    name: str = Field(description="Name of the benefit")


class BenefitsOutput(BaseModel):
    medical_procedures_in_claim: list[MedicalProcedure] = Field(
        description="List of all medical procedures found in claim documents"
    )
    policy_benefits: list[Benefit] = Field(
        description="List of all benefits found in the policy"
    )
    covered: list[MedicalProcedure] = Field(
        description="List of medical procedures present in the claim that are covered by the policy."
    )
    not_covered: list[MedicalProcedure] = Field(
        description="List of medical procedures present in the claim that are NOT covered by the policy."
    )
    policy_type: Literal["INDIVIDUAL", "GROUP"] = Field(
        description="Type of policy. Either 'INDIVIDUAL' or 'GROUP'"
    )
