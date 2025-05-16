from pydantic import Field, BaseModel


class Benefit(BaseModel):
    name: str = Field(description="Name of the benefit")
    cpt_codes: list[int | str] = Field(description="CPT codes associated with the benefit")
    state_specific: list[int] | None = Field(description="State specific codes associated with the benefit (if any)")
    hcpcs_codes: list[str] | None = Field(description="HCPCS codes associated with the benefit")
    icd10_pcs_codes: list[str] | None = Field(description="ICD-10 PCS codes associated with the benefit")


class AllBenefits(BaseModel):
    benefits: list[Benefit] = Field(description="List of benefits")


class BenefitPresentInClaim(BaseModel):
    name: str = Field(description="Name of the benefit")
    cpt_codes: list[int | str] = Field(description="CPT code(s) present in the claim")
    state_specific: list[int] | None = Field(description="State specific code(s) present in the claim")
    hcpcs_codes: list[str] | None = Field(description="HCPCS code(s) present in the claim")
    icd10_pcs_codes: list[str] | None = Field(description="ICD-10 PCS code(s) present in the claim")


class BenefitsPresentInClaim(BaseModel):
    benefits_present: list[BenefitPresentInClaim] = Field(description="List of benefits present in the claim")


class BenefitPresentInPolicy(BaseModel):
    name: str = Field(description="Name of the benefit")
    cpt_codes: list[int | str] = Field(description="CPT code(s) present in the policy")
    state_specific: list[int] | None = Field(description="State specific code(s) present in the policy")
    hcpcs_codes: list[str] | None = Field(description="HCPCS code(s) present in the policy")
    icd10_pcs_codes: list[str] | None = Field(description="ICD-10 PCS code(s) present in the policy")


class BenefitsPresentInPolicy(BaseModel):
    benefits_present: list[BenefitPresentInPolicy] = Field(description="List of benefits present in the policy")


class BenefitsCovered(BaseModel):
    benefits_covered: list[BenefitPresentInClaim] = Field(
        description="List of benefits present in the claim and covered by the policy"
    )


class BenefitsNotCovered(BaseModel):
    benefits_not_covered: list[BenefitPresentInClaim] = Field(
        description="List of benefit codes present in the claim but not covered by the policy"
    )


class BenefitsOutput(BaseModel):
    benefits_present_in_claim: BenefitsPresentInClaim = Field(description="Benefits present in the claim")
    benefits_present_in_policy: BenefitsPresentInPolicy = Field(description="Benefits present in the policy")
    benefits_covered: BenefitsCovered = Field(description="Benefits covered in the claim")
    benefits_not_covered: BenefitsNotCovered = Field(description="Benefits not covered in the claim")
