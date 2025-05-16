from datetime import date

from pydantic import Field, BaseModel

from assurity_poc.models.common import Person


class Exclusion(BaseModel):
    """Exclusion instance, ie. a condition and/or event that is not covered by the policy."""

    text: str = Field(description="Exclusion text")


class Exclusions(BaseModel):
    """
    List of ALL exclusions present in the policy. Look for the „EXCLUSIONS“ section.
    The section text usually contains a list of items, each of which is an exclusion.
    It's crucial that this list includes all exclusions found in the policy document.
    """

    exclusions: list[Exclusion] = Field(description="List of ALL exclusions present in the policy document.")


class Policy(BaseModel):
    """Policy information parsed from a policy document."""

    # Basic information
    policy_number: str = Field(description="Unique identifier for the policy")
    company_code: str = Field(alias="companyCode", description="Code identifying the insurance company")

    # People
    owner: Person = Field(description="Information about the policy owner")
    insured: Person = Field(description="Information about the primary insured")

    # Coverage dates
    issue_date: date = Field(alias="issueDate", description="Date when the policy was issued")
    effective_date: date = Field(
        alias="effectiveDate",
        description="Date when coverage begins (if not specified, use issue date)",
    )
    expiration_date: date | None = Field(default=None, alias="expirationDate", description="Date when coverage ends")
    exclusions: Exclusions = Field(
        alias="exclusions",
        description="List of ALL exclusions in the policy document. Look for the „EXCLUSIONS“ section. The section text usually contains a list of items, each of which is an exclusion. It's crucial that this list includes all exclusions found in the policy document.",
    )
