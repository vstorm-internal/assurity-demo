from datetime import date

from pydantic import Field, BaseModel


class Person(BaseModel):
    """Represents a person in the insurance system.

    Includes policyholders, beneficiaries, claimants, and insured persons.
    """

    id: int | None = Field(default=None, description="Unique identifier for the person in the system")
    type: str | None = Field(
        default=None,
        description="Type/category of the person (e.g., 'POLICYHOLDER', 'BENEFICIARY')",
    )
    first_name: str = Field(alias="firstName", description="Person's first name")
    middle_name: str | None = Field(alias="middleName", default=None, description="Person's middle name")
    last_name: str = Field(alias="lastName", description="Person's last name")
    full_name: str | None = Field(alias="fullName", default=None, description="Complete name of the person")
    date_of_birth: date | str | None = Field(alias="dateOfBirth", description="Person's date of birth")
    emails: list[str] | None = Field(default=None, description="List of email addresses associated with the person")


class Address(BaseModel):
    # Required fields - essential for address
    street: str | None = Field(default=None, description="Street address including apartment/suite number")
    city: str | None = Field(default=None, description="City name")
    state: str | None = Field(default=None, description="Two-letter state code")
    zip_code: str | None = Field(default=None, alias="zipCode", description="5 or 9 digit ZIP code")
    # Optional field
    phone: str | None = Field(default=None, description="Contact phone number")


class Document(BaseModel):
    text: str = Field(description="Original text extracted from the document")
    file_name: str = Field(description="Name of the file")

    def to_json(self):
        return self.model_dump_json()
