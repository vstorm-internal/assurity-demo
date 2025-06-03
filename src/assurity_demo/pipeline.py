import ast
import json

from typing import Any
from pathlib import Path

import pandas as pd

from logzero import logger

from assurity_demo.config import Prompts, AllowedModelsOCR, AllowedModelsClaim, get_settings
from assurity_demo.models import (
    Claim,
    Input,
    Benefit,
    Document,
    DatesOutput,
    ExclusionsOutput,
    MedicalProcedure,
    AdjudicationOutput,
    BenefitPaymentInput,
    RecommendationInput,
    BenefitMappingOutput,
    BenefitPaymentOutput,
    RecommendationOutput,
)
from assurity_demo.utils.file import iterate_over_files
from assurity_demo.utils.helpers import get_benefits, check_text_readability
from assurity_demo.models.benefits import EnhancedBenefitMappingOutput
from assurity_demo.processors.ocr_processor import OCRProcessor
from assurity_demo.processors.claim_processor import ClaimProcessor

settings = get_settings()


class Pipeline:
    def __init__(self, ocr_model: AllowedModelsOCR = None, claim_model: AllowedModelsClaim = None) -> None:
        self.ocr_processor = OCRProcessor()
        self.claim_processor = ClaimProcessor(claim_model) if claim_model else ClaimProcessor()
        self.benefits = self._get_benefits(
            settings.individual_benefits_csv,
            settings.group_benefits_csv,
        )

    def run_ocr_on_claims_in_directory(self, policy_dir: Path, policy_id: str) -> list[Claim]:
        # Each sub folder of policy_dir is a claim, the name of the sub folder is the claim number
        # Process the claim and return a Claim object with policy_id

        # First, process policy-level files (these contain important policy information)
        logger.info("================")
        logger.info(f"Processing policy-level files for Policy: {policy_id}")
        logger.info("================")

        policy_documents = []
        for file in policy_dir.iterdir():
            if (
                file.is_file()
                and file.suffix == ".pdf"
                and "PolicyPages" in file.name
                and "CLAIM_CORRESPONDENCE" not in file.name
                and "DEPOSIT" not in file.name
                and "CHECK" not in file.name
            ):
                logger.info(f"OCR on policy file: {file.name}")
                ocr_results = self.ocr_processor.process_image(file)

                if check_text_readability(ocr_results["similarity"]["overall"]):
                    logger.info(f"Policy file text is readable: {file.name}")
                    document = Document(text=ocr_results["gpt_text"], file_name=file.name)
                    policy_documents.append(document)
                else:
                    logger.warning(f"Policy file text is not readable: {file.name}")

        logger.info(f"Processed {len(policy_documents)} policy-level documents")

        # Now process claims and include policy documents in each claim
        claims = []
        for claim_dir in policy_dir.iterdir():
            if claim_dir.is_dir():
                logger.info("================")
                logger.info(f"Running OCR on claim: {claim_dir.name} under Policy: {policy_id}")
                logger.info("================")
                claim_id = claim_dir.name
                claim_documents = self.run_ocr(claim_dir, should_save_ocr_output=False)

                # Combine policy documents with claim documents
                all_documents = policy_documents + claim_documents

                claims.append(
                    Claim(
                        policy_id=policy_id,
                        claim_id=claim_id,
                        documents=all_documents,
                    )
                )

        return claims

    def run_ocr(self, claim_dir: Path, should_save_ocr_output: bool = True) -> list[Document]:
        claim_documents = []

        ocr_output = []
        num_files_in_claims_dir = len(list(claim_dir.iterdir()))
        num_files_ocr_processed = 0
        num_files_ocr_skipped = 0
        num_files_ocr_failed = 0

        logger.info(f"Running OCR for {claim_dir.name}")
        for file in iterate_over_files(claim_dir):
            if (
                not (file.is_file() and file.suffix == ".pdf")
                or "CLAIM_CORRESPONDENCE" in file.name
                or "DEPOSIT" in file.name
                or "CHECK" in file.name
            ):
                num_files_ocr_skipped += 1
                continue

            # OCR
            logger.info(f"OCR: {file.name}")
            ocr_results = self.ocr_processor.process_image(file)
            ocr_output_tmp = {"text": ocr_results["gpt_text"], "file_name": file.name}
            ocr_output.append(ocr_output_tmp)

            if not check_text_readability(ocr_results["similarity"]["overall"]):
                logger.warning(f"Text is not readable: {file.name}")
                num_files_ocr_failed += 1
                continue
            else:
                logger.info(f"Text is readable: {file.name}")
                document = Document(text=ocr_results["gpt_text"], file_name=file.name)
                claim_documents.append(document)
                num_files_ocr_processed += 1
        if should_save_ocr_output:
            self._save_ocr_output(ocr_output, claim_dir)

        logger.info(
            f"Completed OCR: {claim_dir.name} - {num_files_ocr_processed}/{num_files_in_claims_dir} files processed"
        )
        return claim_documents

    def check_dates(self, claim_documents: list[Document], prompt_name: str) -> Any:
        logger.debug("Checking dates")
        model_input = Input(documents=claim_documents)
        dates_output = self.claim_processor.run(input=model_input, output_class=DatesOutput, prompt_name=prompt_name)

        return dates_output

    def check_exclusions(self, claim_documents: list[Document], prompt_name: str) -> Any:
        logger.debug("Checking exclusions")
        model_input = Input(documents=claim_documents)
        exclusions_output = self.claim_processor.run(
            input=model_input, output_class=ExclusionsOutput, prompt_name=prompt_name
        )

        return exclusions_output

    def map_benefits(self, claim_documents: list[Document], prompt_name: str) -> EnhancedBenefitMappingOutput:
        logger.debug("Mapping benefits")
        model_input = Input(documents=claim_documents)
        benefit_mapping_output = self.claim_processor.run(
            input=model_input,
            output_class=BenefitMappingOutput,
            prompt_name=prompt_name,
        )

        if benefit_mapping_output.policy_type not in ["INDIVIDUAL", "GROUP"]:
            logger.warning(
                f"Policy type is missing or invalid: {benefit_mapping_output.policy_type}. Attempting to infer or default."
            )

        benefits_df = None
        if benefit_mapping_output.policy_type == "INDIVIDUAL":
            benefits_df = self.benefits["individual"]
        elif benefit_mapping_output.policy_type == "GROUP":
            benefits_df = self.benefits["group"]

        # Determine coverage using code-based lookups
        covered_procedures: list[MedicalProcedure] = []
        not_covered_procedures: list[MedicalProcedure] = []

        for medical_procedure in benefit_mapping_output.medical_procedures_in_claim:
            matched_benefit_names = set()

            # Check all procedure codes against the benefits database
            if benefits_df is not None:
                if medical_procedure.cpt_codes:
                    for code in medical_procedure.cpt_codes:
                        matched_benefit_names.update(self._get_benefit_names_from_code(code, "cpt_codes", benefits_df))

                if medical_procedure.hcpcs_codes:
                    for code in medical_procedure.hcpcs_codes:
                        matched_benefit_names.update(
                            self._get_benefit_names_from_code(code, "hcpcs_codes", benefits_df)
                        )

                if medical_procedure.icd10_pcs_codes:
                    for code in medical_procedure.icd10_pcs_codes:
                        matched_benefit_names.update(
                            self._get_benefit_names_from_code(code, "icd10_pcs_codes", benefits_df)
                        )

            if matched_benefit_names:
                primary_benefit = list(matched_benefit_names)[0]  # Take first match
                covered_procedure = medical_procedure.model_copy(deep=True)
                covered_procedure.name = primary_benefit
                covered_procedures.append(covered_procedure)

                logger.debug(
                    f"Procedure covered by '{primary_benefit}': {medical_procedure.name or 'Unnamed'} (codes matched)"
                )
            else:
                not_covered_procedures.append(medical_procedure)
                logger.debug(f"Procedure not covered: {medical_procedure.name or 'Unnamed'} (no matching codes)")

        enhanced_output = EnhancedBenefitMappingOutput(
            medical_procedures_in_claim=benefit_mapping_output.medical_procedures_in_claim,
            policy_benefits=benefit_mapping_output.policy_benefits,
            policy_type=benefit_mapping_output.policy_type,
            document_quality_issues=benefit_mapping_output.document_quality_issues,
            covered=covered_procedures,
            not_covered=not_covered_procedures,
        )

        return enhanced_output

    def _get_benefit_names_from_code(self, code: int | str, column: str, benefits_df: pd.DataFrame) -> list[str]:
        """Helper function to get benefit names from a code and DataFrame column."""
        if benefits_df is None or column not in benefits_df.columns:
            return []
        try:

            def is_code_present(cell_value, target_code):
                if isinstance(cell_value, list):
                    # Handle both integer and string codes in the list
                    if column == "cpt_codes" or column == "state_specific":
                        # CPT codes are stored as integers after range expansion
                        try:
                            target_int = int(target_code)
                            return target_int in cell_value or str(target_int) in map(str, cell_value)
                        except (ValueError, TypeError):
                            return str(target_code) in map(str, cell_value)
                    else:
                        # HCPCS and ICD-10 PCS codes are stored as strings
                        return str(target_code) in map(str, cell_value)

                if isinstance(cell_value, str):
                    if cell_value.startswith("[") and cell_value.endswith("]"):
                        try:
                            parsed_list = ast.literal_eval(cell_value)
                            if column == "cpt_codes" or column == "state_specific":
                                try:
                                    target_int = int(target_code)
                                    return target_int in parsed_list or str(target_int) in map(str, parsed_list)
                                except (ValueError, TypeError):
                                    return str(target_code) in map(str, parsed_list)
                            else:
                                return str(target_code) in map(str, parsed_list)
                        except:  # noqa E722
                            return False
                    else:
                        # Handle comma-separated string format
                        codes_in_cell = [c.strip() for c in cell_value.split(",")]
                        if column == "cpt_codes" or column == "state_specific":
                            try:
                                target_int = int(target_code)
                                return str(target_int) in codes_in_cell or target_int in [
                                    int(c) for c in codes_in_cell if c.isdigit()
                                ]
                            except (ValueError, TypeError):
                                return str(target_code) in codes_in_cell
                        else:
                            return str(target_code) in codes_in_cell
                return False

            filtered_df = benefits_df[benefits_df[column].apply(lambda x: is_code_present(x, code))]
            return filtered_df["name"].tolist()
        except Exception as e:
            logger.error(f"Error in _get_benefit_names_from_code for code {code}, column {column}: {e}")
            return []

    def _get_benefits(self, individual_benefits_csv: Path, group_benefits_csv: Path) -> dict[str, pd.DataFrame]:
        return get_benefits(individual_benefits_csv, group_benefits_csv)

    def calculate_benefit_payments(
        self, claim_documents: list[Document], benefits: list[Benefit], prompt_name: str
    ) -> BenefitPaymentOutput:
        logger.debug("Calculating benefit payments")
        model_input = BenefitPaymentInput(claim_documents=claim_documents, benefits=benefits)

        benefit_payment_output = self.claim_processor.run(
            input=model_input,
            output_class=BenefitPaymentOutput,
            prompt_name=prompt_name,
        )

        return benefit_payment_output

    def get_claim_recommendation(
        self,
        claim_documents: list[Document],
        dates_output: DatesOutput,
        exclusions_output: ExclusionsOutput,
        benefits_output: EnhancedBenefitMappingOutput,
        prompt_name: str,
    ) -> RecommendationOutput:
        model_input = RecommendationInput(
            claim_documents=claim_documents,
            dates=dates_output,
            exclusions=exclusions_output,
            benefits=benefits_output,
        )

        claim_recommendation = self.claim_processor.run(
            input=model_input,
            output_class=RecommendationOutput,
            prompt_name=prompt_name,
        )

        return claim_recommendation

    def run_adjudication_on_claim(self, claim: Claim, output_dir: Path) -> AdjudicationOutput:
        exclusions_output = self.check_exclusions(claim_documents=claim.documents, prompt_name=Prompts.EXCLUSIONS.value)

        # Create proper objects for the required fields
        dates_output = DatesOutput(
            was_policy_active=False,
            was_treatment_within_policy_timeframe=False,
            status="refer",
        )

        benefits_output = self.map_benefits(claim_documents=claim.documents, prompt_name=Prompts.BENEFIT_MAPPING.value)

        # Create a proper FinalDecision object
        decision = RecommendationOutput(
            claimant="John Doe",  # placeholder for now
            decision_recommendation=exclusions_output.status,
            decision_justification=f"Decision based on exclusions: {exclusions_output.details}",
        )
        benefit_payment_output = BenefitPaymentOutput(
            recommended_benefit_payment_amount=0,  # placeholder for now
        )

        return AdjudicationOutput(
            policy_id=claim.policy_id,
            claim_id=claim.claim_id,
            claim_documents=claim.documents,
            dates=dates_output,
            exclusions=exclusions_output,
            benefits=benefits_output,
            benefit_payment=benefit_payment_output,
            decision=decision,
        )

    def run_pipeline(self, claim: Claim, output_dir: Path) -> AdjudicationOutput:
        claim_documents = claim.documents
        # LLM PHASE 1: DATES PHASE
        dates_output = self.check_dates(claim_documents=claim_documents, prompt_name=Prompts.DATES.value)

        # LLM PHASE 2: EXCLUSIONS PHASE
        exclusions_output = self.check_exclusions(claim_documents=claim_documents, prompt_name=Prompts.EXCLUSIONS.value)

        # LLM PHASE 3: BENEFITS MAPPING PHASE
        benefits_output = self.map_benefits(
            claim_documents=claim_documents,
            prompt_name=Prompts.BENEFIT_MAPPING.value,
        )

        # LLM PHASE 4: BENEFIT PAYMENT AMOUNT
        benefits_to_pay = [Benefit(name=medical_proc.name) for medical_proc in benefits_output.covered]
        benefit_payment_output = self.calculate_benefit_payments(
            claim_documents=claim_documents,
            benefits=benefits_to_pay,
            prompt_name=Prompts.BENEFIT_PAYMENT.value,
        )

        # LLM PHASE 5: CLAIM RECOMMENDATION
        recommendation_tmp = self.get_claim_recommendation(
            claim_documents=claim_documents,
            dates_output=dates_output,
            exclusions_output=exclusions_output,
            benefits_output=benefits_output,
            prompt_name=Prompts.CLAIM_RECOMMENDATION.value,
        )

        return AdjudicationOutput(
            policy_id=claim.policy_id,
            claim_id=claim.claim_id,
            claim_documents=claim_documents,
            dates=dates_output,
            exclusions=exclusions_output,
            benefits=benefits_output,
            benefit_payment=benefit_payment_output,
            decision=recommendation_tmp,
        )

    def save_adjudication_output_for_claim(self, adjudication_output: AdjudicationOutput, output_dir: Path):
        with open(
            output_dir / f"{adjudication_output.policy_id}_{adjudication_output.claim_id}_adjudication.json",
            "w",
        ) as f:
            json.dump(adjudication_output.model_dump(), f)

    def save_ocr_output_for_claim(self, claim: Claim, output_dir: Path):
        with open(output_dir / f"{claim.policy_id}_{claim.claim_id}.json", "w") as f:
            json.dump(claim.model_dump(), f)

    def _save_ocr_output(self, ocr_output: list[dict], claim_dir: Path) -> None:
        with open(Path("./res/outputs/ocr") / f"{Path(claim_dir).name}.json", "a") as f:
            json.dump(ocr_output, f)
