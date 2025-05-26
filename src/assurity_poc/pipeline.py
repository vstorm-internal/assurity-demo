import csv
import json

from typing import Any
from pathlib import Path

import pandas as pd

from logzero import logger

from assurity_poc.config import Prompts, get_settings
from assurity_poc.models import (
    Claim,
    Input,
    Benefit,
    Document,
    DatesOutput,
    ExclusionsOutput,
    AdjudicationOutput,
    BenefitPaymentInput,
    ClaimRecommendation,
    RecommendationInput,
    BenefitMappingOutput,
    BenefitPaymentOutput,
    RecommendationOutput,
)
from assurity_poc.utils.file import iterate_over_files
from assurity_poc.utils.helpers import get_benefits, check_text_readability
from assurity_poc.processors.ocr_processor import OCRProcessor
from assurity_poc.processors.claim_processor import ClaimProcessor

settings = get_settings()


class Pipeline:
    def __init__(self) -> None:
        self.ocr_processor = OCRProcessor()
        self.claim_processor = ClaimProcessor()
        self.benefits = self._get_benefits(
            settings.individual_benefits_csv,
            settings.group_benefits_csv,
        )

    def __call__(self, claim_dir: Path) -> Any:
        output = self.run_pipeline(claim_dir)
        self._save_output(output, claim_dir)
        return output

    def run_ocr_on_claims_in_directory(self, policy_dir: Path, policy_id: str) -> list[Claim]:
        # Each sub folder of policy_dir is a claim, the name of the sub folder is the claim number
        # Process the claim and return a Claim object with policy_id
        claims = []
        for claim_dir in policy_dir.iterdir():
            if claim_dir.is_dir():
                logger.info("================")
                logger.info(f"Running OCR on claim: {claim_dir.name} under Policy: {policy_id}")
                logger.info("================")
                claim_id = claim_dir.name
                claim_documents = self.run_ocr(claim_dir, should_save_ocr_output=False)
                claims.append(
                    Claim(
                        policy_id=policy_id,
                        claim_id=claim_id,
                        documents=claim_documents,
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

    def map_benefits(self, claim_documents: list[Document], prompt_name: str) -> BenefitMappingOutput:
        logger.debug("Mapping benefits")
        model_input = Input(documents=claim_documents)
        benefit_mapping_output = self.claim_processor.run(
            input=model_input,
            output_class=BenefitMappingOutput,
            prompt_name=prompt_name,
        )
        benefits_from_codes = self._map_benefit_codes(benefit_mapping_output)
        for benefit in benefits_from_codes:
            if benefit not in benefit_mapping_output.covered:
                benefit_mapping_output.covered.append(benefit)

        return benefit_mapping_output

    def _get_benefits(self, individual_benefits_csv: Path, group_benefits_csv: Path) -> dict[str, pd.DataFrame]:
        return get_benefits(individual_benefits_csv, group_benefits_csv)

    # def make_decision(self, input: RecommendationInput) -> Any:
    #     return self.claim_processor.run(
    #         input=input,
    #         output_class=FinalDecision,
    #         prompt_name=settings.promptlayer_prompt_names[2],
    #     )

    def _map_benefit_codes(self, benefits_output: BenefitMappingOutput) -> list[str]:
        def translate_code(code: int | str, column: str) -> list:
            benefits_tmp = []
            filtered_df = benefits_df[
                benefits_df[column].apply(
                    lambda x: code in x  # noqa: B023
                )  # noqa: B023
            ]
            benefits_tmp = filtered_df["name"].tolist()
            return benefits_tmp

        if benefits_output.policy_type == "INDIVIDUAL":
            benefits_df = self.benefits["individual"]
        elif benefits_output.policy_type == "GROUP":
            benefits_df = self.benefits["group"]
        else:
            raise ValueError(f"Invalid policy type: {benefits_output.policy_type}")

        benefits_from_codes = []
        for medical_procedure in benefits_output.medical_procedures_in_claim:
            if medical_procedure.cpt_codes:
                for code in medical_procedure.cpt_codes:
                    filtered_list = translate_code(code, "cpt_codes")
                    benefits_from_codes.extend(filtered_list)
            if medical_procedure.hcpcs_codes:
                for code in medical_procedure.hcpcs_codes:
                    filtered_list = translate_code(code, "hcpcs_codes")
                    benefits_from_codes.extend(filtered_list)
            if medical_procedure.icd10_pcs_codes:
                for code in medical_procedure.icd10_pcs_codes:
                    filtered_list = translate_code(code, "icd10_pcs_codes")
                    benefits_from_codes.extend(filtered_list)
        benefits_from_codes = list(set(benefits_from_codes))
        # find which benefits obtained from codes in claim documents are present in the policy benefits
        benefits_from_codes_in_policy = [
            benefit for benefit in benefits_from_codes if benefit in benefits_output.policy_benefits
        ]
        return benefits_from_codes_in_policy

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
        benefits_output: BenefitMappingOutput,
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

        recommended_benefit_payment_amount = sum(
            benefit_payment.payment_amount for benefit_payment in benefit_payment_output.benefit_payments
        )
        claim_recommendation = ClaimRecommendation(
            policy_id=claim.policy_id,
            claim_id=claim.claim_id,
            claimant=recommendation_tmp.claimant,
            decision_recommendation=recommendation_tmp.decision_recommendation,
            decision_justification=recommendation_tmp.decision_justification,
            recommended_benefit_payment_amount=recommended_benefit_payment_amount,
        )

        self.save_claim_recommendation_for_claim(
            claim_recommendation=claim_recommendation,
            output_dir=Path("./res/outputs/pipeline_output"),
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

    def save_claim_recommendation_for_claim(self, claim_recommendation: ClaimRecommendation, output_dir: Path):
        with open(
            output_dir / f"{claim_recommendation.policy_id}_{claim_recommendation.claim_id}_recommendation.json",
            "w",
        ) as f:
            json.dump(claim_recommendation.model_dump(), f)

    def save_ocr_output_for_claim(self, claim: Claim, output_dir: Path):
        with open(output_dir / f"{claim.policy_id}_{claim.claim_id}.json", "w") as f:
            json.dump(claim.model_dump(), f)

    def _save_ocr_output(self, ocr_output: list[dict], claim_dir: Path) -> None:
        with open(Path("./res/outputs/ocr") / f"{Path(claim_dir).name}.json", "a") as f:
            json.dump(ocr_output, f)

    def _save_output(self, output: AdjudicationOutput, claim_dir: Path) -> None:
        output_dict = output.model_dump()
        decision_dict = output_dict.pop("decision", {})

        flat_output = {
            "policy_number": Path(claim_dir).name,
            **output_dict,
            **{f"{k}": v for k, v in decision_dict.items()},
        }

        # job_num = None
        job_num = 1
        if Path("./res/pipeline_output_job_00001.csv").exists():
            header = False
            # job_num = Path("./res/pipeline_output_job_00001.csv").stem.split("_")[-1]
            # job_num = int(job_num) + 1
        else:
            header = True
        csv_file_path = Path(f"./res/pipeline_output_job_{job_num:05d}.csv")
        logger.info(f"Saving output to {csv_file_path.name}")
        with open(csv_file_path, "a", newline="") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(flat_output.keys())
            writer.writerow(flat_output.values())
        logger.info(f"Saved output to {csv_file_path.name}")
