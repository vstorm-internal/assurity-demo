import time

from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from logzero import logger

from assurity_poc.pipeline import Pipeline
from assurity_poc.models.output import Claim


def run(**kwargs):
    start_time = time.time()
    logger.info("--------------------------------")
    logger.info(f"Starting Claims Adjudication pipeline at {datetime.now()}")
    logger.info("--------------------------------")

    number_of_claims_to_process = int(kwargs.get("number_of_claims_to_process", 100))
    ocr_output_dir = kwargs.get("ocr_output_dir", None)
    adjudication_output_dir = kwargs.get("adjudication_output_dir", None)

    if ocr_output_dir is None:
        raise ValueError("ocr_output_dir is required")

    if adjudication_output_dir is None:
        raise ValueError("adjudication_output_dir is required")

    ocr_output_dir = Path(ocr_output_dir)
    adjudication_output_dir = Path(adjudication_output_dir)
    if not ocr_output_dir.exists():
        raise ValueError(f"ocr_output_dir does not exist: {ocr_output_dir}")

    if not ocr_output_dir.is_dir():
        raise ValueError(f"ocr_output_dir is not a directory: {ocr_output_dir}")

    if not adjudication_output_dir.exists():
        adjudication_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created adjudication_output_dir: {adjudication_output_dir}")

    if not adjudication_output_dir.is_dir():
        raise ValueError(f"adjudication_output_dir is not a directory: {adjudication_output_dir}")

    pipeline = Pipeline()
    claims_processed = 0

    claims_to_process = Claim.load_from_path(ocr_output_dir)
    total_claims = len(claims_to_process)

    total_claims = min(total_claims, number_of_claims_to_process)

    with tqdm(total=total_claims, desc="Processing claims") as pbar:
        for claim in claims_to_process:
            logger.info(">>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"Running Adjudication on claim: {claim.claim_id} ...")
            logger.info(">>>>>>>>>>>>>>>>>>>>>>")
            claim_adjudication_start_time = time.time()
            adjudication_output = pipeline.run_adjudication_on_claim(claim, output_dir=adjudication_output_dir)
            claim_adjudication_end_time = time.time()
            logger.info(
                f"Adjudication on claim: {claim.claim_id} completed in {claim_adjudication_end_time - claim_adjudication_start_time:.2f} seconds"
            )
            pipeline.save_adjudication_output_for_claim(adjudication_output, adjudication_output_dir)
            claims_processed += 1
            pbar.update(1)

            if claims_processed >= number_of_claims_to_process:
                break

    end_time = time.time()
    logger.info(f"Adjudication pipeline completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Processed {claims_processed} claims")
