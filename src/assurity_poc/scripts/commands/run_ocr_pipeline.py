import time

from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from logzero import logger

from assurity_poc.pipeline import Pipeline


def run(**kwargs):
    start_time = time.time()
    logger.info("--------------------------------")
    logger.info(f"Starting OCR pipeline at {datetime.now()}")
    logger.info("--------------------------------")

    number_of_policies_to_process = int(kwargs.get("number_of_policies_to_process", 100))
    policy_directory = kwargs.get("policy_directory", None)
    output_dir = kwargs.get("output_dir", None)
    policy_id = kwargs.get("policy_id", None)

    if output_dir is None:
        raise ValueError("output_dir is required")

    if policy_directory is None:
        raise ValueError("policy_directory is required")

    policy_directory = Path(policy_directory)
    if not policy_directory.exists():
        raise ValueError(f"policy_directory does not exist: {policy_directory}")

    if not policy_directory.is_dir():
        raise ValueError(f"policy_directory is not a directory: {policy_directory}")

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    if not output_dir.is_dir():
        raise ValueError(f"output_dir is not a directory: {output_dir}")

    pipeline = Pipeline()
    policies_processed = 0

    # Count the number of policy directories
    policy_dirs = [d for d in policy_directory.iterdir() if d.is_dir()]
    total_policies = min(len(policy_dirs), number_of_policies_to_process)

    # Create progress bar
    with tqdm(total=total_policies, desc="Processing policies") as pbar:
        for policy_dir in policy_dirs:
            policy_id = policy_dir.name
            logger.info(">>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"Running OCR on policy: {policy_id} ...")
            logger.info(">>>>>>>>>>>>>>>>>>>>>>")
            policy_ocr_start_time = time.time()
            policy_claims_output = pipeline.run_ocr_on_claims_in_directory(policy_dir, policy_id)
            policy_ocr_end_time = time.time()
            logger.info(
                f"OCR on policy: {policy_id} completed in {policy_ocr_end_time - policy_ocr_start_time:.2f} seconds"
            )

            # We iterate through each claim in the policy_claims_output and save the ocr output
            for claim in policy_claims_output:
                pipeline.save_ocr_output_for_claim(claim, output_dir)

            policies_processed += 1
            pbar.update(1)

            if policies_processed >= number_of_policies_to_process:
                break

    end_time = time.time()
    logger.info(f"OCR pipeline completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Processed {policies_processed} policies")


if __name__ == "__main__":
    run()
