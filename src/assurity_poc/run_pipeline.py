from logzero import logger
import pandas as pd
from pathlib import Path

from assurity_poc.config import get_settings
from assurity_poc.utils.helpers import (
    parse_policy,
    parse_claim,
    was_policy_active_at_time_of_accident,
)
from assurity_poc.parsers import ClaimParser, PolicyParser
from assurity_poc.processors.ocr_processor import OCRProcessor

settings = get_settings()


def run_pipeline(policy_path: str, claim_path: str) -> pd.DataFrame:
    ocr_processor = OCRProcessor()
    claim_parser = ClaimParser()
    policy_parser = PolicyParser()

    # is_file_ub04, is_file_hcfa1500 = check_claim_type(claim_path)
    # if not (is_file_ub04 or is_file_hcfa1500):
    #     logger.warning("Unknown claim type")
    #     return pd.DataFrame([{"policy_path": policy_path, "claim_path": claim_path, "was_policy_active": None, "details": "Unknown claim type"}])
    # else:
    #     logger.info(
    #         "Claim (%s) type verified" % ("UB04" if is_file_ub04 else "HCFA1500")
    #     )

    logger.info(f"Parsing policy from {policy_path}")
    policy = parse_policy(policy_path, policy_parser, ocr_processor)

    if not policy:
        logger.warning("Policy is not readable")
        return pd.DataFrame(
            [
                {
                    "policy_path": policy_path,
                    "claim_path": claim_path,
                    "was_policy_active": None,
                    "details": "Policy is not readable",
                }
            ]
        )

    logger.info(f"Parsing claim from {claim_path}")
    claim = parse_claim(claim_path, claim_parser, ocr_processor)

    if not claim:
        logger.warning("Claim is not readable")
        return pd.DataFrame(
            [
                {
                    "policy_path": policy_path,
                    "claim_path": claim_path,
                    "was_policy_active": None,
                    "details": "Claim is not readable",
                }
            ]
        )

    logger.info("Checking if policy was active at time of accident...")
    was_policy_active = was_policy_active_at_time_of_accident(policy, claim)

    row = {
        "policy_path": policy_path,
        "claim_path": claim_path,
        "was_policy_active": was_policy_active,
        "details": "Success",
    }

    result_df = pd.DataFrame([row])
    return result_df


if __name__ == "__main__":
    for dir in Path("inputs").iterdir():
        claim_fp = dir / "claim.pdf"
        policy_fp = dir / "policy.pdf"
        result_df = run_pipeline(policy_fp, claim_fp)
        header = not Path("pipeline_output_job_00001.csv").exists()
        result_df.to_csv(
            "pipeline_output_job_00001.csv", index=False, mode="a", header=header
        )
        logger.info("Results appended to pipeline_output_job_00001.csv")
