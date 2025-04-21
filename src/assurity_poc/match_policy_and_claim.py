from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from logzero import logger

from assurity_poc.config import get_settings
from assurity_poc.models.claim import ClaimOutput
from assurity_poc.models.policy import PolicyOutput
from assurity_poc.parsers import ClaimParser, PolicyParser
from assurity_poc.processors.ocr_processor import OCRProcessor

settings = get_settings()


def parse_policy(
    policy_path: str, policy_parser: PolicyParser, ocr_processor: OCRProcessor
):
    # do ocr
    results = ocr_processor.process_image(policy_path)

    # parse policy
    policy = policy_parser.parse(results["gpt_text"])

    return policy


def parse_claim(
    claim_path: str, claim_parser: ClaimParser, ocr_processor: OCRProcessor
):
    # do ocr
    results = ocr_processor.process_image(claim_path)

    # parse claim
    claim = claim_parser.parse(results["gpt_text"])

    return claim


def was_policy_active_at_time_of_claim(policy: PolicyOutput, claim: ClaimOutput):
    # Policy is valid if it was issued before the accident date
    if claim.claim.accident_date < policy.policy.issue_date:
        return "Invalid"  # Policy wasn't in effect when accident happened
    elif (
        policy.policy.expiration_date
        and claim.claim.accident_date > policy.policy.expiration_date
    ):
        return "Invalid"  # Policy had expired when accident happened
    else:
        return "Valid"  # Policy was active when accident happened


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def run(policy_path: str, claim_path: str) -> pd.DataFrame:
    ocr_processor = OCRProcessor()
    claim_parser = ClaimParser()
    policy_parser = PolicyParser()

    logger.info(f"Parsing policy from {policy_path}")
    policy = parse_policy(policy_path, policy_parser, ocr_processor)

    logger.info(f"Parsing claim from {claim_path}")
    claim = parse_claim(claim_path, claim_parser, ocr_processor)

    logger.info("Checking if policy was active at time of claim")
    result = was_policy_active_at_time_of_claim(policy, claim)

    logger.info(f"Result: {result}")
    # Create a single row DataFrame with all data
    row = {
        "policy_path": policy_path,
        "claim_path": claim_path,
        "result": result,
        #     **flatten_dict(policy.model_dump(mode="json")),
        #     **flatten_dict(claim.model_dump(mode="json"))
    }

    return pd.DataFrame([row])


def main():
    parser = ArgumentParser(description="Match policy and claim documents")
    parser.add_argument("policy_path", help="Path to the policy document")
    parser.add_argument("claim_path", help="Path to the claim document")
    parser.add_argument("--output", "-o", help="Path to save the results (CSV format)")

    args = parser.parse_args()

    df = run(args.policy_path, args.claim_path)

    if args.output:
        # Always append, write header only if file doesn't exist
        header = not Path(args.output).exists()
        df.to_csv(args.output, index=False, mode="a", header=header)
        logger.info(f"Results appended to {args.output}")
    else:
        print(df.to_string())


if __name__ == "__main__":
    main()
