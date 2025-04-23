from pathlib import Path

from logzero import logger

from assurity_poc.config import get_settings
from assurity_poc.models.claim import ClaimOutput
from assurity_poc.models.policy import PolicyOutput
from assurity_poc.parsers import ClaimParser, PolicyParser
from assurity_poc.processors.ocr_processor import OCRProcessor

from assurity_poc.processors.image_matching import ImageMatcher

settings = get_settings()

UB04_BLANK_PATH = Path("./res/image_match/blanks/UB04_blank.png")
HCFA1500_BLANK_PATH = Path("./res/image_match/blanks/HCFA1500_blank.png")


def is_ub04(file_path: Path) -> bool:
    image_matcher = ImageMatcher()
    ub04_similarity = image_matcher(file_path, UB04_BLANK_PATH)

    return ub04_similarity["hash_diff"] <= settings.hash_threshold


def is_hcfa1500(file_path: Path) -> bool:
    image_matcher = ImageMatcher()
    hcfa1500_similarity = image_matcher(file_path, HCFA1500_BLANK_PATH)

    return hcfa1500_similarity["hash_diff"] <= settings.hash_threshold


def check_claim_type(file_path: Path) -> tuple[bool, bool]:
    is_file_ub04 = is_ub04(file_path)
    is_file_hcfa1500 = is_hcfa1500(file_path)

    if not (is_file_ub04 or is_file_hcfa1500):
        logger.warning("Unknown claim type")
        return False, False

    if is_file_ub04:
        logger.info("UB04")
    elif is_file_hcfa1500:
        logger.info("HCFA1500")

    return is_file_ub04, is_file_hcfa1500


def check_text_readability(similarity_score: float) -> bool:
    if similarity_score > settings.similarity_threshold:
        logger.info("Text is readable")
        return True
    else:
        logger.info("Text is not readable")
        return False


def parse_policy(
    policy_path: str, policy_parser: PolicyParser, ocr_processor: OCRProcessor
):
    # do ocr
    results = ocr_processor.process_image(policy_path)

    is_readable = check_text_readability(results["similarity"]["overall"])

    if not is_readable:
        return None

    # parse policy
    policy = policy_parser.parse(results["gpt_text"])

    return policy


def parse_claim(
    claim_path: str, claim_parser: ClaimParser, ocr_processor: OCRProcessor
):
    # do ocr
    results = ocr_processor.process_image(claim_path)

    is_readable = check_text_readability(results["similarity"]["overall"])

    if not is_readable:
        return None

    # parse claim
    claim = claim_parser.parse(results["gpt_text"])

    return claim


def was_policy_active_at_time_of_accident(policy: PolicyOutput, claim: ClaimOutput):
    # Policy is valid if it was issued before the accident date
    if claim.claim.accident_date < policy.policy.issue_date:
        return False  # Policy wasn't in effect when accident happened
    elif (
        policy.policy.expiration_date
        and claim.claim.accident_date > policy.policy.expiration_date
    ):
        return False  # Policy had expired when accident happened
    else:
        return True  # Policy was active when accident happened


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
