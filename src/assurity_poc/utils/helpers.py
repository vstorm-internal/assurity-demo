from pathlib import Path

from logzero import logger

from assurity_poc.config import get_settings
from assurity_poc.models.claim import ClaimDocument
from assurity_poc.models.policy import Policy
from assurity_poc.processors.image_matching import ImageMatcher

settings = get_settings()

UB04_BLANK_PATH = Path("./res/image_match/blanks/UB04_blank.png")
HCFA1500_BLANK_PATH = Path("./res/image_match/blanks/HCFA1500_blank.png")


def is_ub04(file_path: Path) -> bool:
    image_matcher = ImageMatcher()
    ub04_similarity = image_matcher(str(file_path), str(UB04_BLANK_PATH))

    return ub04_similarity["hash_diff"] <= settings.hash_threshold


def is_hcfa1500(file_path: Path) -> bool:
    image_matcher = ImageMatcher()
    hcfa1500_similarity = image_matcher(str(file_path), str(HCFA1500_BLANK_PATH))

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
        return True
    else:
        return False


# def parse_policy(
#     policy_path: str, policy_parser: PolicyParser, ocr_processor: OCRProcessor
# ):
#     # do ocr
#     results = ocr_processor.process_image(policy_path)

#     is_readable = check_text_readability(results["similarity"]["overall"])

#     if not is_readable:
#         return None

#     # parse policy
#     policy = policy_parser.parse(results["gpt_text"])

#     return policy


# def parse_claim(
#     claim_path: str, claim_parser: ClaimParser, ocr_processor: OCRProcessor
# ):
#     # do ocr
#     results = ocr_processor.process_image(claim_path)

#     is_readable = check_text_readability(results["similarity"]["overall"])

#     if not is_readable:
#         return None

#     # parse claim
#     claim = claim_parser.parse(results["gpt_text"])

#     return claim


def was_policy_active_at_time_of_accident(policy: Policy, claim: ClaimDocument) -> bool:
    # Policy is active if it was issued before the accident date and hasn't expired
    if claim.accident_date < policy.issue_date:
        return False  # Policy wasn't in effect when accident happened
    elif policy.expiration_date and claim.accident_date > policy.expiration_date:
        return False  # Policy had expired when accident happened
    else:
        return True  # Policy was active when accident happened


def was_treatment_completed_within_policy_timeframe(
    policy: Policy, claim: ClaimDocument
) -> bool:
    # Treatment is valid if it was completed within the policy timeframe
    if claim.treatment_date < policy.issue_date:
        return False  # Policy wasn't in effect when treatment happened
    elif policy.expiration_date and claim.treatment_date > policy.expiration_date:
        return False  # Policy had expired when treatment happened
    else:
        return True  # Policy was active when treatment happened
