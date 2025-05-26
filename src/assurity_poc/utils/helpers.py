import re

from pathlib import Path

import pandas as pd

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


def was_policy_active_at_time_of_accident(policy: Policy, claim: ClaimDocument) -> bool:
    # Policy is active if it was issued before the accident date and hasn't expired
    if claim.accident_date < policy.issue_date:
        return False  # Policy wasn't in effect when accident happened
    elif policy.expiration_date and claim.accident_date > policy.expiration_date:
        return False  # Policy had expired when accident happened
    else:
        return True  # Policy was active when accident happened


def was_treatment_completed_within_policy_timeframe(policy: Policy, claim: ClaimDocument) -> bool:
    # Treatment is valid if it was completed within the policy timeframe
    if claim.treatment_date < policy.issue_date:
        return False  # Policy wasn't in effect when treatment happened
    elif policy.expiration_date and claim.treatment_date > policy.expiration_date:
        return False  # Policy had expired when treatment happened
    else:
        return True  # Policy was active when treatment happened


def get_benefits(individual_benefits_csv: str | Path, group_benefits_csv: str | Path) -> dict[str, pd.DataFrame]:
    individual_benefits_df = parse_benefits(individual_benefits_csv)
    group_benefits_df = parse_benefits(group_benefits_csv)

    return {
        "individual": individual_benefits_df,
        "group": group_benefits_df,
    }


def parse_benefits(benefits_csv: str | Path) -> pd.DataFrame:
    benefits_df = pd.read_csv(benefits_csv)

    COLUMN_NAME_MAP = {
        "Benefit": "name",
        "CPT Codes": "cpt_codes",
        "State Specific": "state_specific",
        "HCPCS Codes": "hcpcs_codes",
        "ICD-10 PCS Codes": "icd10_pcs_codes",
    }
    benefits_df = benefits_df.rename(columns=COLUMN_NAME_MAP)
    benefits_df["cpt_codes"] = benefits_df["cpt_codes"].apply(parse_cpt_codes)
    benefits_df["icd10_pcs_codes"] = benefits_df["icd10_pcs_codes"].apply(parse_icd10_pcs_codes)
    benefits_df["hcpcs_codes"] = benefits_df["hcpcs_codes"].apply(parse_hcpcs_codes)
    benefits_df["state_specific"] = benefits_df["state_specific"].apply(parse_cpt_codes)

    return benefits_df


def parse_icd10_pcs_codes(cell):
    if pd.isna(cell) or not isinstance(cell, str):
        return []
    return [code.strip() for code in cell.split(",") if code.strip()]


# def parse_cpt_codes(cell):
#     if pd.isna(cell) or not isinstance(cell, str):
#         return []
#     codes = []
#     for part in cell.split(','):
#         part = part.strip()
#         # Match range: e.g. 99202 - 99205
#         m = re.match(r"(\d+)\s*-\s*(\d+)", part)
#         if m:
#             start, end = int(m.group(1)), int(m.group(2))
#             codes.extend(list(range(start, end + 1)))
#         else:
#             # Match single code
#             m = re.match(r"(\d+)", part)
#             if m:
#                 codes.append(int(m.group(1)))
#             elif part:  # Keep non-empty comments as-is
#                 codes.append(part)
#     return codes


def parse_cpt_codes(cell):
    if pd.isna(cell) or not isinstance(cell, str):
        return []
    codes = []
    # Find all code ranges (e.g., 99202 - 99205)
    for m in re.finditer(r"(\d+)\s*-\s*(\d+)", cell):
        start, end = int(m.group(1)), int(m.group(2))
        codes.extend(list(range(start, end + 1)))
    # Remove all code ranges from the string
    cell_wo_ranges = re.sub(r"\d+\s*-\s*\d+", "", cell)
    # Find all single codes (e.g., 99202)
    for m in re.finditer(r"\b\d{5}\b", cell_wo_ranges):
        codes.append(int(m.group(0)))
    # Remove all single codes from the string
    cell_wo_codes = re.sub(r"\b\d{5}\b", "", cell_wo_ranges)
    # What remains is the comment (if any), after stripping commas and whitespace
    comment = cell_wo_codes.strip(" ,")
    if comment:
        codes.append(comment)
    return codes


def parse_hcpcs_codes(cell: str) -> list[str]:
    if pd.isna(cell) or not isinstance(cell, str):
        return []
    codes = []
    for part in cell.split(","):
        part = part.strip()
        # Match range: e.g. G0380 - G0384
        m = re.match(r"([A-Z]\d+)\s*-\s*([A-Z]\d+)", part)
        if m:
            prefix = m.group(1)[0]
            start, end = int(m.group(1)[1:]), int(m.group(2)[1:])
            codes.extend([f"{prefix}{i:04d}" for i in range(start, end + 1)])
        else:
            m = re.match(r"([A-Z]\d+)", part)
            if m:
                codes.append(m.group(1))
            elif part:
                codes.append(part)
    return codes
