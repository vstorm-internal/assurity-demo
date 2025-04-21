from pathlib import Path

from assurity_poc.config import get_settings
from assurity_poc.utils.helpers import image_matcher

settings = get_settings()

UB04_BLANK_PATH = Path("./res/image_match/blanks/UB04_blank.png")
HCFA1500_BLANK_PATH = Path("./res/image_match/blanks/HCFA1500_blank.png")


def is_ub04(file_path: Path) -> bool:
    ub04_similarity = image_matcher(file_path, UB04_BLANK_PATH)

    return ub04_similarity["hash_diff"] <= settings.hash_threshold


def is_hcfa1500(file_path: Path) -> bool:
    hcfa1500_similarity = image_matcher(file_path, HCFA1500_BLANK_PATH)

    return hcfa1500_similarity["hash_diff"] <= settings.hash_threshold
