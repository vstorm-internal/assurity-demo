"""
This script is used to demo the pipeline, in particular checking for exclusions.
"""

import json

from random import choice
from pathlib import Path

from logzero import logger

from assurity_poc.config import Prompts
from assurity_poc.models import Document
from assurity_poc.pipeline import Pipeline


def run_demo() -> None:
    input_dir = Path("./res/outputs/ocr/")
    output_dir = Path("./res/outputs/exclusions/")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing pipeline")
    pipeline = Pipeline()
    PROMPT = Prompts.EXCLUSIONS.value

    files = list(input_dir.glob("*.json"))
    processed_files = list(output_dir.glob("*.json"))
    files = [f for f in files if f not in processed_files]

    file = choice(files)

    # Load the OCR data
    with open(file, "r") as f:
        logger.debug(f"Loading {file}")
        ocr_data = json.load(f)

    # Convert to Document objects
    documents = [Document(text=claim_file["text"], file_name=claim_file["file_name"]) for claim_file in ocr_data]

    # Check for exclusions
    res = pipeline.check_exclusions(documents, PROMPT)

    with open(output_dir / f"{file.stem}.json", "w") as f:
        json.dump(res.model_dump(), f)

    logger.info("Pipeline completed")


if __name__ == "__main__":
    run_demo()
