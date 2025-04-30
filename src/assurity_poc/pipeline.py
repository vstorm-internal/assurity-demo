import csv
from pathlib import Path

from logzero import logger

from assurity_poc.config import get_settings
from assurity_poc.models import Document, Input, Output
from assurity_poc.processors.claim_processor import ClaimProcessor
from assurity_poc.processors.ocr_processor import OCRProcessor
from assurity_poc.utils.file import iterate_over_files
from assurity_poc.utils.helpers import check_text_readability

settings = get_settings()


class Pipeline:
    def __init__(self):
        self.ocr_processor = OCRProcessor()
        self.claim_processor = ClaimProcessor()

    def __call__(self, claim_dir: Path) -> Output:
        output = self.run_pipeline(claim_dir)
        self._save_output(output, claim_dir)
        return output

    def run_ocr(self, claim_dir: Path) -> list[Document]:
        claim_documents = []

        for file in iterate_over_files(claim_dir):
            if not (file.is_file() and file.suffix == ".pdf"):
                continue

            # OCR
            logger.info(f"OCR: {file}")
            ocr_results = self.ocr_processor.process_image(file)
            if not check_text_readability(ocr_results["similarity"]["overall"]):
                logger.warning(f"Text is not readable: {file}")
                continue
            else:
                logger.info(f"Text is readable: {file}")
                document = Document(text=ocr_results["gpt_text"])
                claim_documents.append(document)

        return claim_documents

    def run_claim_processing(self, claim_documents: list[Document]) -> Output:
        model_input = Input(documents=claim_documents)
        output = self.claim_processor.run(model_input)
        return output

    def run_pipeline(self, claim_dir: Path) -> Output:
        claim_documents = self.run_ocr(claim_dir)
        output = self.run_claim_processing(claim_documents)
        return output

    def _save_output(self, output: Output, claim_dir: Path):
        output_dict = output.model_dump()
        decision_dict = output_dict.pop("decision", {})

        flat_output = {
            "policy_number": Path(claim_dir).name,
            **output_dict,
            **{f"decision_{k}": v for k, v in decision_dict.items()},
        }

        job_num = None
        if Path("./res/pipeline_output_job_00001.csv").exists():
            header = False
            job_num = "./res/pipeline_output_job_00001.csv".split("_")[2]
            job_num = int(job_num) + 1
        else:
            header = True
            job_num = 1
        csv_file_path = f"./res/pipeline_output_job_{job_num:05d}.csv"
        with open(csv_file_path, "a", newline="") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(flat_output.keys())
            writer.writerow(flat_output.values())
