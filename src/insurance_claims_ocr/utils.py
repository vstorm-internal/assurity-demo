import re
from pathlib import Path
from typing import Iterator

import fire
import numpy as np
import torch
from Levenshtein import ratio
from logzero import logger
from PIL import Image, ImageSequence
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from insurance_claims_ocr.config import get_settings

settings = get_settings()


def convert_tiff_to_pdf(tiff_path: Path, split_pages: bool = False) -> None:
    if not (
        (tiff_path.suffix == ".tiff" or tiff_path.suffix == ".tif")
        and tiff_path.is_file()
        and tiff_path.exists()
    ):
        logger.warning(f"{tiff_path} is not a valid TIFF file.")
        raise ValueError(f"{tiff_path} is not a valid TIFF file.")
    else:
        # pdf_path = tiff_path.absolute().with_suffix(".pdf")
        output_dir = settings.data_dir / "converted" / tiff_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / Path(tiff_path.stem).with_suffix(".pdf")

        logger.debug(f"{tiff_path.name} -> {pdf_path.name}")

        image = Image.open(tiff_path)

        if split_pages:
            split_and_save(image, pdf_path)
        else:
            save_no_split(image, pdf_path)

        logger.debug("OK")


def split_and_save(image: Image.Image, pdf_path: Path) -> None:
    """Split the image into pages and save each page as a separate PDF file"""
    for i, page in enumerate(ImageSequence.Iterator(image)):
        page = page.convert("RGB")
        page.save(pdf_path.with_name(f"{pdf_path.stem}_page_{i + 1}.pdf"))
        logger.debug(f"Saved page {i + 1} of {pdf_path.name}")


def save_no_split(image: Image.Image, pdf_path: Path) -> None:
    """Save the image as a PDF file without splitting it into pages"""
    images = []
    for i, page in enumerate(ImageSequence.Iterator(image)):
        page = page.convert("RGB")
        images.append(page)
    if len(images) == 1:
        images[0].save(pdf_path)
    else:
        images[0].save(pdf_path, save_all=True, append_images=images[1:])
    logger.debug(f"Saved {pdf_path.name}")


def tiff_to_jpeg(tiff_path: Path) -> None:
    jpeg_path = settings.output_dir / Path(tiff_path.stem).with_suffix(".jpeg")

    print(f"{tiff_path.name} -> {jpeg_path.name}")

    tiff_image = Image.open(tiff_path)
    # Convert the image to JPEG format
    jpeg_image = tiff_image.convert("RGB")

    # Save the JPEG image
    jpeg_image.save(str(jpeg_path))
    print("OK")


def iterate_over_files(dir_path: Path) -> Iterator[Path]:
    """Iterate over all files in the directory"""
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            yield file_path
        else:
            yield from iterate_over_files(file_path)


def convert_all_tiff_to_pdf(dir_path: Path, split_pages: bool = False) -> None:
    """Convert all TIFF files to PDF files"""
    for file_path in iterate_over_files(dir_path):
        try:
            convert_tiff_to_pdf(Path(file_path), split_pages)
        except Exception as e:
            logger.error(f"Error converting {file_path}: {e}")


def preprocess_text(text):
    # 1) Lowercase
    text = text.lower()
    # 2) Remove Markdown table lines (---) and header separators (|---)
    text = re.sub(r"\|?-+\|?", " ", text)
    # 3) Remove remaining table pipes '|'
    text = text.replace("|", " ")
    # 4) Remove Markdown emphasis symbols (e.g. '**')
    text = re.sub(r"\*{1,}", " ", text)
    # 5) Remove headings like '# some title'
    text = re.sub(r"^[#]+\s+", "", text, flags=re.MULTILINE)
    # 6) Remove leading bullet chars like '- ', '* ', etc.
    text = re.sub(r"^[\-*]\s+", "", text, flags=re.MULTILINE)
    # 7) Remove other punctuation except alphanumeric/whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    # 8) Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Text similarity
def compute_levenshtein_ratio(text1, text2):
    return ratio(text1, text2)


def jaccard_similarity(a, b):
    set_a, set_b = set(a.split()), set(b.split())
    return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0


def tfidf_cosine(a, b):
    vec = TfidfVectorizer().fit_transform([a, b])
    return cosine_similarity(vec[0], vec[1])[0][0]


## Embedding-based similarity
def embedding_cosine(a, b):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec1 = model.encode(a, convert_to_tensor=True)
    vec2 = model.encode(b, convert_to_tensor=True)
    emb_score = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()
    return emb_score


def overall_score(levenshtein: float, jaccard: float, tfidf: float, embedding: float):
    normalized_embedding = (embedding + 1) / 2  # normalize to [0, 1]
    return (levenshtein + jaccard + tfidf + normalized_embedding) / 4


def compute_text_similarity(text1: str, text2: str):
    levenshtein = compute_levenshtein_ratio(text1, text2)
    jaccard = jaccard_similarity(text1, text2)
    tfidf = tfidf_cosine(text1, text2)
    embedding = embedding_cosine(text1, text2)

    return {
        "levenshtein": levenshtein,
        "jaccard": jaccard,
        "tfidf": tfidf,
        "embedding": embedding,
        "overall": overall_score(
            levenshtein=levenshtein,
            jaccard=jaccard,
            tfidf=tfidf,
            embedding=embedding,
        ),
    }


def split_pdf_into_pages(pdf_path: Path) -> Iterator[Path]:
    """Split a PDF file into individual pages and save them as separate PDFs"""
    # Create output directory next to original PDF
    output_dir = pdf_path.parent / pdf_path.stem + "_pages"
    output_dir.mkdir(exist_ok=True)

    # Open PDF and split into pages
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)

        output_path = output_dir / f"{pdf_path.stem}_page_{i + 1}.pdf"
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
        yield output_path

def convert_pdf_to_image(pdf_path: Path) -> Image.Image:
    """Convert a PDF file to an image"""
    logger.debug(f"Converting {pdf_path} to image")
    pdf_path = Path(pdf_path)
    if pdf_path.suffix != ".pdf":
        logger.warning(f"{pdf_path} is not a valid PDF file.")
        return None
    pages = convert_from_path(pdf_path, 300)  # resolution in DPI
    return pages[0]


def fire_convert(dir_path: str | Path, split_pages: bool = True) -> None:
    """Convert all TIFF files to PDF files"""
    dir_path = Path(dir_path)
    convert_all_tiff_to_pdf(dir_path, split_pages=split_pages)


if __name__ == "__main__":
    fire.Fire(fire_convert)
