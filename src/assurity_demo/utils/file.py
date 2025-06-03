from typing import Iterator
from pathlib import Path

import fire

from PIL import Image, ImageSequence
from PyPDF2 import PdfReader, PdfWriter
from logzero import logger
from pdf2image import convert_from_path

from assurity_demo.config import get_settings

settings = get_settings()


def convert_tiff_to_pdf(tiff_path: Path, split_pages: bool = False, output_dir: Path | None = None) -> None:
    if not ((tiff_path.suffix == ".tiff" or tiff_path.suffix == ".tif") and tiff_path.is_file() and tiff_path.exists()):
        logger.warning(f"{tiff_path} is not a valid TIFF file.")
        raise ValueError(f"{tiff_path} is not a valid TIFF file.")
    else:
        if output_dir is None:
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


def split_and_save(image: Image.Image, pdf_path: Path | str) -> None:
    """Split the image into pages and save each page as a separate PDF file"""
    pdf_path = Path(pdf_path)
    for i, page in enumerate(ImageSequence.Iterator(image)):
        page = page.convert("RGB")
        page.save(pdf_path.with_name(f"{pdf_path.stem}_page_{i + 1}.pdf"))
        logger.debug(f"Saved page {i + 1} of {pdf_path.name}")


def save_no_split(image: Image.Image, pdf_path: Path | str) -> None:
    """Save the image as a PDF file without splitting it into pages"""
    pdf_path = Path(pdf_path)
    images = []
    for i, page in enumerate(ImageSequence.Iterator(image)):
        page = page.convert("RGB")
        images.append(page)
    if len(images) == 1:
        images[0].save(pdf_path)
    else:
        images[0].save(pdf_path, save_all=True, append_images=images[1:])
    logger.debug(f"Saved {pdf_path.name}")


def convert_tiff_to_jpeg(tiff_path: Path | str) -> None:
    tiff_path = Path(tiff_path)
    jpeg_path = settings.output_dir / Path(tiff_path.stem).with_suffix(".jpeg")

    print(f"{tiff_path.name} -> {jpeg_path.name}")

    tiff_image = Image.open(tiff_path)
    # Convert the image to JPEG format
    jpeg_image = tiff_image.convert("RGB")

    # Save the JPEG image
    jpeg_image.save(str(jpeg_path))
    print("OK")


def iterate_over_files(dir_path: Path | str) -> Iterator[Path]:
    """Iterate over all files in the directory"""
    dir_path = Path(dir_path)
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            yield file_path
        else:
            yield from iterate_over_files(file_path)


def convert_all_tiff_to_pdf(dir_path: Path | str, split_pages: bool = False) -> None:
    """Convert all TIFF files to PDF files"""
    for file_path in iterate_over_files(dir_path):
        try:
            convert_tiff_to_pdf(Path(file_path), split_pages)
        except Exception as e:
            logger.error(f"Error converting {file_path}: {e}")


def split_pdf_into_pages(pdf_path: Path | str) -> Iterator[Path]:
    """Split a PDF file into individual pages and save them as separate PDFs"""
    pdf_path = Path(pdf_path)
    # Create output directory next to original PDF
    output_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
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


def convert_pdf_to_image(pdf_path: Path | str) -> Image.Image:
    """Convert a PDF file to an image"""
    logger.debug(f"Converting {pdf_path} to image")
    pdf_path = Path(pdf_path)
    if pdf_path.suffix != ".pdf":
        logger.warning(f"{pdf_path} is not a valid PDF file.")
        return None
    pages = convert_from_path(pdf_path, 300)  # resolution in DPI
    return pages[-1]


def fire_convert(dir_path: str | Path, split_pages: bool = True) -> None:
    """Convert all TIFF files to PDF files"""
    dir_path = Path(dir_path)
    convert_all_tiff_to_pdf(dir_path, split_pages=split_pages)


if __name__ == "__main__":
    fire.Fire(fire_convert)
