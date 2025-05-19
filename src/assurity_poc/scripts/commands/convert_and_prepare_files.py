import time

from pathlib import Path
from datetime import datetime, timedelta

from PIL import Image
from tqdm import tqdm
from logzero import logger


def run(**kwargs):
    start_time = time.time()
    script_start_dt = datetime.now()
    logger.info("--------------------------------")
    logger.info(f"Starting Convert and Prepare Files at {script_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("--------------------------------")

    input_directory_str = kwargs.get("input_directory", None)
    output_directory_str = kwargs.get("output_directory", None)
    should_delete_original_files = kwargs.get("delete_original_files", False)
    skip_existing_files = kwargs.get("skip_existing_files", True)

    if input_directory_str is None:
        raise ValueError("input_directory is required")

    input_path = Path(input_directory_str)

    if not input_path.exists():
        raise ValueError(f"input_directory does not exist: {input_path}")

    if not input_path.is_dir():
        raise ValueError(f"input_directory is not a directory: {input_path}")

    should_save_converted_file_in_line = False
    output_path_obj = None

    if output_directory_str is None:
        should_save_converted_file_in_line = True
        logger.info(
            "Output directory not provided. Converted files will be saved in the same location as source files."
        )
    else:
        output_path_obj = Path(output_directory_str)
        if not output_path_obj.exists():
            logger.info(f"Output directory '{output_path_obj}' does not exist. Creating it.")
            output_path_obj.mkdir(parents=True, exist_ok=True)
        elif not output_path_obj.is_dir():
            raise ValueError(f"Specified output_directory '{output_path_obj}' exists but is not a directory.")
        logger.info(f"Output directory set to: '{output_path_obj}'")

    logger.info(f"Scanning for .tif files in '{input_path}'...")

    tif_files = []
    try:
        tif_files = list(input_path.rglob("*.tif"))
    except Exception as e:
        logger.error(f"Error during scanning for .tif files in '{input_path}': {e}")
        script_end_dt = datetime.now()
        end_time = time.time()
        duration_seconds = end_time - start_time
        duration = timedelta(seconds=duration_seconds)
        duration_str = str(duration)
        logger.info("--------------------------------")
        logger.info(
            f"Convert and Prepare Files aborted due to error during file scanning at {script_end_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(f"Total script duration: {duration_str.split('.')[0] if '.' in duration_str else duration_str}")
        logger.info("--------------------------------")
        return

    files_found = len(tif_files)
    files_converted = 0
    files_failed = 0
    files_deleted = 0
    files_skipped = 0
    if files_found == 0:
        logger.info(f"No .tif files found in '{input_path}'.")
    else:
        logger.info(f"Found {files_found} .tif file(s). Starting conversion process...")
        for tif_file_path_obj in tqdm(tif_files, desc="Converting TIFF to PDF", unit="file"):
            try:
                output_pdf_target_path = None
                if should_save_converted_file_in_line:
                    output_pdf_target_path = tif_file_path_obj.with_suffix(".pdf")
                else:
                    relative_path = tif_file_path_obj.relative_to(input_path)
                    output_target_file_dir = output_path_obj / relative_path.parent
                    output_target_file_dir.mkdir(parents=True, exist_ok=True)
                    output_pdf_target_path = output_target_file_dir / (tif_file_path_obj.stem + ".pdf")

                if skip_existing_files and output_pdf_target_path.exists():
                    logger.info(
                        f"Skipping file '{tif_file_path_obj}' because it already exists at '{output_pdf_target_path}'"
                    )
                    files_skipped += 1
                    continue

                image = Image.open(tif_file_path_obj)

                if image.mode != "RGB":
                    has_alpha = "A" in image.mode or (image.mode == "P" and "transparency" in image.info)

                    if has_alpha:
                        img_to_convert = image
                        if img_to_convert.mode != "RGBA":
                            img_to_convert = img_to_convert.convert("RGBA")

                        background = Image.new("RGB", img_to_convert.size, (255, 255, 255))
                        background.paste(img_to_convert, mask=img_to_convert.split()[3])
                        image = background
                    elif image.mode in ("1", "L", "P"):
                        image = image.convert("RGB")
                    elif image.mode not in ("RGB", "RGBA"):
                        image = image.convert("RGB")

                image.save(output_pdf_target_path, "PDF", resolution=100.0, save_all=True)
                files_converted += 1

                if should_delete_original_files:
                    tif_file_path_obj.unlink()
                    files_deleted += 1
            except Exception as e:
                files_failed += 1
                logger.error(f"Failed to convert file '{tif_file_path_obj}': {e}")

    script_end_dt = datetime.now()
    end_time = time.time()
    duration_seconds = end_time - start_time
    duration = timedelta(seconds=duration_seconds)
    duration_str = str(duration)

    logger.info("--------------------------------")
    logger.info(f"Convert and Prepare Files completed at {script_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Processing Summary:")
    logger.info(f"  Total .tif files found: {files_found}")
    logger.info(f"  Successfully converted to PDF: {files_converted}")
    logger.info(f"  Failed to convert: {files_failed}")
    logger.info(f"  Deleted original files: {files_deleted}")
    logger.info(f"  Skipped files: {files_skipped}")
    logger.info(f"Total script duration: {duration_str.split('.')[0] if '.' in duration_str else duration_str}")
    logger.info("--------------------------------")
