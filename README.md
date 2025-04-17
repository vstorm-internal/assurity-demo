# Insurance Claims OCR Pipeline

A Python package for extracting and structuring information from insurance claim forms using OCR and LLMs.

## Features

- Extracts text from insurance claim form images using LlamaParse with GPT-4 Vision
- Structures extracted data into standardized formats for UB04 and HCFA1500 claims
- Uses LangChain with GPT-4 for intelligent text analysis and data extraction
- Handles patient info, provider details, claim information, and insurance data
- Robust error handling and validation

## Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- OpenAI API key
- LlamaParse API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/assurity-poc.git
cd assurity-poc
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key and LlamaParse API key to the `.env` file

## Usage

1. Activate the Poetry environment:
```bash
poetry shell
```

2. Run the script:
```bash
python src/insurance_claims_ocr/main.py
```

## Project Structure

```sh
.
├── src/insurance_claims_ocr/   # Source code
├── res/                        # Sample data and test images
├── benchmarks/                 # Benchmarking scripts
├── notebooks/                  # Jupyter notebooks for development
├── pyproject.toml              # Project dependencies and configuration
├── poetry.lock                 # Locked dependencies
└── .env                        # Environment variables
```

## TIFF to PDF Conversion
To convert TIFF files to PDF, run the following command:
```bash
poetry run python src/insurance_claims_ocr/utils.py <TIFF_FILES_ROOT_DIR>
```
where `<TIFF_FILES_ROOT_DIR>` is the path to the directory containing the TIFF files.
It can be directory of directories, and it will convert all TIFF files in the directory and its subdirectories.

## Development

- Install development dependencies:
```bash
poetry install --with dev
```

- Run pre-commit hooks:
```bash
pre-commit install
```