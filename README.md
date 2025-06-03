# Insurance Claims Adjudication – Streamlit App

A modern Streamlit app for browsing, adjudicating, and managing insurance claims using OCR and LLMs. Designed for speed, usability, and robust claim processing.

---

## Features

- **Browse & Select Claims:** Instantly view and select from pre-processed claims with existing OCR results.
- **Visual PDF Preview:** See the first page of each claim as an image, directly in the browser.
- **Model Selection:** Choose OCR and claim adjudication models from the sidebar before running the pipeline.
- **Upload New Claims:** Upload new PDF claim forms and run OCR (LlamaParse) on demand.
- **Run Full Pipeline:** Process selected claims through a multi-phase adjudication pipeline with a single click.
- **Session State & History:** Track processing history and manage state for a seamless user experience.
- **Robust Error Handling:** Clear feedback and validation for all user actions.

---

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/assurity-poc.git
   cd assurity-poc
   ```

2. **Install dependencies (using Poetry):**
   ```bash
   poetry install
   ```

3. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key and LlamaParse API key to the `.env` file

4. **Run the Streamlit app:**
   ```bash
   poetry run streamlit run streamlit_app.py
   ```

5. **Open your browser:**
   - Go to the URL shown in the terminal (usually http://localhost:8501)

---

## Prerequisites

- Python 3.11+
- Poetry
- OpenAI API key
- Gemini API key
- LlamaParse API key

---

## Project Structure

```
.
├── src/assurity_demo/        # Core business logic, models, pipeline
├── res/                     # Pre-processed claims, OCR outputs, sample data
├── streamlit_app.py         # Main Streamlit app entry point
├── .env.example             # Environment variable template
├── pyproject.toml           # Dependencies & config
└── README.md
```

---

## Development

- Install dev dependencies:
  ```bash
  poetry install --with dev
  ```
- Run pre-commit hooks:
  ```bash
  pre-commit install
  ```

---