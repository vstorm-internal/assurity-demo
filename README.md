## GPT-4o Vision Integration

The system now includes direct image processing using OpenAI's GPT-4o vision capabilities through LangChain.
Instead of relying solely on Tesseract OCR, the system can now:

1. Send document images directly to GPT-4o
2. Have the model perform OCR, classification, and data extraction in one step
3. Use this approach as a fallback when traditional methods have low confidence

To enable this feature:

1. Install additional dependencies:
   ```bash
   poetry add langchain langchain-openai
   ```

2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Ensure your OpenAI account has access to GPT-4o

## Donut Document Understanding Model

The system now incorporates ClovaAI's Donut (Document Understanding Transformer) model for document classification and information extraction. Donut is specifically designed for document understanding tasks and can directly process document images without requiring separate OCR.

### Features:

1. **End-to-end document understanding**: Processes document images directly
2. **Document classification**: Identifies document types (invoices, forms, etc.)
3. **Information extraction**: Extracts structured data based on document type
4. **Complementary to OCR**: Can operate alongside or as a fallback for traditional OCR

### Installation

```bash
poetry add transformers timm sentencepiece
```

### Usage

The pipeline automatically attempts to use Donut:
- After vision-based classification but before OCR for document type identification
- As a fallback for information extraction when other methods have mediocre confidence
- Before resorting to more expensive LLM-based methods

### Evaluation

You can evaluate the Donut model on your documents:

```bash
python evaluate_donut.py --dir path/to/documents --output results.json
```