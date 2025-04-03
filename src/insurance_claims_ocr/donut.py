import re
from typing import Any

from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from pathlib import Path
import json


processor = DonutProcessor.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2", use_fast=True
)
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2"
)


def _parse_model_output(output: str) -> dict[str, Any]:
    """
    Parse the model's output into structured data.

    Args:
        output: Raw model output text
        doc_type: Document type for type-specific parsing

    Returns:
        Structured data dictionary
    """
    # Donut typically outputs in a JSON-like format with special tokens
    # The exact parsing logic depends on the model's output format

    # Try to convert to proper JSON by fixing common issues
    # Replace OCR errors in JSON structure
    cleaned_output = output

    # Fix missing quotes around keys
    cleaned_output = re.sub(r"(\w+):", r'"\1":', cleaned_output)

    # Ensure quotes are consistent
    cleaned_output = cleaned_output.replace("'", '"')

    try:
        # Try to parse as JSON
        if cleaned_output.startswith("{") and cleaned_output.endswith("}"):
            data = json.loads(cleaned_output)
            return data
        else:
            # If not valid JSON, return as plain text
            return {"text": cleaned_output}
    except json.JSONDecodeError:
        # If JSON parsing fails, extract key-value pairs using regex
        results = {}
        pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
        matches = re.findall(pattern, cleaned_output)

        for key, value in matches:
            results[key] = value

        return results if results else {"text": cleaned_output}


def process_image(image_path):
    image = Image.open(image_path)

    pixel_values = processor(image, return_tensors="pt").pixel_values
    print(pixel_values.shape)

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    sequence = re.sub(
        r"<.*?>", "", sequence, count=1
    ).strip()  # remove first task start token

    return _parse_model_output(sequence)


if __name__ == "__main__":
    outputs = dict()
    for file in Path("./data/inputs/jpeg").glob("*.jpeg"):
        outputs[file.name] = process_image(file)

    with open("./data/donut_output.json", "w") as f:
        json.dump(outputs, f)
