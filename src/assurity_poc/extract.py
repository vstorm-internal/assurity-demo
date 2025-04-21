import json
from pathlib import Path

from fire import Fire
from llama_cloud.core.api_error import ApiError
from llama_cloud_services import LlamaExtract

from assurity_poc.config import get_settings
from assurity_poc.models.policy import Policy

settings = get_settings()


def extract(image_path: str):
    extractor = LlamaExtract(api_key=settings.llamaparse_api_key)

    try:
        existing_agent = extractor.get_agent(name="policy-extractor")
        if existing_agent:
            extractor.delete_agent(existing_agent.id)
    except ApiError as e:
        if e.status_code == 404:
            pass
        else:
            raise

    agent = extractor.create_agent(name="policy-extractor", data_schema=Policy)
    # Check if the image exists
    if not Path(image_path).exists():
        print(f"Error: File not found at {image_path}")
        return

    # Process the image
    try:
        print(f"Extracting policy from image: {image_path}")
        result = agent.extract(image_path)
        with open(
            settings.data_dir
            / f"results/extract/{image_path.parent.parent}/policy.json",
            "a",
        ) as f:
            json.dump(result.data, f)
        print(
            f"Policy extracted and saved to {settings.data_dir / f'results/extract/{image_path.parent.parent}/policy.json'}"
        )
    except Exception as e:
        print(f"Error extracting policy from image: {str(e)}")


if __name__ == "__main__":
    Fire(extract)
