import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from assurity_poc.utils import convert_pdf_to_image


class DITClassifier:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/dit-base-finetuned-rvlcdip"
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            "microsoft/dit-base-finetuned-rvlcdip"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def classify_image(self, image_path: str) -> str:
        # load document image
        if str(image_path).endswith(".pdf"):
            image = convert_pdf_to_image(image_path)
        else:
            image = Image.open(image_path)

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits

        # model predicts one of the 16 RVL-CDIP classes
        predicted_class_idx = logits.argmax(-1).item()
        return self.model.config.id2label[predicted_class_idx]
