import re

from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

from insurance_claims_ocr.utils import convert_pdf_to_image

class DonutClassifier:
    def __init__(self):
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def classify_image(self, image_path: str) -> str:
        # load document image
        if str(image_path).endswith(".pdf"):
            image = convert_pdf_to_image(image_path)
        else:
            image = Image.open(image_path)

        # prepare decoder inputs
        task_prompt = "<s_rvlcdip>"
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        outputs = self.model.generate(
            pixel_values.to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device),
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        return self.processor.token2json(sequence)