import os

import cv2
import PIL
import fire
import numpy as np
import imagehash

from PIL import Image
from skimage.metrics import structural_similarity as ssim

from assurity_poc.config import get_settings
from assurity_poc.utils.file import convert_pdf_to_image

PIL.Image.MAX_IMAGE_PIXELS = 9e99

os.environ["TOKENIZERS_PARALLELISM"] = "false"

settings = get_settings()


class ImageMatcher:
    def __call__(self, image1: str, image2: str):
        return self.run(image1, image2)

    def resize_to_match(self, img1, img2):
        """Resize the larger image to match the dimensions of the smaller one"""
        # Convert PIL Image to numpy array if needed
        if not isinstance(img1, np.ndarray):
            img1 = np.array(img1)
        if not isinstance(img2, np.ndarray):
            img2 = np.array(img2)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 != h2 or w1 != w2:
            # Resize the larger image to match the smaller one
            if h1 * w1 > h2 * w2:
                img1 = cv2.resize(img1, (w2, h2))
            else:
                img2 = cv2.resize(img2, (w1, h1))

        return img1, img2

    # SSIM - Structural Similarity Index
    def calculate_ssim(self, img1, img2):
        # Ensure both images are numpy arrays
        if not isinstance(img1, np.ndarray):
            img1 = np.array(img1)
        if not isinstance(img2, np.ndarray):
            img2 = np.array(img2)

        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Resize images to match dimensions
        img1, img2 = self.resize_to_match(img1, img2)

        # Double check dimensions match
        if img1.shape != img2.shape:
            raise ValueError(f"Image dimensions still don't match after resize: {img1.shape} vs {img2.shape}")

        score, _ = ssim(img1, img2, full=True)
        return score

    # PSNR - Peak Signal-to-Noise Ratio
    def calculate_psnr(self, img1, img2):
        # Remove the redundant imread since images are already loaded
        img1, img2 = self.resize_to_match(img1, img2)

        # Ensure both images have the same number of channels
        if len(img1.shape) != len(img2.shape):
            if len(img1.shape) == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if len(img2.shape) == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        psnr = cv2.PSNR(img1, img2)
        return psnr

    # imagehash - Perceptual hash comparison
    def calculate_hash(self, img1, img2):
        # Convert numpy arrays to PIL Images for imagehash
        if isinstance(img1, np.ndarray):
            img1 = Image.fromarray(img1)
        if isinstance(img2, np.ndarray):
            img2 = Image.fromarray(img2)

        hash1 = imagehash.average_hash(img1)
        hash2 = imagehash.average_hash(img2)

        diff = hash1 - hash2
        return diff

    def classify_image(self, img1, img2):
        ssim_score = self.calculate_ssim(img1, img2)  # 0-1, 1 is perfect match
        psnr_score = self.calculate_psnr(img1, img2)  # higher values indicate better image quality and less noise
        hash_diff = self.calculate_hash(img1, img2)  # lower values suggest more similarity

        return {
            "ssim_score": ssim_score,
            "psnr_score": psnr_score,
            "hash_diff": hash_diff,
            # "is_similar": ssim_score > settings.ssim_threshold
            # and psnr_score >= settings.psnr_threshold
            # and hash_diff < settings.hash_threshold,
        }

    def run(self, image1: str, image2: str):
        if str(image1).endswith(".pdf"):
            img1 = convert_pdf_to_image(image1)
        else:
            img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)

        if str(image2).endswith(".pdf"):
            img2 = convert_pdf_to_image(image2)
        else:
            img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            raise ValueError("Failed to load one or both images")

        result = self.classify_image(img1, img2)
        return result


if __name__ == "__main__":
    fire.Fire(ImageMatcher)
