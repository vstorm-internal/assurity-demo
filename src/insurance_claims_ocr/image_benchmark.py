import os
from pathlib import Path
import random

import pandas as pd
import PIL

from tqdm import tqdm

from insurance_claims_ocr.donut import DonutClassifier
from insurance_claims_ocr.dit import DITClassifier
from insurance_claims_ocr.image_matching import ImageMatcher
from insurance_claims_ocr.utils import iterate_over_files

PIL.Image.MAX_IMAGE_PIXELS=9e99

os.environ['TOKENIZERS_PARALLELISM']="false"

DATA_DIR = Path('./res/image_match/inputs')
UB04_BLANK_PATH = Path('./res/image_match/blanks/UB04_blank.png')
HCFA1500_BLANK_PATH = Path('./res/image_match/blanks/HCFA1500_blank.png')

def run_benchmark():
    donut_classifier = DonutClassifier()
    dit_classifier = DITClassifier()

    image_matcher = ImageMatcher()

    files = list(iterate_over_files(DATA_DIR))

    results_dict = {}
    for fp in tqdm(files):
        subresults_dict = {}
        try:
            subresults_dict['image_similarity'] = {
                'UB04': image_matcher(fp, UB04_BLANK_PATH),
                'HCFA1500': image_matcher(fp, HCFA1500_BLANK_PATH),
            }
            subresults_dict['donut_classification'] = donut_classifier.classify_image(fp)
            subresults_dict['dit_classification'] = dit_classifier.classify_image(fp)
        except Exception as e:
            print(f"Error processing file {fp.name}: {str(e)}")
            continue
        results_dict[fp.name] = subresults_dict

    rows = []
    for filename, details in results_dict.items():
        row = {'filename': filename}
        img_sim = details.get('image_similarity', {})
        ub04 = img_sim.get('UB04', {})
        hcfa = img_sim.get('HCFA1500', {})
        donut = details.get('donut_classification', {})
        dit = details.get('dit_classification', {})
        
        row['image_similarity.UB04.ssim_score'] = ub04.get('ssim_score', '')
        row['image_similarity.UB04.psnr_score'] = ub04.get('psnr_score', '')
        row['image_similarity.UB04.hash_diff'] = ub04.get('hash_diff', '')
        row['image_similarity.UB04.is_similar'] = ub04.get('is_similar', '')
        row['image_similarity.HCFA1500.ssim_score'] = hcfa.get('ssim_score', '')
        row['image_similarity.HCFA1500.psnr_score'] = hcfa.get('psnr_score', '')
        row['image_similarity.HCFA1500.hash_diff'] = hcfa.get('hash_diff', '')
        row['donut_classification.class'] = donut.get('class', '')
        row['dit_classification.class'] = dit
        rows.append(row)

    df = pd.DataFrame(rows)

    df_name = f'image_similarity_results.csv'
    df.to_csv(df_name, index=False)

if __name__ == '__main__':
    run_benchmark()
