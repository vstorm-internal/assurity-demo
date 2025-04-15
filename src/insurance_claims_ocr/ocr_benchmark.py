import os
from pathlib import Path
import random

import pandas as pd
import PIL

from tqdm import tqdm

from insurance_claims_ocr.ocr_pipeline import OCRPipeline
from insurance_claims_ocr.utils import iterate_over_files

PIL.Image.MAX_IMAGE_PIXELS=9e99

os.environ['TOKENIZERS_PARALLELISM']="false"

DATA_DIR = Path('./res/data/converted/')
UB04_BLANK_PATH = Path('./res/image_match/UB04_blank.png')
HCFA1500_BLANK_PATH = Path('./res/image_match/HCFA1500_blank.png')

def run_benchmark():
    ocr_pipe = OCRPipeline()

    files = list(iterate_over_files(DATA_DIR))

    for x in (4, 10, 50, 100, 500):
        if x != 500:
            continue
        subset = random.sample(files, x)
        results_dict = {}
        for fp in tqdm(subset):
            subresults_dict = {}
            try:
                subresults_dict['ocr'] = ocr_pipe.process_image(fp)
            except Exception as e:
                print(f"Error processing file {fp.name}: {str(e)}")
                continue
            results_dict[fp.name] = subresults_dict

        rows = []
        for filename, details in results_dict.items():
            row = {'filename': filename}
            ocr = details.get('ocr', {})
            sim = ocr.get('similarity', {})
            
            row['ocr.gpt_text'] = ocr.get('gpt_text', '')
            row['ocr.gemini_text'] = ocr.get('gemini_text', '')
            row['text_similarity.levenshtein'] = sim.get('levenshtein', '')
            row['text_similarity.jaccard'] = sim.get('jaccard', '')
            row['text_similarity.tfidf'] = sim.get('tfidf', '')
            row['text_similarity.embedding'] = sim.get('embedding', '')
            row['text_similarity.overall'] = sim.get('overall', '') 

            rows.append(row)

        df = pd.DataFrame(rows)

        df_name = f'ocr_results_{x}.csv'
        df.to_csv(df_name, index=False)

if __name__ == '__main__':
    run_benchmark()
