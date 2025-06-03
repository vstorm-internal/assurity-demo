import re

import torch

from Levenshtein import ratio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_text(text):
    # 1) Lowercase
    text = text.lower()
    # 2) Remove Markdown table lines (---) and header separators (|---)
    text = re.sub(r"\|?-+\|?", " ", text)
    # 3) Remove remaining table pipes '|'
    text = text.replace("|", " ")
    # 4) Remove Markdown emphasis symbols (e.g. '**')
    text = re.sub(r"\*{1,}", " ", text)
    # 5) Remove headings like '# some title'
    text = re.sub(r"^[#]+\s+", "", text, flags=re.MULTILINE)
    # 6) Remove leading bullet chars like '- ', '* ', etc.
    text = re.sub(r"^[\-*]\s+", "", text, flags=re.MULTILINE)
    # 7) Remove other punctuation except alphanumeric/whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    # 8) Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Text similarity
def compute_levenshtein_ratio(text1, text2):
    return ratio(text1, text2)


def jaccard_similarity(a, b):
    set_a, set_b = set(a.split()), set(b.split())
    return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0


def tfidf_cosine(a, b):
    vec = TfidfVectorizer().fit_transform([a, b])
    return cosine_similarity(vec[0], vec[1])[0][0]


## Embedding-based similarity
def embedding_cosine(a, b):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec1 = model.encode(a, convert_to_tensor=True)
    vec2 = model.encode(b, convert_to_tensor=True)
    emb_score = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()
    return emb_score


def overall_score(levenshtein: float, jaccard: float, tfidf: float, embedding: float):
    normalized_embedding = (embedding + 1) / 2  # normalize to [0, 1]
    return (levenshtein + jaccard + tfidf + normalized_embedding) / 4


def compute_text_similarity(text1: str, text2: str) -> dict[str, float]:
    levenshtein = compute_levenshtein_ratio(text1, text2)
    jaccard = jaccard_similarity(text1, text2)
    tfidf = tfidf_cosine(text1, text2)
    embedding = embedding_cosine(text1, text2)

    return {
        "levenshtein": levenshtein,
        "jaccard": jaccard,
        "tfidf": tfidf,
        "embedding": embedding,
        "overall": overall_score(
            levenshtein=levenshtein,
            jaccard=jaccard,
            tfidf=tfidf,
            embedding=embedding,
        ),
    }
