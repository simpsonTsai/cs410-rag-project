from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from config import TA_MODE, TA_MAX_EMBED


if TA_MODE:
    BGE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
else:
    BGE_MODEL_NAME ="sentence-transformers/all-MiniLM-L6-v2"


def build_bge_embeddings(docs: List[Dict[str, Any]]) -> np.ndarray:
    texts = [d["text"] for d in docs]
    if TA_MODE:
        texts = texts[:TA_MAX_EMBED]
    bge_embedder = SentenceTransformer(BGE_MODEL_NAME)
    embs = bge_embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")
    return embs


def build_faiss_index(embs: np.ndarray) -> faiss.Index:
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index
