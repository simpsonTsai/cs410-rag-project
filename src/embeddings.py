from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

BGE_MODEL_NAME = "BAAI/bge-m3"
TA_MAX_EMBED = 50   # TA quick-test cutoff


def build_bge_embeddings(docs: List[Dict[str, Any]]) -> np.ndarray:
    texts = [d["text"] for d in docs]
    texts = texts[:TA_MAX_EMBED] # TA-friendly cutoff (CPU safe)
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
