from typing import List, Dict, Any

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss

from embeddings import BGE_MODEL_NAME


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class VetRetriever:
    """
    Hybrid retriever:
    - BM25 (rank_bm25)
    - Dense BGE-M3 + FAISS
    - BGE reranker
    """

    def __init__(self, docs: List[Dict[str, Any]], embs: np.ndarray, faiss_index: faiss.Index):
        self.docs = docs
        self.texts = [d["text"] for d in docs]
        self.embs = embs
        self.faiss_index = faiss_index

        # BM25
        self.corpus_tokens = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # Dense embedder for queries
        self.query_embedder = SentenceTransformer(BGE_MODEL_NAME)

        # Neural reranker
        self.reranker = CrossEncoder("BAAI/bge-reranker-large", max_length=512)

    # ----- dense / BM25 / hybrid -----

    def dense_search(self, query: str, k: int = 80) -> pd.DataFrame:
        q_emb = self.query_embedder.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")
        scores, idx = self.faiss_index.search(q_emb, k)
        idx = idx[0]
        scores = scores[0]

        rows = []
        for score, i in zip(scores, idx):
            d = self.docs[i]
            rows.append({
                "doc_id": d["doc_id"],
                "page": d["page"],
                "text": d["text"],
                "tag": d["tag"],
                "dense_score": float(score),
            })
        return pd.DataFrame(rows)

    def bm25_search(self, query: str, k: int = 80) -> pd.DataFrame:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        idx_sorted = np.argsort(scores)[::-1][:k]

        rows = []
        for i in idx_sorted:
            d = self.docs[i]
            rows.append({
                "doc_id": d["doc_id"],
                "page": d["page"],
                "text": d["text"],
                "tag": d["tag"],
                "bm25_score": float(scores[i]),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _minmax_norm(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([1.0] * len(series), index=series.index)
        return (series - mn) / (mx - mn)

    def hybrid_candidates(
        self,
        query: str,
        k_dense: int = 80,
        k_bm25: int = 80,
        alpha: float = 0.5,
        top_k: int = 30,
    ) -> pd.DataFrame:
        dense_df = self.dense_search(query, k=k_dense)
        bm25_df = self.bm25_search(query, k=k_bm25)

        merged = pd.merge(
            dense_df,
            bm25_df,
            on=["doc_id", "page", "text", "tag"],
            how="outer"
        ).fillna(0.0)

        merged["dense_score_norm"] = self._minmax_norm(merged["dense_score"])
        merged["bm25_score_norm"] = self._minmax_norm(merged["bm25_score"])
        merged["hybrid_score"] = (
            (1 - alpha) * merged["dense_score_norm"] +
            alpha * merged["bm25_score_norm"]
        )

        merged = merged.sort_values("hybrid_score", ascending=False).head(top_k)
        return merged.reset_index(drop=True)

    def rerank_with_bge(
        self,
        query: str,
        candidates: pd.DataFrame,
        top_k: int = 5,
        alpha_hybrid: float = 0.5
    ) -> pd.DataFrame:
        """
        Neural reranking on top of hybrid candidates.
        """
        texts = candidates["text"].tolist()
        pairs = [[query, t] for t in texts]
        scores = self.reranker.predict(pairs)

        cand = candidates.copy()
        cand["rerank_score"] = scores
        cand["combined_score"] = (
            alpha_hybrid * cand["hybrid_score"]
            + (1 - alpha_hybrid) * cand["rerank_score"]
        )
        cand = cand.sort_values("combined_score", ascending=False).head(top_k)
        return cand.reset_index(drop=True)

    def retrieve_with_rerank(
        self,
        query: str,
        k_dense: int = 80,
        k_bm25: int = 80,
        alpha: float = 0.5,
        top_k_candidates: int = 30,
        top_k_final: int = 5,
    ) -> pd.DataFrame:
        candidates = self.hybrid_candidates(
            query,
            k_dense=k_dense,
            k_bm25=k_bm25,
            alpha=alpha,
            top_k=top_k_candidates,
        )
        if candidates.empty:
            return candidates
        return self.rerank_with_bge(query, candidates, top_k=top_k_final)
