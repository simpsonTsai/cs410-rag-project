from typing import List

import pandas as pd

from .retriever import VetRetriever


def retrieve_multi_aspect(
    retriever: VetRetriever,
    sub_queries: List[str],
    k_dense: int = 80,
    k_bm25: int = 80,
    alpha: float = 0.5,
    top_k_candidates: int = 30,
    top_k_final: int = 5,
) -> pd.DataFrame:
    all_rows = []
    for sq in sub_queries:
        reranked = retriever.retrieve_with_rerank(
            sq,
            k_dense=k_dense,
            k_bm25=k_bm25,
            alpha=alpha,
            top_k_candidates=top_k_candidates,
            top_k_final=top_k_final,
        )
        if reranked.empty:
            continue
        reranked = reranked.copy()
        reranked["sub_query"] = sq
        all_rows.append(reranked)
    if not all_rows:
        return pd.DataFrame()
    merged = pd.concat(all_rows, ignore_index=True)
    merged = merged.drop_duplicates(subset=["doc_id", "page", "text"])
    merged = merged.sort_values("combined_score", ascending=False)
    return merged.reset_index(drop=True)
