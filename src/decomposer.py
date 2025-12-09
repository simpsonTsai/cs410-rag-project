from typing import List
import re

from groq import Groq


def decompose_case_query(
    client: Groq,
    main_query: str,
    model: str = "llama-3.3-70b-versatile",
) -> List[str]:
    """
    Use LLM to decompose the main clinical query into 2–4 focused sub-queries.
    """
    system = (
        "You are an information retrieval expert for veterinary clinical cases. "
        "Decompose a single complex clinical query into 2–4 focused English sub-queries."
    )
    user = f"Original query:\n{main_query}\n\nWrite 2–4 focused sub-queries, one per line."

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=512,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    raw = resp.choices[0].message.content or ""
    sub_queries: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*\d\.\)\s]+", "", line).strip()
        if line:
            sub_queries.append(line)
    if not sub_queries:
        sub_queries = [main_query]
    return sub_queries
