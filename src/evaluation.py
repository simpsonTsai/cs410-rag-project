from typing import List

import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from groq import Groq

CORRECTNESS_RUBRIC = """
You are a strict evaluator of correctness in veterinary QA.

Grades:
A = fully correct and complete → 10
B = mostly correct with minor omissions → 8
C = partially correct → 6
D = mostly incorrect → 3
F = dangerous / wrong → 0

Return ONLY: A / B / C / D / F
"""

HALLUCINATION_SYSTEM_PROMPT = """
You evaluate how much of the answer is NOT supported by evidence.

Score 0–10:
0 = fully grounded
1–3 = minor unsupported additions
4–6 = partially unsupported
7–9 = largely unsupported
10 = answer contradicts or fabricates

Return ONLY:
score: <integer>
"""

EVIDENCE_RELEVANCE_SYSTEM_PROMPT = """
You are a senior veterinary clinician evaluating information relevance for clinical question answering.

Your job is to assess whether an evidence passage provides clinically meaningful support to answer the question.

Scoring rubric (0–5):
0 = irrelevant
1 = weakly related, not clinically useful
2 = somewhat related but generic
3 = helpful but incomplete
4 = clinically meaningful and helpful
5 = directly answers the question

Return ONLY:
score: <integer>
"""


def judge_correctness_once(
    client: Groq,
    question: str,
    answer: str,
    gold: str,
    model: str = "llama-3.1-8b-instant",
) -> int:
    user_prompt = f"""
Question:
{question}

System answer:
{answer}

Gold answer:
{gold}
"""
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": CORRECTNESS_RUBRIC},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    grade = raw[0].upper() if raw else "C"
    mapping = {"A": 10, "B": 8, "C": 6, "D": 3, "F": 0}
    return mapping.get(grade, 6)


def judge_correctness(
    client: Groq,
    question: str,
    answer: str,
    gold: str,
    model: str = "llama-3.1-8b-instant",
) -> float:
    scores = [
        judge_correctness_once(client, question, answer, gold, model)
        for _ in range(3)
    ]
    return float(sorted(scores)[1])  # median


def judge_hallucination_score(
    client: Groq,
    query: str,
    evidences: List[str],
    answer: str,
    model: str = "llama-3.1-8b-instant",
) -> int:
    ev_text = "\n---\n".join(evidences)
    user_prompt = f"""
Query:
{query}

Evidence:
{ev_text}

Answer:
{answer}
"""
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": HALLUCINATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    out = resp.choices[0].message.content or ""
    m = re.search(r"score\s*:\s*([0-9]+)", out)
    return int(m.group(1)) if m else 5


def judge_evidence_relevance(
    client: Groq,
    query: str,
    evidence: str,
    model: str = "llama-3.1-8b-instant",
) -> int:
    user_prompt = f"""
Query:
{query}

Evidence:
{evidence}
"""
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": EVIDENCE_RELEVANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    out = resp.choices[0].message.content or ""
    m = re.search(r"score\s*:\s*([0-5])", out)
    return int(m.group(1)) if m else 2


def evaluate_system(
    client: Groq,
    df: pd.DataFrame,
    model: str = "llama-3.1-8b-instant",
) -> pd.DataFrame:
    correctness_scores = []
    hallucination_scores = []
    relevance_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q = str(row["query"])
        ans = str(row["answer"])
        gold = str(row["gold_answer"])
        evs = row["evidence_texts"]

        correctness_scores.append(judge_correctness(client, q, ans, gold, model))

        if not evs:
            hallucination_scores.append(10.0)
            relevance_scores.append(None)
            continue

        ev_list = evs if isinstance(evs, list) else [evs]
        hallucination_scores.append(
            judge_hallucination_score(client, q, ev_list[:3], ans, model)
        )
        rel = [
            judge_evidence_relevance(client, q, evidence, model)
            for evidence in ev_list[:3]
        ]
        relevance_scores.append(float(np.mean(rel)))

    out = df.copy()
    out["correctness_score"] = correctness_scores
    out["hallucination_score"] = hallucination_scores
    out["evidence_relevance"] = relevance_scores
    return out
