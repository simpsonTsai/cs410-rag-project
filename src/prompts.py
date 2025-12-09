from dataclasses import dataclass
from typing import Optional, List

import pandas as pd


@dataclass
class ClinicalCase:
    species: str = "cat"
    age_years: Optional[float] = None
    chronicity: Optional[str] = None  # "acute", "chronic"
    key_signs: Optional[List[str]] = None
    other_notes: Optional[str] = None
    problem_title: Optional[str] = None


SYSTEM_PROMPT = "You are a veterinary clinical decision support assistant."


def build_case_query(case: ClinicalCase) -> str:
    parts: List[str] = []
    species = case.species or "cat"
    parts.append(species)
    if case.age_years is not None:
        parts.append(f"approximately {case.age_years}-year-old")
    if case.chronicity:
        parts.append(case.chronicity)
    parts.append("patient")
    if case.key_signs:
        parts.append("with " + ", ".join(case.key_signs))
    base = " ".join(parts)
    if case.other_notes:
        base += f". Additional notes: {case.other_notes}"
    base += ". What are the likely causes, mechanisms, and recommended diagnostic and treatment approaches?"
    return base


def format_sources_for_prompt(df: pd.DataFrame, max_chars_per_source: int = 800) -> str:
    lines = []
    for i, row in df.iterrows():
        text = row["text"]
        text = text.replace("\n", " ").strip()
        text = text[:max_chars_per_source]
        lines.append(f"[{i+1}] (page {row['page']}, tag: {row['tag']})\n{text}\n")
    return "\n".join(lines)


def build_clinical_prompt(case: ClinicalCase, query_str: str, evidence_df: pd.DataFrame) -> str:
    sources_str = format_sources_for_prompt(evidence_df)
    case_str = f"Problem: {case.problem_title or ''}\n\nMain query:\n{query_str}"
    return f"""You are reading a veterinary internal medicine textbook.

Clinical case:
{case_str}

Relevant evidence snippets:
{sources_str}

Task:
Provide a structured clinical summary including:
1. Summary of the case
2. Mechanism? (pathophysiology)
3. Where? (anatomical localization)
4. What? (most likely diagnostic categories / diseases)
5. Key differential diagnoses
6. Recommended diagnostic plan
7. Management and treatment considerations
"""


def build_clinical_prompt_improved(
    case: ClinicalCase,
    main_query: str,
    sub_queries: List[str],
    evidence_df: pd.DataFrame
) -> str:
    from .prompts import format_sources_for_prompt  # safe self-import for static checkers
    sources_str = format_sources_for_prompt(evidence_df)
    subq_str = "\n".join(f"- {sq}" for sq in sub_queries)
    case_str = f"Problem: {case.problem_title or ''}\n\nMain query:\n{main_query}"
    return f"""You are reading a veterinary internal medicine textbook.

Clinical case:
{case_str}

Decomposed retrieval sub-queries:
{subq_str}

Aggregated evidence snippets:
{sources_str}

Task:
Integrate information across all aspects to provide a structured clinical reasoning summary:
1. Case summary
2. Mechanism? (pathophysiology)
3. Where? (anatomical localization)
4. What? (most likely diagnostic categories / diseases)
5. Key differential diagnoses
6. Recommended diagnostic plan
7. Management and treatment considerations
"""


def case_to_free_text(case: ClinicalCase) -> str:
    parts = [f"Species: {case.species}"]
    if case.age_years is not None:
        parts.append(f"Age: {case.age_years} years")
    if case.chronicity:
        parts.append(f"Chronicity: {case.chronicity}")
    if case.key_signs:
        parts.append("Key signs: " + ", ".join(case.key_signs))
    if case.other_notes:
        parts.append("Other notes: " + case.other_notes)
    return "\n".join(parts)
