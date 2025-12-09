from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
from groq import Groq

from prompts import (
    ClinicalCase,
    SYSTEM_PROMPT,
    build_case_query,
    build_clinical_prompt,
    build_clinical_prompt_improved,
    case_to_free_text,
)
from retriever import VetRetriever
from decomposer import decompose_case_query
from fusion import retrieve_multi_aspect
from evaluation import evaluate_system


def generate_answer_with_groq(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def rag_answer_case_baseline(
    client: Groq,
    retriever: VetRetriever,
    case: ClinicalCase,
) -> Dict[str, Any]:
    query_str = build_case_query(case)
    evidence_df = retriever.retrieve_with_rerank(query_str)
    user_prompt = build_clinical_prompt(case, query_str, evidence_df)
    answer = generate_answer_with_groq(client, SYSTEM_PROMPT, user_prompt)
    return {
        "case": case,
        "query": query_str,
        "evidence": evidence_df,
        "answer": answer,
    }


def rag_answer_case_improved(
    client: Groq,
    retriever: VetRetriever,
    case: ClinicalCase,
) -> Dict[str, Any]:
    main_query = build_case_query(case)
    sub_queries = decompose_case_query(client, main_query)
    evidence_df = retrieve_multi_aspect(retriever, sub_queries)
    user_prompt = build_clinical_prompt_improved(case, main_query, sub_queries, evidence_df)
    answer = generate_answer_with_groq(client, SYSTEM_PROMPT, user_prompt)
    return {
        "case": case,
        "query": main_query,
        "sub_queries": sub_queries,
        "evidence": evidence_df,
        "answer": answer,
    }


def gpt_only_answer_case(client: Groq, case: ClinicalCase) -> Dict[str, Any]:
    """
    GPT-only baseline: no retrieval, only LLM prior.
    """
    case_text = case_to_free_text(case)
    system_prompt = (
        "You are a veterinary clinician. Answer based ONLY on your own general knowledge. "
        "You do NOT have access to any external documents. If unsure, say so."
    )
    user_prompt = f"Clinical case:\n{case_text}\n\nTask: Provide a structured clinical reasoning summary."

    answer = generate_answer_with_groq(
        client,
        system_prompt,
        user_prompt,
    )
    return {
        "case": case,
        "query": case_text,
        "evidence": pd.DataFrame(),  # no retrieval
        "answer": answer,
    }


@dataclass
class EvalCase:
    case_id: str
    case: ClinicalCase
    gold_answer: str


def build_eval_df_for_system(
    client: Groq,
    retriever: VetRetriever,
    eval_cases: List[EvalCase],
    system_name: str,
) -> pd.DataFrame:
    rows = []
    for ec in eval_cases:
        if system_name == "baseline":
            out = rag_answer_case_baseline(client, retriever, ec.case)
        elif system_name == "improved":
            out = rag_answer_case_improved(client, retriever, ec.case)
        elif system_name == "gpt_only":
            out = gpt_only_answer_case(client, ec.case)
        else:
            raise ValueError(f"Unknown system_name={system_name}")

        ev_df = out.get("evidence", pd.DataFrame())
        evidence_texts = ev_df["text"].tolist() if not ev_df.empty else []

        rows.append({
            "case_id": ec.case_id,
            "system": system_name,
            "query": out["query"],
            "answer": out["answer"],
            "evidence_texts": evidence_texts,
            "gold_answer": ec.gold_answer,
        })
    return pd.DataFrame(rows)


def build_default_eval_cases() -> List[EvalCase]:
    return [
        # CASE 1
        EvalCase(
            case_id="cat_acute_sneeze",
            case=ClinicalCase(
                species="cat",
                age_years=4,
                chronicity="acute",
                key_signs=["sneezing", "nasal discharge"],
                other_notes="indoor-only, onset 2 days ago, mild lethargy",
                problem_title="Cat with acute sneezing and nasal discharge",
            ),
            gold_answer=(
                "Acute sneezing and nasal discharge in young adult cats are most commonly caused by "
                "upper respiratory infections, especially feline herpesvirus-1 and calicivirus. "
                "Mechanism involves inflammation of the nasal mucosa. Differential diagnoses include "
                "Chlamydia felis, Mycoplasma, foreign body, and less commonly dental disease or neoplasia. "
                "Diagnostics include PCR testing, cytology, and imaging if obstruction is suspected. "
                "Management is primarily supportive care, hydration, and antiviral or antibiotic therapy as indicated."
            ),
        ),
        # CASE 2
        EvalCase(
            case_id="cat_chronic_cough",
            case=ClinicalCase(
                species="cat",
                age_years=6,
                chronicity="chronic",
                key_signs=["coughing", "wheezing"],
                other_notes="episodes triggered by exercise",
                problem_title="Cat with chronic cough and wheezing",
            ),
            gold_answer=(
                "Chronic cough with expiratory wheeze in cats is strongly suggestive of feline asthma or chronic bronchitis. "
                "Mechanism involves lower airway inflammation and bronchoconstriction. "
                "Differentials include heartworm disease, parasitic lung migration, pneumonia, and neoplasia. "
                "Diagnostics include thoracic radiographs, airway cytology, and heartworm testing. "
                "Treatment includes corticosteroids and bronchodilators."
            ),
        ),
        # CASE 3
        EvalCase(
            case_id="dog_acute_vomit",
            case=ClinicalCase(
                species="dog",
                age_years=2,
                chronicity="acute",
                key_signs=["vomiting"],
                other_notes="ate garbage earlier today",
                problem_title="Dog with acute vomiting",
            ),
            gold_answer=(
                "Acute vomiting in young dogs with dietary indiscretion is most commonly due to gastroenteritis. "
                "Mechanisms include gastric irritation or inflammation. "
                "Differentials include foreign body obstruction, pancreatitis, toxin ingestion, and infectious causes. "
                "Diagnostics include abdominal palpation, radiographs, and bloodwork. "
                "Treatment includes fluids, antiemetics, and dietary rest unless obstruction is suspected."
            ),
        ),
        # CASE 4
        EvalCase(
            case_id="cat_flutd",
            case=ClinicalCase(
                species="cat",
                age_years=5,
                chronicity="acute",
                key_signs=["straining to urinate", "frequent trips to litter box"],
                other_notes="crying when attempting to urinate",
                problem_title="Cat with dysuria and urinary straining",
            ),
            gold_answer=(
                "Acute dysuria and frequent attempts to urinate in adult male cats strongly suggest feline lower urinary tract disease. "
                "Mechanisms include urethral obstruction or sterile cystitis. "
                "Differentials: urethral plug, uroliths, idiopathic cystitis, UTI. "
                "Diagnostics: bladder palpation, urinalysis, imaging for stones. "
                "Treatment: immediate relief of obstruction if present, analgesia, fluids, and urinary diet as indicated."
            ),
        ),
        # CASE 5
        EvalCase(
            case_id="dog_ckd",
            case=ClinicalCase(
                species="dog",
                age_years=10,
                chronicity="chronic",
                key_signs=["increased thirst", "increased urination", "weight loss"],
                other_notes="decreased appetite over the past month",
                problem_title="Dog with PU/PD and weight loss",
            ),
            gold_answer=(
                "Chronic polyuria, polydipsia, and weight loss in older dogs are commonly associated with chronic kidney disease. "
                "Mechanisms involve reduced renal concentrating ability and progressive nephron loss. "
                "Differentials include diabetes mellitus, hyperadrenocorticism, pyelonephritis, and hypercalcemia. "
                "Diagnostics: CBC, chemistry panel, urinalysis, SDMA, imaging. "
                "Management includes renal diets, fluids, phosphate binders, and blood pressure control."
            ),
        ),
        # CASE 6
        EvalCase(
            case_id="dog_seizure",
            case=ClinicalCase(
                species="dog",
                age_years=3,
                chronicity="acute",
                key_signs=["seizure"],
                other_notes="normal between episodes",
                problem_title="Dog with new-onset seizure",
            ),
            gold_answer=(
                "New-onset seizure in a young adult dog is commonly due to idiopathic epilepsy. "
                "Mechanism involves abnormal neuronal excitability. "
                "Differentials include toxin exposure, metabolic disease, infection, or structural brain lesions. "
                "Diagnostics: bloodwork, bile acids, MRI if structural disease suspected. "
                "Treatment includes anticonvulsants such as phenobarbital or levetiracetam."
            ),
        ),
        # CASE 7
        EvalCase(
            case_id="dog_dermatitis",
            case=ClinicalCase(
                species="dog",
                age_years=4,
                chronicity="chronic",
                key_signs=["itching", "red skin"],
                other_notes="seasonal worsening",
                problem_title="Dog with chronic pruritus",
            ),
            gold_answer=(
                "Chronic pruritus with seasonal worsening strongly suggests atopic dermatitis. "
                "Mechanisms include hypersensitivity reactions and barrier dysfunction. "
                "Differentials: flea allergy dermatitis, food allergy, secondary bacterial or yeast infection. "
                "Diagnostics: flea control trial, skin scrapings, cytology, diet trial. "
                "Treatment: anti-itch therapy, allergen avoidance, antimicrobial therapy if needed."
            ),
        ),
        # CASE 8
        EvalCase(
            case_id="cat_conjunctivitis",
            case=ClinicalCase(
                species="cat",
                age_years=1,
                chronicity="acute",
                key_signs=["red eye", "ocular discharge"],
                problem_title="Cat with acute conjunctivitis",
            ),
            gold_answer=(
                "Acute conjunctivitis in young cats is commonly due to feline herpesvirus-1 or Chlamydia felis. "
                "Mechanisms involve conjunctival inflammation and infection. "
                "Differentials include Mycoplasma, foreign body, allergy, or corneal ulceration. "
                "Diagnostics: fluorescein stain, PCR testing. "
                "Treatment: topical antibiotics, antiviral therapy, and supportive care."
            ),
        ),
        # CASE 9
        EvalCase(
            case_id="dog_lameness",
            case=ClinicalCase(
                species="dog",
                age_years=7,
                chronicity="acute",
                key_signs=["hindlimb lameness"],
                problem_title="Dog with acute hindlimb lameness",
            ),
            gold_answer=(
                "Acute hindlimb lameness in middle-aged dogs commonly results from cranial cruciate ligament rupture. "
                "Mechanisms include ligament degeneration and sudden overload. "
                "Differentials: luxating patella, hip dysplasia flare, iliopsoas strain, fractures. "
                "Diagnostics: orthopedic exam, drawer test, radiographs. "
                "Treatment: surgical stabilization or conservative management depending on severity."
            ),
        ),
        # CASE 10
        EvalCase(
            case_id="cat_hyperthyroid",
            case=ClinicalCase(
                species="cat",
                age_years=13,
                chronicity="chronic",
                key_signs=["weight loss", "increased appetite", "restlessness"],
                problem_title="Elderly cat with weight loss and polyphagia",
            ),
            gold_answer=(
                "Weight loss with polyphagia and restlessness in older cats strongly suggests hyperthyroidism. "
                "Mechanisms include excess thyroid hormone and increased metabolic rate. "
                "Differentials: diabetes mellitus, GI disease, or renal disease. "
                "Diagnostics: total T4 measurement, CBC, chemistry panel. "
                "Treatment includes methimazole, radioactive iodine therapy, or surgical thyroidectomy."
            ),
        ),
    ]


def run_full_experiment(client: Groq, retriever: VetRetriever):
    """
    Run all three systems (baseline / improved / GPT-only)
    on the default evaluation set, compute metrics, and return
    evaluation DataFrames and aggregate scores.
    """
    eval_cases = build_default_eval_cases()

    print("Running Baseline RAG...")
    df_baseline = build_eval_df_for_system(client, retriever, eval_cases, "baseline")

    print("Running Improved RAG...")
    df_improved = build_eval_df_for_system(client, retriever, eval_cases, "improved")

    print("Running GPT-only...")
    df_gptonly = build_eval_df_for_system(client, retriever, eval_cases, "gpt_only")

    print("Evaluating Baseline...")
    df_baseline_eval = evaluate_system(client, df_baseline)

    print("Evaluating Improved...")
    df_improved_eval = evaluate_system(client, df_improved)

    print("Evaluating GPT-only...")
    df_gpt_eval = evaluate_system(client, df_gptonly)

    print("\n=== BASELINE ===")
    print(df_baseline_eval.mean(numeric_only=True))

    print("\n=== IMPROVED ===")
    print(df_improved_eval.mean(numeric_only=True))

    print("\n=== GPT-only ===")
    print(df_gpt_eval.mean(numeric_only=True))

    systems = ["Baseline", "Improved", "GPT-only"]

    correctness_vals = [
        df_baseline_eval["correctness_score"].dropna().astype(float).mean(),
        df_improved_eval["correctness_score"].dropna().astype(float).mean(),
        df_gpt_eval["correctness_score"].dropna().astype(float).mean(),
    ]

    hallucination_vals = [
        df_baseline_eval["hallucination_score"].dropna().astype(float).mean(),
        df_improved_eval["hallucination_score"].dropna().astype(float).mean(),
        df_gpt_eval["hallucination_score"].dropna().astype(float).mean(),
    ]

    relevance_vals = [
        df_baseline_eval["evidence_relevance"].dropna().astype(float).mean(),
        df_improved_eval["evidence_relevance"].dropna().astype(float).mean(),
        df_gpt_eval["evidence_relevance"].dropna().astype(float).mean(),
    ]
    # scale 0–5 → 0–10
    relevance_vals = [val * 2 for val in relevance_vals]

    return (
        systems,
        correctness_vals,
        hallucination_vals,
        relevance_vals,
        df_baseline_eval,
        df_improved_eval,
        df_gpt_eval,
    )
