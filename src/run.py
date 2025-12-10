import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from chunks import load_pdf_text, build_chunks
from embeddings import build_bge_embeddings, build_faiss_index
from retriever import VetRetriever
from agent import run_full_experiment
from plotting import (
    plot_correctness_bar,
    plot_hallucination_bar,
    plot_relevance_bar,
    plot_radar_chart,
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ta_mode", action="store_true", help="Enable TA quick test mode (CPU fast)")
args = parser.parse_args()
TA_MODE = args.ta_mode

def init_vetrag_pipeline(pdf_path: str | None = None):
    base_dir = Path(__file__).resolve().parent.parent
    if pdf_path is None:
        pdf_path = str(base_dir / "data" / "databook.pdf")

    print("Loading PDF...")
    pages = load_pdf_text(pdf_path)
    print(f"  Loaded {len(pages)} pages.")

    print("Building chunks...")
    docs = build_chunks(pages)
    print(f"  Built {len(docs)} chunks.")

    print("Building BGE embeddings...")
    embs = build_bge_embeddings(docs)

    print("Building FAISS index...")
    faiss_index = build_faiss_index(embs)

    print("Initializing retriever...")
    retriever = VetRetriever(docs, embs, faiss_index)

    print("Initializing Groq client...")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Please create a .env file with GROQ_API_KEY=...")

    client = Groq(api_key=api_key)

    print("✅ VetRAG pipeline initialized.")
    return client, retriever


def main():
    load_dotenv()

    client, retriever = init_vetrag_pipeline()

    (
        systems,
        correctness_vals,
        hallucination_vals,
        relevance_vals,
        df_baseline_eval,
        df_improved_eval,
        df_gpt_eval,
    ) = run_full_experiment(client, retriever)

    # Bar charts
    plot_correctness_bar(systems, correctness_vals)
    plot_hallucination_bar(systems, hallucination_vals)
    plot_relevance_bar(systems, relevance_vals)

    # Radar chart
    plot_radar_chart(
        [
            correctness_vals[0],
            hallucination_vals[0],
            relevance_vals[0],
        ],
        [
            correctness_vals[1],
            hallucination_vals[1],
            relevance_vals[1],
        ],
        [
            correctness_vals[2],
            hallucination_vals[2],
            relevance_vals[2],
        ],
    )


if __name__ == "__main__":
    main()
