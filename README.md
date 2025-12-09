# CS410 RAG Project – Veterinary Clinical Decision Support

This repository contains my UIUC CS410 2025 FALL final project: a retrieval-augmented generation (RAG) agent for veterinary internal medicine cases, with query decomposition and multi-aspect retrieval.

The system compares three configurations:

- **Baseline RAG**: single-query retrieval + reranking
- **Improved RAG**: LLM-based query decomposition → multi-aspect retrieval → fusion
- **GPT-only**: no retrieval, pure LLM prior

It then evaluates them on a small set of synthetic clinical cases using LLM-based metrics:
correctness, hallucination, and evidence relevance.

---

## Project Structure

```text
cs410-rag-project/
│
├── src/
│   ├── chunks.py          # PDF loading + chunking
│   ├── embeddings.py      # BGE-M3 embeddings + FAISS index
│   ├── retriever.py       # BM25 + dense + hybrid + reranker
│   ├── decomposer.py      # LLM-based query decomposition
│   ├── prompts.py         # dataclasses + prompt templates
│   ├── fusion.py          # multi-aspect retrieval and fusion
│   ├── agent.py           # RAG pipelines + evaluation dataset
│   ├── evaluation.py      # LLM-based evaluation metrics
│   ├── plotting.py        # bar charts + radar chart
│   └── run.py             # main entry point
│
├── data/
│   └── databook.pdf       # veterinary internal medicine databook (textbook)
│
├── notebooks/
│   └── demo.ipynb         # optional Colab-style demo (not required for grading)
│
├── docs/
│   └── usage.md           # extended documentation (optional)
│
├── requirements.txt
├── .env.example
└── README.md
