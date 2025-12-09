<h1 align="center">🐾 VetRAG — Veterinary Retrieval-Augmented Generation System</h1>

<p align="center">
  <strong>CS410 Final Project — UIUC</strong><br>
  A multi-aspect RAG system for veterinary clinical decision support.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Groq-LLM-orange">
  <img src="https://img.shields.io/badge/RAG-System-green">
  <img src="https://img.shields.io/badge/FAISS-Indexing-blue">
</p>

---

## 📘 Overview

**VetRAG** is a modular Retrieval-Augmented Generation system designed for assisting with veterinary clinical reasoning.  
It extends baseline RAG with:

- Hybrid dense + BM25 search  
- BGE reranker  
- LLM-based multi-query decomposition  
- Multi-aspect evidence fusion  
- Structured clinical output generation  
- Automated evaluation across correctness, hallucination, and evidence relevance  

This system is fully reproducible and engineered using clean Python modules.

---


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
```

---

## Installation

1. Clone the repository


git clone https://github.com/simpsonTsai/cs410-rag-project
cd cs410-rag-project


2. Create a virtual environment

python3 -m venv venv
source venv/bin/activate      # Mac/Linux

venv\Scripts\activate         # Windows


3. Install dependencies

pip install -r requirements.txt


4. API Keys Setup
#####
---------------------------------
5. How to Run the System

python src/run.py
```text
This script:

Loads the PDF

Builds chunks

Builds embeddings and FAISS index

Initializes the retriever

Initializes Groq LLM client

Runs:

Baseline RAG

Improved (query decomposition + multi-aspect fusion)

GPT-only (no retrieval)

Evaluates correctness, hallucination, and evidence relevance

Generates radar and bar charts

All results print directly to console and display visualizations.

```
