<h1 align="center">🐾 VetRAG — Veterinary Retrieval-Augmented Generation System</h1>

<p align="center">
  <strong>2025 Fall CS410 Final Project — University of Illinois Urbana-Champaign</strong><br>
  <strong>Instructor: Prof. ChengXiang Zhai<br>
  <strong>Project Author: Chia Yang Tsai <br>
  <strong>NetID: ct68<br>
  <strong>Email: ct68@illinois.edu<br>
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
## 🐍 Python Version

This project is tested and supported on **Python 3.10**.

Please use Python **3.10.x** to ensure smooth installation of all dependencies
(e.g., `faiss-cpu` and `sentence-transformers`) without requiring system-level
build tools such as `cmake`.

### 1. Clone the repository

```bash
- git clone https://github.com/simpsonTsai/cs410-rag-project.git
- cd cs410-rag-project
```


### 2. Create a virtual environment
```bash
- python3 -m venv venv
```
#### Mac/Linux
```bash
- source venv/bin/activate    
```
#### Windows
```bash
- venv\Scripts\activate         
```

### 3. Install dependencies
```bash
- pip install -r requirements.txt
```

## API Keys Setup

### 1. Create your .env file:
```bash
- cp .env.example .env
```
### 2. Open .env and add your key:
```bash
- nano .env
```
```bash
- GROQ_API_KEY=your_key_here
```
After put in your GROQ API KEY:
1. Ctrl + 0 (store)
2. Enter
3. Ctrl + X (exit)

## Required Data
Place the veterinary reference book (PDF) inside the data/ folder:
data/databook.pdf



## How to Run the System
To run the entire system — baseline RAG, improved RAG, GPT-only baseline, evaluation metrics, and visualizations — run:
```bash
- python src/run.py
```
This performs:

1. PDF loading & chunking
2. BGE-M3 embeddings + FAISS index
3. Hybrid BM25 + dense retrieval
4. Cross-encoder reranking
5. Query decomposition (improved system)
6. Evidence fusion
7. LLM answer generation
8. Evaluation (correctness, hallucination, relevance)
9. Radar / bar charts visualization

## RAG System Variants

### Baseline RAG
- Single-query retrieval
- Hybrid BM25 + dense search
- Cross-encoder reranking

### Improved RAG
- LLM-based query decomposition (2–4 sub-queries)
- Multi-aspect evidence fusion
- Structured clinical reasoning

### GPT-only
- Answers using only the LLM prior
- No retrieval (serves as degenerate baseline)

## Evaluation Metrics

The evaluation compares baseline, improved, and GPT-only systems across:

- Correctness (0–10):
LLM-graded via rubric (A/B/C/D/F → numeric score).

- Hallucination (0–10, lower = better):
Measures unsupported or fabricated content.

- Evidence Relevance (0–10):
Scaled from original 0–5 rubric.

- Results include printed tables and:
1. Bar charts
2. Radar chart comparing three systems
   
## Data Source & Copyright Notice

databook.pdf used in this project is derived from a copyrighted publication:

© 2006 Elsevier Limited. All rights reserved.
First published in 2006.
ISBN-10: 0702024880
ISBN-13: 978-0702024887

This material is copyrighted and proprietary.
The PDF file is NOT redistributed as part of this repository.

The document was accessed through legitimate academic means and is used solely for educational and non-commercial purposes in the context of the UIUC CS410 course project.
Any derived data (e.g., embeddings, indexes, or preprocessed text segments) are not intended to reconstruct the original work and are used strictly for system demonstration and evaluation purposes.

All rights remain with Elsevier Limited.
For copyright and permissions information, please refer to Elsevier’s official website.
