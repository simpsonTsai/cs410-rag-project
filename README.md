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
├── pic/
│   ├──  allnumber.png         # evaluation result
│   ├──  correctness.png
│   ├──  hallucination.png
│   ├──  evidencyrelevancy.png
│   ├── radarchart.png 
├── docs/
│   └── usage.md           # extended documentation (optional)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🐍 Python Version

This project is tested and supported on **Python 3.10**.

Please use Python **3.10.x** to ensure smooth installation of all dependencies
(e.g., `faiss-cpu` and `sentence-transformers`) without requiring system-level
build tools such as `cmake`.

## Quick Test Mode (For TA / CPU-Based Grading)

This project includes a built-in **TA quick-test mode** to ensure smooth
and reproducible grading on CPU-only machines.

### Purpose
The goal of this mode is to allow graders to quickly verify the **complete
RAG pipeline** (chunking → embedding → retrieval → generation → evaluation)
without requiring GPU resources or long execution time.

### Behavior in TA Mode
When TA mode is enabled (default):
- Only the first **10 PDF pages** are loaded
- At most **50 text chunks** are generated
- At most **50 chunks** are embedded for dense retrieval
- The system logic and data flow remain unchanged

On a CPU-only machine, the full pipeline completes within **1–3 minutes**.

### Full Experiment Results
All evaluation figures, analyses, and reported results in this project were
generated using the **full dataset and complete pipeline**.
TA mode is provided strictly for grading and verification convenience and
does not alter the system design or conclusions.

### How to Switch Modes
TA mode is enabled by default.
To run the full experiment, set the following flag to `False`:

# src/config.py
TA_MODE = False


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/simpsonTsai/cs410-rag-project.git
```
```bash
cd cs410-rag-project
```


### 2. Create a virtual environment
```bash
python3 -m venv venv
```
#### Mac/Linux
```bash
source venv/bin/activate    
```
#### Windows
```bash
venv\Scripts\activate         
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## API Keys Setup

### 1. Create your .env file:
```bash
cp .env.example .env
```
### 2. Open .env and add your key:
```bash
nano .env
```
GROQ_API_KEY=your_key_here

After put in your GROQ API KEY:
1. Ctrl + 0 (store)
2. Enter
3. Ctrl + X (exit)

## Required Data
Place the veterinary reference book (PDF) inside the data/ folder:
data/databook.pdf

An appropriate reference of databook.pdf is shown below.

## How to Run the System
To run the entire system — baseline RAG, improved RAG, GPT-only baseline, evaluation metrics, and visualizations — run:
```bash
python src/run.py
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

This project makes use of information derived from a copyrighted academic publication for educational and non-commercial purposes only, as part of a course project for UIUC CS410 (Text Information Systems).

The original source is:

*Rand, J. M. (Ed.). **Problem-Based Feline Medicine.***
© 2006 Elsevier Limited. All rights reserved.  
ISBN-10: 0702024880  
ISBN-13: 978-0702024887


The original PDF document (databook.pdf) is copyrighted and proprietary and is NOT redistributed as part of this repository.

### Use of Derived Data
This project performs preprocessing on the source material, including but not limited to:
- text extraction
- chunking / segmentation
- embedding generation
- index construction for information retrieval
All derived artifacts (e.g., text chunks, embeddings, indexes) are used solely to demonstrate system design and retrieval behavior for the course assignment.

These derived data are non-reconstructive and cannot be used to recover the original copyrighted work.
No attempt is made to reproduce, redistribute, or substitute the original publication.

All intellectual property rights remain with Elsevier Limited.
For official copyright and permissions information, please refer to Elsevier’s website.
## Important Notes

This repository does not contain the original databook.pdf.
Access to the original material was obtained through legitimate academic means.
This project is intended strictly for educational demonstration, not for clinical, commercial, or redistribution purposes.
