# rag-agents

A personal learning project to get hands-on with **Retrieval-Augmented Generation (RAG)**, **agentic AI** and how to **evaluate** them, using **LangChain** and **LangGraph** with small local models served by **Ollama**.

This is not a production system. The goal is to understand each building block by implementing it, measuring it, and comparing a simple RAG pipeline with an agentic version of it.

## Goals

- Understand the RAG principle: embeddings, vector search, prompting with retrieved context.
- Learn how to evaluate a RAG system, separating retrieval errors from generation errors.
- Use LangChain for the building blocks and LangGraph to orchestrate an agentic workflow.
- Run everything locally on a small GPU with small models (about 3B parameters).
- Measure whether the agentic version actually improves on the simple pipeline, rather than assume it does.

## How it works

```
INDEXING (once)                     QUERYING (each question)
src/ingest.py                       src/rag_chain.py -> src/retriever.py

passages -> embeddings -> Chroma    question -> embedding -> top-k passages -> LLM -> answer
```

1. **Indexing:** every passage of the corpus is turned into a vector by an embedding model and stored in a Chroma vector database.
2. **Retrieval:** the question is embedded with the same model, and the k closest passages are fetched.
3. **Generation:** the passages and the question are put in a prompt, and a local LLM answers using only that context.
4. **Evaluation:** a fixed sample of questions is run through the system, and answers and retrieved passages are scored.

## Dataset

[`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) from Hugging Face:

- `text-corpus`: 3,200 short Wikipedia passages (about 390 characters on average).
- `question-answer`: 918 questions with short answers, many of them yes/no.

The questions are not linked to specific passages, so retrieval quality can only be estimated with a proxy (see [`eval/`](eval/README.md)). Check the dataset card for its license.

## Models

Served locally by Ollama and set in `config.py`:

| Role | Model |
|---|---|
| Embeddings | `nomic-embed-text` |
| Generation | a ~3B model (e.g. `llama3.2:3b` or `ministral-3:3b`) |

## Project structure

```
rag-agents/
├── config.py          # models, paths and parameters in one place
├── requirements.txt
├── src/               # pipeline code (see src/README.md)
├── eval/              # evaluation code and results (see eval/README.md)
└── db/                # Chroma index, generated locally (see db/README.md)
```

## Setup

Requirements: Python 3.10+, [Ollama](https://ollama.com) running locally.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull nomic-embed-text
ollama pull llama3.2:3b        # or the model set in config.py
```

## Usage

```bash
# 1. Inspect the dataset
python src/explore.py

# 2. Build the vector index (about a minute for 3,081 passages)
python src/ingest.py

# 3. Ask a few test questions
python src/rag_chain.py

# 4. Evaluate on a sample of questions
python eval/run_eval.py --name rag_baseline --n 50
```

## Evaluation approach

Questions are scored in two groups:

- **Yes/no questions:** accuracy on the first word of the answer.
- **Open questions:** exact match, token-level F1, whether the gold answer appears in the generated answer, and whether it appears in the retrieved passages.

Comparing the last two shows where errors come from: bad retrieval or bad use of good passages. Details and limits are in [`eval/README.md`](eval/README.md).

## Roadmap

- [x] Dataset exploration
- [x] Indexing with Chroma and Ollama embeddings
- [x] Simple RAG pipeline (LangChain building blocks)
- [x] Evaluation script
- [ ] Baseline results
- [ ] LangGraph agent: query rewriting, relevance grading, retry loop
- [ ] Comparison simple RAG vs agent (quality, latency, number of LLM calls)
- [ ] Possibly a multi-agent version, only if the single agent shows a clear limit

## Results

To be filled in once the evaluation has been run.

| System | Yes/no acc. | Open F1 | Answer in retrieved | Mean latency |
|---|---|---|---|---|
| Simple RAG | TBD | TBD | TBD | TBD |
| LangGraph agent | TBD | TBD | TBD | TBD |

## Limitations

- Small models (about 3B) follow instructions less reliably than large ones, especially in agent loops.
- Retrieval quality is estimated through a proxy, not measured against labelled passages.
- String-based metrics measure surface similarity, not meaning: a correct but rephrased answer can be scored low.
- Small evaluation samples give noisy numbers; avoid reading much into small differences.

## Tech stack

Python, LangChain, LangGraph, Chroma, Ollama, Hugging Face `datasets`.