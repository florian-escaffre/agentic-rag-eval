# src/

Pipeline code: indexing, retrieval and answer generation. Shared settings are in `../config.py`.

| File | Role |
|---|---|
| `explore.py` | Loads the dataset and prints its structure, sizes and one example of each part. Run it first to check the format. |
| `ingest.py` | Builds the vector index (see below). |
| `retriever.py` | Opens the index and returns the passages closest to a question. |
| `rag_chain.py` | Simple RAG: retrieval, prompt, generation. |
| `agent_graph.py` | Reserved for the LangGraph agent (not implemented yet). |

## `ingest.py`

1. Downloads the 3,200 passages from Hugging Face.
2. Drops passages shorter than `MIN_PASSAGE_CHARS` (near-empty, nothing to retrieve).
3. Wraps each passage in a LangChain `Document`, keeping its original id as metadata.
4. Embeds the documents with `nomic-embed-text` (via Ollama) and stores them in Chroma under `db/`, in batches.

Each document is stored under its passage id, so re-running the script overwrites instead of duplicating. Passages are already short, so they are not split further: one passage is one chunk.

## `retriever.py`

- `get_vector_store()` opens the existing Chroma index. The result is cached (`lru_cache`), so the database is opened once per process, not once per question. It must use the same embedding model as `ingest.py`.
- `retrieve(question, k)` embeds the question and returns the `k` most similar passages.

## `rag_chain.py`

`answer(question, k)`:

1. retrieves the `k` passages,
2. builds a prompt asking to answer only from the context, briefly, with "yes"/"no" for closed questions and "I don't know" if the context is insufficient,
3. calls the local LLM with `temperature=0` so runs are reproducible,
4. returns the question, the answer **and the retrieved documents**, so retrieval can be evaluated separately.

Running the file directly asks two example questions.