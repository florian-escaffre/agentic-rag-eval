from pathlib import Path

# Project root = folder containing this file
ROOT = Path(__file__).parent

# Where Chroma persists the vector index (git-ignored)
DB_DIR = ROOT / "db"

# Hugging Face dataset and name of the Chroma collection
DATASET_NAME = "rag-datasets/rag-mini-wikipedia"
COLLECTION_NAME = "wiki_passages"

# Ollama models: embeddings for retrieval, LLM for answer generation
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "ministral-3:3b"

# Ingestion settings
MIN_PASSAGE_CHARS = 20  # arbitrary threshold to drop near-empty passages
BATCH_SIZE = 128        # documents embedded per call

# Retrieval setting: number of passages given to the LLM per question
TOP_K = 4