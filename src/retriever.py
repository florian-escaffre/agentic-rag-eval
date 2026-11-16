import sys
from functools import lru_cache
from pathlib import Path

# Make the project root importable so `import config` works
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import config


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Open the existing Chroma index built by ingest.py.

    lru_cache stores the returned object after the first call, so the
    connection to Chroma is opened only once per Python process instead
    of once per question.
    """
    # Same embedding model as in ingest.py: vectors from different
    # models are not comparable
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.DB_DIR),
    )


def retrieve(question: str, k: int = config.TOP_K) -> list:
    """Return the k passages closest to the question (as Documents)."""
    vector_store = get_vector_store()  # cached after the first call
    return vector_store.similarity_search(question, k=k)