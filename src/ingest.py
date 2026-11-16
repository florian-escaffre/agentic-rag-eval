import sys
from pathlib import Path

# Make the project root importable so `import config` works
# when the script is run as `python src/ingest.py`
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

import config


def load_documents() -> list[Document]:
    """Download the corpus and convert each passage into a LangChain Document."""
    # The "text-corpus" config has a single split named "passages"
    corpus = load_dataset(config.DATASET_NAME, "text-corpus")["passages"]

    documents = []
    for row in corpus:
        text = row["passage"].strip()

        # Skip near-empty passages: they carry no useful information
        if len(text) < config.MIN_PASSAGE_CHARS:
            continue

        # Keep the original passage id as metadata to trace the source later
        documents.append(
            Document(page_content=text, metadata={"passage_id": row["id"]})
        )

    print(f"Kept {len(documents)} / {len(corpus)} passages")
    return documents


def main() -> None:
    documents = load_documents()

    # Embedding model: turns a text into a vector (run locally by Ollama).
    # The same model must be used later to embed the user questions.
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

    # Open (or create) the persistent vector store in the db/ folder
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.DB_DIR),
    )

    # Index documents in batches to limit memory use and show progress
    for start in tqdm(range(0, len(documents), config.BATCH_SIZE)):
        batch = documents[start : start + config.BATCH_SIZE]

        # Stable IDs: re-running the script overwrites instead of duplicating
        ids = [str(doc.metadata["passage_id"]) for doc in batch]

        # Embeds each document and stores vector + text + metadata
        vector_store.add_documents(batch, ids=ids)

    # Sanity check: number of documents actually stored
    print("Done. Documents in store:", vector_store._collection.count())


if __name__ == "__main__":
    main()