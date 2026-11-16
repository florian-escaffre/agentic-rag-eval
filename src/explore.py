from datasets import load_dataset

corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")

print(corpus)
print(qa)

# First split of each config (split names not assumed)
docs = corpus[list(corpus.keys())[0]]
questions = qa[list(qa.keys())[0]]

print("\nCorpus columns:", docs.column_names, "| size:", len(docs))
print("QA columns:", questions.column_names, "| size:", len(questions))

print("\n--- Example document ---")
print(docs[0])
print("\n--- Example question ---")
print(questions[0])

# "passage" is the dataset's own column name
if "passage" in docs.column_names:
    lengths = [len(doc["passage"]) for doc in docs]
    print(
        f"\nDocument length (characters): min={min(lengths)}, "
        f"mean={sum(lengths) // len(lengths)}, max={max(lengths)}"
    )