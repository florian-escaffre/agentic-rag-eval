import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama

import config
from src.retriever import retrieve

PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
Be brief. For yes/no questions, answer "yes" or "no".
If the context does not contain the answer, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""


def answer(question: str, k: int = config.TOP_K) -> dict:
    """Simple RAG: retrieve passages, then generate an answer from them."""
    documents = retrieve(question, k=k)
    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # temperature=0 for reproducible answers (important for evaluation)
    llm = ChatOllama(model=config.LLM_MODEL, temperature=0)
    response = llm.invoke(prompt)

    # Return documents too: needed later to evaluate retrieval
    return {"question": question, "answer": response.content, "documents": documents}


if __name__ == "__main__":
    for q in [
        "Was Abraham Lincoln the sixteenth President of the United States?",
        "What is the capital of Uruguay?",
    ]:
        result = answer(q)
        print("Q:", result["question"])
        print("A:", result["answer"])
        print("Top passage:", result["documents"][0].page_content[:150], "\n")