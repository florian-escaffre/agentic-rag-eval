import argparse
import json
import random
import sys
import time
from pathlib import Path

# Make the project root importable (for `config`, `eval`, `src`)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from tqdm import tqdm

import config
from eval.metrics import contains, exact_match, normalize, parse_yes_no, token_f1
from src.rag_chain import answer

# Folder where each experiment is saved as a JSON file
RESULTS_DIR = config.ROOT / "eval" / "results"


def mean(values: list[float]) -> float:
    """Average of a list; NaN if empty (e.g. no open question in the sample).

    Booleans work too: True counts as 1 and False as 0, so the mean
    of a list of booleans is a proportion (accuracy).
    """
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    # --- Command-line arguments: one script, many experiments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="experiment name")
    parser.add_argument("--n", type=int, default=100, help="number of questions")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Select the questions ---
    questions = list(load_dataset(config.DATASET_NAME, "question-answer")["test"])
    # Fixed seed: same random sample at every run, so experiments are comparable
    random.Random(args.seed).shuffle(questions)
    questions = questions[: args.n]

    # --- Run the RAG system on each question ---
    rows = []
    for item in tqdm(questions):
        gold = item["answer"]

        # Measure end-to-end time (retrieval + generation)
        start = time.perf_counter()
        result = answer(item["question"], k=args.k)
        latency = time.perf_counter() - start

        # All retrieved passages as one text, to check if they contain the answer
        retrieved_text = " ".join(doc.page_content for doc in result["documents"])

        row = {
            "question": item["question"],
            "gold": gold,
            "prediction": result["answer"],
            "latency_s": round(latency, 2),
            # Yes/no questions are scored differently from open questions
            "is_yes_no": normalize(gold) in {"yes", "no"},
        }
        if row["is_yes_no"]:
            # Correct if the model's first word matches the expected yes/no
            row["correct"] = parse_yes_no(result["answer"]) == normalize(gold)
        else:
            row["exact_match"] = exact_match(result["answer"], gold)
            row["f1"] = token_f1(result["answer"], gold)
            # Did the model mention the right answer, even if verbose?
            row["answer_in_generation"] = contains(result["answer"], gold)
            # Proxy for retrieval quality: gold answer appears in retrieved passages
            row["answer_in_retrieved"] = contains(retrieved_text, gold)
        rows.append(row)

    # --- Aggregate the results ---
    yes_no = [r for r in rows if r["is_yes_no"]]
    open_q = [r for r in rows if not r["is_yes_no"]]
    summary = {
        "n_total": len(rows),
        "n_yes_no": len(yes_no),
        "n_open": len(open_q),
        "yes_no_accuracy": mean([r["correct"] for r in yes_no]),
        "open_exact_match": mean([r["exact_match"] for r in open_q]),
        "open_f1": mean([r["f1"] for r in open_q]),
        "open_answer_in_generation": mean([r["answer_in_generation"] for r in open_q]),
        "open_answer_in_retrieved": mean([r["answer_in_retrieved"] for r in open_q]),
        "mean_latency_s": mean([r["latency_s"] for r in rows]),
    }

    # --- Save everything: settings, summary and per-question details ---
    output = {
        "config": {
            "name": args.name,
            "llm": config.LLM_MODEL,
            "embedding": config.EMBEDDING_MODEL,
            "k": args.k,
            "n": args.n,
            "seed": args.seed,
        },
        "summary": summary,
        "rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{args.name}.json"
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2))
    print("Saved to", path)


if __name__ == "__main__":
    main()