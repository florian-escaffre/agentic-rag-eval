# eval/

Evaluation of the RAG system on the dataset's questions.

| File | Role |
|---|---|
| `metrics.py` | Text comparison functions. |
| `run_eval.py` | Runs the system on a sample of questions, scores it, saves the results. |
| `results/` | One JSON file per experiment. |

## Running

```bash
python eval/run_eval.py --name rag_baseline --n 50
```

| Option | Meaning | Default |
|---|---|---|
| `--name` | Experiment name (required), used as the output file name | none |
| `--n` | Number of questions | 100 |
| `--k` | Number of retrieved passages | `config.TOP_K` |
| `--seed` | Seed for the question sample | 42 |

The same seed gives the same sample, so different systems are compared on the same questions.

## Question types

The dataset mixes two kinds of questions, scored differently:

- **Yes/no:** correct if the first word of the answer matches the expected "yes" or "no".
- **Open:** four measures:
  - `exact_match`: identical to the gold answer after normalization (strictest).
  - `f1`: overlap of words with the gold answer (partial credit).
  - `answer_in_generation`: the gold answer appears in the model's answer, even if verbose.
  - `answer_in_retrieved`: the gold answer appears in the retrieved passages.

## Metrics (`metrics.py`)

- `normalize`: lowercase, remove punctuation and articles, collapse spaces.
- `exact_match`, `token_f1`, `contains`: the measures above.
- `parse_yes_no`: reads "yes" or "no" at the start of an answer, otherwise `None` (counted as wrong).

## Reading the results

If `answer_in_retrieved` is high but `answer_in_generation` is low, retrieval worked and the model used the passages badly. If `answer_in_retrieved` is already low, the problem is upstream, in retrieval.

## Output

`results/<name>.json` contains:

- `config`: models, `k`, sample size and seed,
- `summary`: aggregated metrics and mean latency,
- `rows`: question, gold answer, prediction and per-question scores.

Reading a few rows by hand is more informative than the averages.

## Limitations

- The questions are not linked to passages, so `answer_in_retrieved` is a proxy: a short answer can appear in an irrelevant passage, and a correct answer can be phrased differently from the text.
- `contains` matches substrings, not whole words (e.g. "1" is found in "1850").
- String-based metrics ignore meaning.
- Small samples are noisy.
- Mean latency includes retrieval and generation, and the first question also pays the model loading cost.