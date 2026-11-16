import re
import string
from collections import Counter

# Set of punctuation characters, built once for fast lookups
PUNCTUATION = set(string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse spaces.

    Makes comparisons insensitive to case, punctuation and articles,
    e.g. "The Eiffel Tower." -> "eiffel tower".
    """
    text = text.lower()
    # Remove every punctuation character
    text = "".join(ch for ch in text if ch not in PUNCTUATION)
    # Remove articles as whole words (\b = word boundary)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # split() + join() collapses runs of whitespace into single spaces
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> bool:
    """True only if both texts are identical after normalization (strictest metric)."""
    return normalize(prediction) == normalize(gold)


def token_f1(prediction: str, gold: str) -> float:
    """F1 over shared words between prediction and gold answer.

    More tolerant than exact match: partial overlap earns partial credit.
    """
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()

    # Counter intersection keeps the minimum count of each shared word,
    # so repeated words are counted correctly
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())

    # No shared word: score is 0 (also avoids dividing by zero below)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)  # share of predicted words that are correct
    recall = overlap / len(gold_tokens)     # share of expected words that were found
    return 2 * precision * recall / (precision + recall)  # harmonic mean


def contains(text: str, gold: str) -> bool:
    """True if the normalized gold answer appears inside the normalized text.

    Used on the generated answer and on the retrieved passages.
    Note: substring match, not whole-word match (e.g. "1" is found in "1850").
    """
    gold_norm = normalize(gold)
    # bool(gold_norm) guards against an empty gold answer matching everything
    return bool(gold_norm) and gold_norm in normalize(text)


def parse_yes_no(text: str) -> str | None:
    """Return 'yes' or 'no' if the answer starts with one of them, else None.

    The `str | None` return type means the function may return nothing
    (Python 3.10+ syntax). None counts as an incorrect answer in evaluation.
    """
    words = normalize(text).split()
    return words[0] if words and words[0] in {"yes", "no"} else None