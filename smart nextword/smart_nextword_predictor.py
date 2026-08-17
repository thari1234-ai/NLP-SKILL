import re
import math
from collections import defaultdict, Counter

import numpy as np
import nltk
from nltk.tokenize import word_tokenize

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None


nltk.download("punkt", quiet=True)


SAMPLE_TEXT = """
The quick brown fox jumps over the lazy dog. The dog sleeps near the house.
The fox is quick and the dog is loyal. People love the quick fox and the lazy dog.
The model predicts the next word using simple statistics from text data.
Natural language processing helps machines understand patterns in words.
Machine learning models learn from context to suggest better next words.
"""


def clean_text(text):
    """Lowercase and keep only word-like tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_wikitext2(max_examples=50):
    """Try to load WikiText-2 if the dataset package is available; otherwise use fallback text."""
    if load_dataset is not None:
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [example["text"] for example in dataset[:max_examples]]
            if texts:
                return texts
        except Exception:
            pass

    return [SAMPLE_TEXT]


def tokenize_corpus(texts):
    tokens = []
    for text in texts:
        cleaned = clean_text(text)
        tokens.extend(word_tokenize(cleaned))
    return tokens


def build_ngram_model(tokens, n=3):
    """Create a simple n-gram model using context -> next-word counts."""
    model = defaultdict(Counter)
    context_counts = Counter()
    vocabulary = set(tokens)

    for i in range(len(tokens) - (n - 1)):
        context = tuple(tokens[i : i + n - 1])
        next_word = tokens[i + n - 1]
        model[context][next_word] += 1
        context_counts[context] += 1

    return model, context_counts, vocabulary


def predict_next_word(model, context, context_counts, vocabulary, top_k=5):
    """Predict likely next words based on the most recent context."""
    if not context:
        return []

    context = tuple(context[-2:])
    candidates = model.get(context, {})

    if not candidates:
        if context_counts:
            fallback = Counter()
            for ctx, counts in model.items():
                fallback.update(counts)
            candidates = dict(fallback)
        else:
            return []

    vocab_size = max(len(vocabulary), 1)
    scored_words = []

    total_context_count = context_counts.get(context, 0)
    if total_context_count == 0:
        total_context_count = sum(candidates.values())

    for word, count in candidates.items():
        probability = (count + 1) / (total_context_count + vocab_size)
        score = math.log(probability)
        scored_words.append((word, score))

    scored_words.sort(key=lambda item: item[1], reverse=True)
    return scored_words[:top_k]


def demo():
    texts = load_wikitext2()
    tokens = tokenize_corpus(texts)
    model, context_counts, vocabulary = build_ngram_model(tokens, n=3)

    queries = [
        ["the"],
        ["quick", "brown"],
        ["the", "fox"],
        ["natural", "language"],
        ["next", "word"],
    ]

    print("Smart Next-Word Predictor")
    print("=" * 30)
    print("Using a simple trigram model from WikiText-2 style text.\n")

    for query in queries:
        predictions = predict_next_word(model, query, context_counts, vocabulary, top_k=5)
        print(f"Context: {' '.join(query)}")
        if predictions:
            for word, score in predictions:
                print(f"  - {word:<12} score={score:.4f}")
        else:
            print("  - No prediction available")
        print()


if __name__ == "__main__":
    demo()
