"""Academic plagiarism / similarity detector.

This script implements a lightweight plagiarism detection workflow inspired by the
PAN Plagiarism Corpus 2011. It uses pandas, numpy, and scikit-learn to compare
pairs of documents and estimate whether one document is likely plagiarized from
another by combining lexical similarity and TF-IDF cosine similarity.

Usage examples:
    python assignment.py
    python assignment.py --corpus "C:/path/to/PAN11"
    python assignment.py --text-a "some text" --text-b "similar content"
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    """Normalize a document for lexical comparison."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text.lower())
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


def jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a and not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def dice_similarity(a: str, b: str) -> float:
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a and not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    if inter == 0:
        return 0.0
    denom = len(tokens_a) + len(tokens_b)
    if denom == 0:
        return 0.0
    return (2 * inter) / denom


def char_ngram_overlap(a: str, b: str, n: int = 3) -> float:
    text_a = re.sub(r"\s+", "", clean_text(a))
    text_b = re.sub(r"\s+", "", clean_text(b))

    grams_a = {text_a[i : i + n] for i in range(max(len(text_a) - n + 1, 0))}
    grams_b = {text_b[i : i + n] for i in range(max(len(text_b) - n + 1, 0))}

    if not grams_a and not grams_b:
        return 0.0
    union = grams_a | grams_b
    if not union:
        return 0.0
    return len(grams_a & grams_b) / len(union)


def tfidf_cosine_similarity(a: str, b: str) -> float:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    vectors = vectorizer.fit_transform([a, b])
    similarity = cosine_similarity(vectors[0], vectors[1])[0, 0]
    return float(similarity)


def extract_similarity_features(text_a: str, text_b: str) -> np.ndarray:
    """Compute one feature vector for a document pair."""
    a = clean_text(text_a)
    b = clean_text(text_b)

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    common = set(tokens_a) & set(tokens_b)

    if not tokens_a and not tokens_b:
        token_overlap = 0.0
    else:
        token_overlap = len(common) / max(min(len(tokens_a), len(tokens_b)), 1)

    feature_vector = np.array(
        [
            jaccard_similarity(a, b),
            dice_similarity(a, b),
            token_overlap,
            char_ngram_overlap(a, b, n=3),
            char_ngram_overlap(a, b, n=5),
            tfidf_cosine_similarity(a, b),
        ],
        dtype=float,
    )
    return feature_vector


def build_synthetic_dataset() -> pd.DataFrame:
    """Generate realistic labeled pairs when no PAN corpus is available."""
    base_source = """The impact of climate change on global agriculture is becoming increasingly urgent.
    Farmers across the world are facing unpredictable rainfall, rising temperatures, and soil degradation.
    These conditions reduce crop yields and threaten food security in many developing regions.
    Governments and scientists emphasize the need for resilient farming systems, irrigation infrastructure,
    and sustainable agricultural practices to adapt to a changing environment."""

    paraphrased_source = """Climate change is having a major effect on agriculture worldwide.
    Unstable rainfall, higher temperatures, and land degradation make it harder for farmers to produce food.
    Many communities are experiencing lower harvests and greater risk to food supply.
    Experts recommend resilient farming methods, improved irrigation, and long-term sustainability strategies.
    These measures can help societies respond to environmental change more effectively."""

    copied_source = """The impact of climate change on global agriculture is becoming increasingly urgent.
    Farmers worldwide are facing unpredictable rainfall, rising temperatures, and soil degradation.
    These conditions reduce crop yields and threaten food security in many developing regions.
    Governments and scientists stress the value of resilient farming systems, irrigation networks,
    and sustainable agricultural practices to adapt to environmental change."""

    unrelated_text = """A new smartphone model was released with a large display, improved camera sensors,
    and longer battery life. Analysts expect strong sales in the coming quarter, while the company
    expands its retail network and invests in artificial intelligence research."""

    records = [
        {"text_a": base_source, "text_b": copied_source, "label": 1},
        {"text_a": base_source, "text_b": paraphrased_source, "label": 1},
        {"text_a": base_source, "text_b": unrelated_text, "label": 0},
        {"text_a": unrelated_text, "text_b": paraphrased_source, "label": 0},
        {"text_a": copied_source, "text_b": paraphrased_source, "label": 1},
        {"text_a": unrelated_text, "text_b": "A different article about mobile technology, software updates, and consumer demand.", "label": 0},
    ]
    return pd.DataFrame(records)


def discover_pan_files(corpus_dir: str | None) -> List[Tuple[str, str]]:
    """Attempt to read a PAN 2011-style corpus when it exists on disk."""
    if not corpus_dir:
        return []

    root = Path(corpus_dir)
    if not root.exists():
        return []

    suspicious_files = []
    for candidate in [root / "suspicious-document", root / "suspicious", root / "suspicious-documents"]:
        if candidate.exists():
            suspicious_files.extend(list(candidate.rglob("*.txt")))

    source_files = []
    for candidate in [root / "source-document", root / "source", root / "source-documents"]:
        if candidate.exists():
            source_files.extend(list(candidate.rglob("*.txt")))

    if not suspicious_files or not source_files:
        return []

    source_map: dict[str, str] = {}
    for path in source_files:
        source_map[path.stem] = path.read_text(encoding="utf-8", errors="ignore")

    pairs: List[Tuple[str, str]] = []
    for file_path in suspicious_files:
        key = file_path.stem
        base_key = key.split("_", 1)[0]
        source_text = source_map.get(key) or source_map.get(base_key)
        if source_text is None:
            continue
        pairs.append((file_path.read_text(encoding="utf-8", errors="ignore"), source_text))
    return pairs


def load_dataset(corpus_dir: str | None) -> pd.DataFrame:
    """Load labeled examples from the PAN corpus or fall back to synthetic data."""
    pairs = discover_pan_files(corpus_dir)
    if pairs:
        rows = []
        for suspicious_text, source_text in pairs:
            rows.append({"text_a": suspicious_text, "text_b": source_text, "label": 1})
        return pd.DataFrame(rows)

    return build_synthetic_dataset()


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    for _, row in df.iterrows():
        features.append(extract_similarity_features(row["text_a"], row["text_b"]))
        labels.append(int(row["label"]))

    return np.vstack(features), np.array(labels, dtype=int)


def train_plagiarism_model(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["Not plagiarized", "Plagiarized"])

    print("\nModel accuracy:", round(accuracy, 4))
    print("\nClassification report:\n", report)

    return model


def predict_single_pair(model, text_a: str, text_b: str) -> float:
    feature_row = extract_similarity_features(text_a, text_b).reshape(1, -1)
    probability = model.predict_proba(feature_row)[0, 1]
    return float(probability)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Academic similarity/plagiarism detector")
    parser.add_argument("--corpus", type=str, default=None, help="Path to a PAN 2011 corpus directory")
    parser.add_argument("--text-a", type=str, default=None, help="First text to compare")
    parser.add_argument("--text-b", type=str, default=None, help="Second text to compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading dataset...")
    df = load_dataset(args.corpus)
    X, y = build_feature_matrix(df)

    print(f"Prepared {len(df)} document pairs.")
    model = train_plagiarism_model(X, y)

    if args.text_a and args.text_b:
        score = predict_single_pair(model, args.text_a, args.text_b)
        print(f"\nPlagiarism probability score: {score:.4f}")
        print("Interpretation: Higher score means stronger evidence of plagiarism.")

    print("\nExample usage:")
    print("  python assignment.py --text-a \"your text A\" --text-b \"your text B\"")
    print("  python assignment.py --corpus \"path/to/PAN11\"")


if __name__ == "__main__":
    main()
