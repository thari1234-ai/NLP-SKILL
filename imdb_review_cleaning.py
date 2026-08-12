import os
import re
import sys
from pathlib import Path

import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    from nltk.corpus import wordnet
except ImportError:
    raise ImportError(
        "NLTK is required. Install it with `pip install nltk` and rerun the script."
    )

DATASET_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/imdb_50k.csv"
)
DEFAULT_INPUT = "imdb_50k.csv"
DEFAULT_OUTPUT = "imdb_50k_cleaned.csv"


def download_dataset(path: Path) -> None:
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests is required to download the dataset automatically. "
            "Install it with `pip install requests` or place the dataset file manually."
        )

    print(f"Downloading dataset from {DATASET_URL} to {path}...")
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    print(f"Downloaded dataset to {path}.")


def ensure_nltk_data() -> None:
    resources = [
        ("stopwords", "corpora/stopwords"),
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab/english"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]
    for package, resource_name in resources:
        try:
            nltk.data.find(resource_name)
        except LookupError:
            nltk.download(package)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Dataset file {path} not found.")
        download_dataset(path)

    df = pd.read_csv(path)
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(
            "Dataset must contain at least 'review' and 'sentiment' columns."
        )
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("\nDataset shape:", df.shape)
    print("\nMissing values by column:")
    print(df.isna().sum())
    duplicate_count = df.duplicated(subset=["review"]).sum()
    print(f"\nDuplicate reviews: {duplicate_count}")


def get_wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    return word_tokenize(text)


def remove_stopwords(tokens: list[str], stopword_set: set[str]) -> list[str]:
    return [token for token in tokens if token not in stopword_set]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    try:
        tagged_tokens = pos_tag(tokens)
    except LookupError:
        tagged_tokens = [(token, "N") for token in tokens]
    return [
        lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens
    ]


def preprocess_review(text: str, stopword_set: set[str]) -> str:
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens, stopword_set)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def compare_samples(df: pd.DataFrame, n: int = 10) -> None:
    print("\nSample comparison of original and cleaned reviews:")
    display_df = df[["review", "cleaned_review"]].head(n)
    print(display_df.to_string(index=False))


def main(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT) -> None:
    ensure_nltk_data()

    dataset_path = Path(input_path)
    df = load_dataset(dataset_path)
    inspect_data(df)

    stopword_set = set(stopwords.words("english"))

    df["cleaned_review"] = df["review"].apply(lambda text: preprocess_review(text, stopword_set))

    print("\nFinished cleaning reviews.")
    compare_samples(df)

    df.to_csv(output_path, index=False)
    print(f"\nExported cleaned dataset to {output_path}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    output_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    main(input_file, output_file)
