# IMDb Review Cleaning

This repository contains a Python preprocessing pipeline for the IMDb 50K review dataset.

## What it does

- Loads the IMDb review dataset with `pandas`
- Inspects missing values and duplicate reviews
- Converts review text to lowercase
- Removes HTML tags, URLs, punctuation, numbers, and special characters via regex
- Removes English stopwords with `nltk`
- Tokenizes text
- Applies lemmatization
- Stores cleaned reviews in a new `cleaned_review` column
- Exports the cleaned dataset as CSV

## Usage

1. Install dependencies:

```bash
pip install pandas nltk requests
```

2. Run the script:

```bash
python imdb_review_cleaning.py
```

If `imdb_50k.csv` is not found in the workspace, the script downloads it automatically from a public GitHub mirror.

## Output

- `imdb_50k_cleaned.csv`

Sample columns:

- `review`
- `sentiment`
- `cleaned_review`
