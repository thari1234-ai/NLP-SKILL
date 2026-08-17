HMM POS Tagger
================

This folder contains a simple Hidden Markov Model part-of-speech tagger using the Universal Dependencies English EWT dataset.

Files
- `HMM_POS_tagger.py`: Script to download UD data, train an HMM, run Viterbi, evaluate on test set, and tag an input sentence.
- `requirements.txt`: `requests` and `scikit-learn` (optional) listed.

Usage
-----
Install dependencies (recommended inside a virtualenv):

```powershell
pip install -r postaggomg\requirements.txt
```

Run the demo (will download UD files into `postaggomg/data`):

```powershell
python postaggomg\HMM_POS_tagger.py --sentence "The student reads a book"
```

This prints token-level accuracy on the UD test set and the predicted tags for the input sentence.
