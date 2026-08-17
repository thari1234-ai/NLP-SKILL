import os
import math
import argparse
from collections import defaultdict, Counter

try:
    import requests
except Exception:
    requests = None


UD_BASE = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/"
FILES = {
    'train': 'en_ewt-ud-train.conllu',
    'test': 'en_ewt-ud-test.conllu'
}


def download_file(url, dest):
    if os.path.exists(dest):
        return dest
    if requests is None:
        raise RuntimeError('requests not installed; cannot download dataset')
    r = requests.get(url)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        f.write(r.content)
    return dest


def save_split_from_hf(dataset, split_name, dest_path):
    """Save a dataset split (HF datasets) to a CoNLL-U style file with tokens and UPOS."""
    with open(dest_path, 'w', encoding='utf-8') as f:
        for ex in dataset[split_name]:
            tokens = ex.get('tokens') or ex.get('words') or ex.get('sentence')
            upos = ex.get('upos')
            if not tokens or not upos:
                continue
            for i, (tok, tag) in enumerate(zip(tokens, upos), start=1):
                f.write(f"{i}\t{tok}\t_\t{tag}\t_\t_\t0\t_\t_\t_\n")
            f.write("\n")


def load_conllu(path):
    sentences = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding='utf-8') as f:
        sent = []
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sentences.append(sent)
                    sent = []
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            idx = parts[0]
            if '-' in idx or '.' in idx:
                continue
            word = parts[1]
            upos = parts[3]
            sent.append((word, upos))
        if sent:
            sentences.append(sent)
    return sentences


class HMMTagger:
    def __init__(self):
        self.tags = set()
        self.vocab = set()
        self.trans_counts = defaultdict(Counter)  # prev_tag -> next_tag -> count
        self.emit_counts = defaultdict(Counter)  # tag -> word -> count
        self.tag_counts = Counter()
        self.transition_probs = defaultdict(dict)
        self.emission_probs = defaultdict(dict)

    def train(self, sentences):
        START = '<s>'
        for sent in sentences:
            prev = START
            self.tag_counts[START] += 1
            for word, tag in sent:
                self.vocab.add(word)
                self.tags.add(tag)
                self.trans_counts[prev][tag] += 1
                self.emit_counts[tag][word] += 1
                self.tag_counts[tag] += 1
                prev = tag
            # end sentence -> add transition to STOP if desired (not required for Viterbi here)

        self.tags = sorted(self.tags)
        V = len(self.vocab)
        N = len(self.tags) + 1  # include START

        # compute transition probabilities with add-one smoothing
        for prev_tag, ctr in self.trans_counts.items():
            total = sum(ctr.values())
            for tag in self.tags:
                self.transition_probs[prev_tag][tag] = math.log((ctr.get(tag, 0) + 1) / (total + N))

        # emissions with add-one smoothing
        for tag in self.tags:
            total = self.tag_counts[tag]
            for word in self.vocab:
                self.emission_probs[tag][word] = math.log((self.emit_counts[tag].get(word, 0) + 1) / (total + V))

    def viterbi(self, words):
        START = '<s>'
        V = len(self.vocab)
        tags = self.tags

        # Initialization
        dp = []  # list of dicts tag -> (logprob, backpointer)
        first = {}
        for t in tags:
            trans = self.transition_probs.get(START, {}).get(t, math.log(1e-12))
            emit = self.emission_probs[t].get(words[0], None)
            if emit is None:
                # unseen word -> use smoothed emission probability
                emit = math.log(1 / (self.tag_counts[t] + V))
            first[t] = (trans + emit, START)
        dp.append(first)

        # Recursion
        for i in range(1, len(words)):
            cur = {}
            w = words[i]
            for t in tags:
                emit = self.emission_probs[t].get(w, None)
                if emit is None:
                    emit = math.log(1 / (self.tag_counts[t] + V))
                best_prob = None
                best_prev = None
                for prev_t, (prob_prev, _) in dp[i-1].items():
                    trans = self.transition_probs.get(prev_t, {}).get(t, math.log(1e-12))
                    p = prob_prev + trans + emit
                    if best_prob is None or p > best_prob:
                        best_prob = p
                        best_prev = prev_t
                cur[t] = (best_prob, best_prev)
            dp.append(cur)

        # Termination: pick best final tag
        last = dp[-1]
        best_tag = max(last.items(), key=lambda x: x[1][0])[0]
        tags_seq = [best_tag]
        for i in range(len(words)-1, 0, -1):
            best_tag = dp[i][best_tag][1]
            tags_seq.append(best_tag)
        tags_seq.reverse()
        return tags_seq


def evaluate(tagger, sentences):
    y_true = []
    y_pred = []
    for sent in sentences:
        words = [w for w, t in sent]
        true_tags = [t for w, t in sent]
        pred_tags = tagger.viterbi(words)
        y_true.extend(true_tags)
        y_pred.extend(pred_tags)

    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    acc = correct / len(y_true)

    # try to use sklearn classification_report if available
    report = None
    try:
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, zero_division=0)
    except Exception:
        # simple per-tag metrics fallback
        tags = sorted(set(y_true) | set(y_pred))
        per_tag = {}
        for tag in tags:
            tp = sum(1 for a, b in zip(y_true, y_pred) if a == tag and b == tag)
            fp = sum(1 for a, b in zip(y_true, y_pred) if a != tag and b == tag)
            fn = sum(1 for a, b in zip(y_true, y_pred) if a == tag and b != tag)
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            per_tag[tag] = (prec, rec, f1)
        lines = []
        for tag, (p, r, f) in per_tag.items():
            lines.append(f"{tag}\tP={p:.3f}\tR={r:.3f}\tF1={f:.3f}")
        report = '\n'.join(lines)

    return acc, report


def main(args):
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    # download files if possible; if rate-limited, try huggingface datasets as fallback
    paths = {}
    need_hf_fallback = False
    for split, fname in FILES.items():
        url = UD_BASE + fname
        dest = os.path.join(data_dir, fname)
        if not os.path.exists(dest):
            if requests is None:
                need_hf_fallback = True
                break
            try:
                print('Downloading', fname)
                download_file(url, dest)
            except Exception as e:
                print('Download failed:', e)
                need_hf_fallback = True
                break
        paths[split] = dest

    if need_hf_fallback:
        print('Attempting to load dataset via the Hugging Face `datasets` library as a fallback...')
        try:
            from datasets import load_dataset
        except Exception as e:
            print('datasets library not available. Install with `pip install datasets` or download files manually.')
            return
        ds = load_dataset('universal_dependencies', 'en_ewt')
        for split, fname in FILES.items():
            dest = os.path.join(data_dir, fname)
            split_name = 'train' if split == 'train' else ('validation' if split == 'dev' else 'test')
            # some HF configs use 'validation' instead of 'dev'
            if split_name not in ds:
                # try common alternatives
                split_name = split
            print('Saving', split_name, 'to', dest)
            save_split_from_hf(ds, split_name, dest)
            paths[split] = dest

    print('Loading training data...')
    train_sents = load_conllu(paths['train'])
    print('Loading test data...')
    test_sents = load_conllu(paths['test'])

    tagger = HMMTagger()
    print('Training HMM...')
    tagger.train(train_sents)

    print('Evaluating on test set...')
    acc, report = evaluate(tagger, test_sents)
    print(f'Token-level accuracy: {acc:.4f}')
    print('Classification report:\n')
    print(report)

    if args.sentence:
        words = args.sentence.strip().split()
        preds = tagger.viterbi(words)
        mapping = ' '.join([f"{w}→{t}" for w, t in zip(words, preds)])
        print('\nInput sentence tagging:')
        print(mapping)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), 'data'))
    p.add_argument('--sentence', default='The student reads a book')
    args = p.parse_args()
    main(args)
