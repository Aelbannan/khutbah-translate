"""Shared fuzzy-matching machinery for finding quoted texts (Quran, hadith) in STT output.

Two stages:
  1. `NgramIndex.candidates`: character-4-gram TF-IDF retrieval over normalised docs.
  2. `local_align`: word-level Smith-Waterman between the query and a candidate,
     using fuzzy word similarity so STT spelling noise still aligns.
"""

from pathlib import Path

import numpy as np

NGRAM = 4
MATCH_SIM = 0.72  # word similarity counted as "the same word" (STT spelling noise)
GAP = -0.6
MISMATCH = -0.8


def grams(text: str) -> set[str]:
    t = f" {text} "
    return {t[i : i + NGRAM] for i in range(len(t) - NGRAM + 1)}


class NgramIndex:
    """Inverted index from char n-grams to doc ids, stored as CSR numpy arrays."""

    def __init__(self, vocab: dict[str, int], offsets: np.ndarray, postings: np.ndarray, n_docs: int):
        self.vocab, self.offsets, self.postings = vocab, offsets, postings
        df = np.diff(offsets).astype(np.float64)
        self.idf = np.log(n_docs / np.maximum(df, 1))
        self.n_docs = n_docs

    @classmethod
    def build(cls, docs: list[str]) -> "NgramIndex":
        vocab: dict[str, int] = {}
        gram_ids, doc_ids = [], []
        for d, text in enumerate(docs):
            ids = [vocab.setdefault(g, len(vocab)) for g in grams(text)]
            gram_ids.extend(ids)
            doc_ids.extend([d] * len(ids))
        g = np.asarray(gram_ids, dtype=np.int32)
        order = np.argsort(g, kind="stable")
        postings = np.asarray(doc_ids, dtype=np.int32)[order]
        offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.add.at(offsets, g + 1, 1)
        return cls(vocab, np.cumsum(offsets), postings, len(docs))

    @classmethod
    def cached(cls, docs: list[str], path: Path) -> "NgramIndex":
        """Load from `path` (.npz) if present and current, else build and save."""
        if path.exists():
            z = np.load(path, allow_pickle=False)
            if int(z["n_docs"]) == len(docs):
                vocab = {g: i for i, g in enumerate(z["vocab"].tolist())}
                return cls(vocab, z["offsets"], z["postings"], len(docs))
        idx = cls.build(docs)
        words = np.array(sorted(idx.vocab, key=idx.vocab.get))
        np.savez(path, vocab=words, offsets=idx.offsets, postings=idx.postings, n_docs=len(docs))
        return idx

    def candidates(self, query: str, k: int) -> list[int]:
        ids = [i for g in grams(query) if (i := self.vocab.get(g)) is not None]
        if not ids:
            return []
        scores = np.zeros(self.n_docs)
        for i in ids:
            np.add.at(scores, self.postings[self.offsets[i] : self.offsets[i + 1]], self.idf[i])
        k = min(k, int((scores > 0).sum()))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top])].tolist()


def _sw_best(sub: np.ndarray) -> tuple[float, int, int]:
    """Best local-alignment score and its end (i, j), linear gaps, vectorised per row."""
    n, m = sub.shape
    prev = np.zeros(m + 1)
    k = np.arange(m)
    best = (0.0, 0, 0)
    for i in range(n):
        tmp = np.maximum(np.maximum(prev[:-1] + sub[i], prev[1:] + GAP), 0.0)
        # Left gaps: row[j] = max_{k<=j} tmp[k] + GAP*(j-k)
        row = np.maximum.accumulate(tmp - GAP * k) + GAP * k
        j = int(row.argmax())
        if row[j] > best[0]:
            best = (float(row[j]), i + 1, j + 1)
        prev = np.concatenate(([0.0], row))
    return best


def local_align(sim: np.ndarray) -> tuple[float, int, int, int, int, int] | None:
    """Smith-Waterman over a query x reference word-similarity matrix (values 0-1).

    Returns (score, q_start, q_end, r_start, r_end, matched_words), ends exclusive.
    """
    sub = np.where(sim >= MATCH_SIM, sim, MISMATCH)
    score, qe, re_ = _sw_best(sub)
    if score <= 0:
        return None
    # The start is the end of the best alignment on the reversed prefix.
    _, qi, ri = _sw_best(sub[:qe, :re_][::-1, ::-1])
    qs, rs = qe - qi, re_ - ri
    matched = int((sim[qs:qe, rs:re_].max(axis=1) >= MATCH_SIM).sum())
    return score, qs, qe, rs, re_, matched


def crop(sim: np.ndarray, pad: int = 8) -> tuple[np.ndarray, int]:
    """Restrict a long reference to the window with the most word hits.

    Returns (cropped sim, column offset). Keeps alignment cost bounded for long
    documents (hadith with long isnads) without changing the result in practice.
    """
    n, m = sim.shape
    width = 2 * n + 2 * pad
    if m <= width:
        return sim, 0
    # Score each window by how many *distinct query words* hit inside it, so
    # frequent words (الله, ان) scattered over a long text don't win.
    hits = (sim >= MATCH_SIM).astype(np.int32)
    csum = np.concatenate([np.zeros((n, 1), np.int32), hits.cumsum(axis=1)], axis=1)
    per_row = csum[:, width:] - csum[:, : m - width + 1]  # (n, m - width + 1)
    window = (per_row > 0).sum(axis=0) * 1000 + per_row.sum(axis=0)  # ties: more hits
    start = int(window.argmax())
    return sim[:, start : start + width], start
