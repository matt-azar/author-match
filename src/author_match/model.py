"""CPU baselines for open-set, text-only authorship verification."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .text import mask_hashtags, words

METHODS = ("word", "char", "style")
_BATCH = 1024

_FUNCTION_WORDS = tuple(
    "a an the and but or so if because as when that this then than just also "
    "to for from in on with of at by it its is was are do have can would "
    "i me my we you your they he she who what how really very too kinda "
    "tbh idk imo lol lmao yeah like".split()
)
_PUNCT = ".,!?;:'\"’()-…/#@*"
_REPEATED_LETTER = re.compile(r"([^\W\d_])\1{2,}", re.I)
_REPEATED_PUNCT = re.compile(r"([!?.])\1+")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U0001F1E6-\U0001F1FF\u2600-\u27BF]")
_EMOTICON = re.compile(r"(?::|;|=|8)-?[)(DPp/]|<3")
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")


def _style(text: str) -> list[float]:
    tokens = words(text)
    counts = Counter(token.casefold().replace("’", "'") for token in tokens)
    length = len(tokens)
    # Pseudocount denominators keep one exclamation mark in a four-word
    # reply from having the same evidential weight as a recurring habit.
    w = length + 5
    c = len(text) + 20
    letters = sum(map(str.isalpha, text))
    sentences = len(_SENTENCE_END.findall(text))
    features = [
        np.log1p(length), len(counts) / w, sum(map(len, tokens)) / w,
        sum(ch.isupper() for ch in text) / (letters + 5),
        sum(t.isupper() and len(t) > 1 for t in tokens) / w,
        sum(t.istitle() for t in tokens) / w,
        sum("'" in t or "’" in t for t in tokens) / w,
        sum(ch.isdigit() for ch in text) / c, text.count("\n") / w,
        text.count("  ") / w, sentences / w,
        len(_REPEATED_LETTER.findall(text)) / w,
        len(_REPEATED_PUNCT.findall(text)) / w,
        len(_EMOJI.findall(text)) / w, len(_EMOTICON.findall(text)) / w,
        float(text.rstrip()[-1:] in (".", "!", "?")),
        float(bool(_EMOJI.search(text[-2:]))),
        float(bool(text[:1].islower())),
    ]
    features.extend(text.count(symbol) / w for symbol in _PUNCT)
    features.extend(counts[token] / w for token in _FUNCTION_WORDS)
    return features


def _vectorize(vectorizer: TfidfVectorizer, texts, fit: bool):
    if fit:
        try:
            return vectorizer.fit_transform(texts).tocsr()
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
            # All training posts may be shorter than the requested n-gram.
            # Preserve the all-zero feature space during inference as well.
            vectorizer.vocabulary_ = {}
    if not vectorizer.vocabulary_:
        return csr_matrix((len(texts), 0), dtype=np.float32)
    return vectorizer.transform(texts).tocsr()


class Verifier:
    """Score two cleaned posts. The output is an uncalibrated ranking score."""

    def __init__(self, method: str = "style"):
        if method not in METHODS:
            raise ValueError(f"unknown model {method!r}; choose one of {METHODS}")
        self.method = method
        if method == "word":
            self.vectorizer = TfidfVectorizer(
                analyzer="word", token_pattern=r"(?u)\b\w+\b", dtype=np.float32,
                sublinear_tf=True, max_features=100_000,
            )
        else:
            self.vectorizer = TfidfVectorizer(
                analyzer="char", ngram_range=(3, 3), dtype=np.float32, lowercase=False,
                sublinear_tf=True, max_features=100_000,
            )
        self.classifier = None
        self.fitted = False

    def _encode(self, texts, fit=False):
        lexical_texts = [mask_hashtags(t) for t in texts] if self.method == "style" else texts
        lexical = _vectorize(self.vectorizer, lexical_texts, fit)
        style = None
        if self.method == "style":
            style = np.asarray([_style(t) for t in lexical_texts], dtype=np.float32)
        return lexical, style

    @staticmethod
    def _pair_features(encoded, left, right):
        """Gather only the sparse rows needed for this batch of pairs."""
        lexical, style = encoded
        similarity = np.asarray(
            lexical[left].multiply(lexical[right]).sum(axis=1)
        ).reshape(-1, 1).clip(0, 1)
        if style is None:
            return similarity
        a, b = style[left], style[right]
        # Shared non-zero habits and disagreement carry different information.
        return np.column_stack((similarity, np.abs(a - b), np.minimum(a, b)))

    @staticmethod
    def _post_index(posts):
        index = {post.id: i for i, post in enumerate(posts)}
        if len(index) != len(posts):
            raise ValueError("duplicate post IDs")
        return index

    @staticmethod
    def _indices(index, pairs):
        try:
            left = np.fromiter((index[p.left] for p in pairs), dtype=np.intp, count=len(pairs))
            right = np.fromiter((index[p.right] for p in pairs), dtype=np.intp, count=len(pairs))
        except KeyError as exc:
            raise ValueError(f"pair references unknown post ID {exc.args[0]!r}") from exc
        return left, right

    def _features(self, encoded, left, right):
        for start in range(0, len(left), _BATCH):
            stop = start + _BATCH
            yield self._pair_features(encoded, left[start:stop], right[start:stop])

    def fit(self, posts, pairs):
        """Fit all vocabulary, scaling, and classifier parameters on training data."""
        posts, pairs = list(posts), list(pairs)
        if not posts:
            raise ValueError("training needs at least one post")
        self.fitted = False
        index = self._post_index(posts)
        encoded = self._encode([p.text for p in posts], fit=True)
        if self.method == "style":
            labels = np.asarray([p.same for p in pairs], dtype=bool)
            if len(np.unique(labels)) != 2:
                raise ValueError("style training requires same-author and different-author pairs")
            left, right = self._indices(index, pairs)
            # Write into the final matrix so training doesn't retain a second
            # full set of dense rows while assembling large pair collections.
            features = np.empty((len(pairs), 1 + 2 * encoded[1].shape[1]), dtype=np.float32)
            for start in range(0, len(pairs), _BATCH):
                stop = start + _BATCH
                features[start:stop] = self._pair_features(encoded, left[start:stop], right[start:stop])
            self.classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.2, max_iter=400, solver="liblinear", random_state=0),
            ).fit(features, labels)
        self.fitted = True
        return self

    def _predict(self, features):
        if self.classifier is None:
            return features[:, 0].astype(float)
        # Fix the precision at the score boundary so batched BLAS reductions
        # and individual predictions agree on inclusive decision thresholds.
        return self.classifier.predict_proba(features)[:, 1].round(12)

    def _check_fitted(self):
        if not self.fitted:
            raise ValueError("fit the verifier before scoring")

    def scores(self, posts, pairs) -> np.ndarray:
        self._check_fitted()
        posts, pairs = list(posts), list(pairs)
        if not pairs:
            return np.empty(0, dtype=float)
        index = self._post_index(posts)
        left, right = self._indices(index, pairs)
        encoded = self._encode([p.text for p in posts])
        return np.concatenate([self._predict(f) for f in self._features(encoded, left, right)])

    def score_pair(self, clean_left: str, clean_right: str) -> float:
        self._check_fitted()
        encoded = self._encode([clean_left, clean_right])
        features = self._pair_features(encoded, [0], [1])
        return float(self._predict(features)[0])
