"""Text-only comparisons between two arbitrary collections of separate posts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import re
from typing import Iterable

import numpy as np
from sklearn.utils import murmurhash3_32

from .data import Post, _deduplicate
from .model import _FUNCTION_WORDS, _style
from .text import clean_text, comparison_words, mask_hashtags

_HASH_BINS = 1 << 16
_SKETCH_SIZE = 4096
_FUNCTION = frozenset(_FUNCTION_WORDS)
_TOKENS = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|[^\w\s]|\n", re.UNICODE)
_STYLE_DIM = len(_style(""))
MIN_POSTS = 2
MIN_WORDS = 15
FEATURE_VERSION = 1
FEATURE_COUNT = 9 + 3 * _STYLE_DIM


@lru_cache(maxsize=32768)
def _bucket(token):
    return murmurhash3_32(token, positive=True) % _HASH_BINS


def _shingles(tokens):
    for i in range(len(tokens) - 4):
        yield int.from_bytes(hashlib.blake2b(
            " ".join(tokens[i:i + 5]).encode(), digest_size=8,
        ).digest(), "big")


def _shape(token):
    if not token.isalpha():
        return "DIGIT" if token.isdecimal() else token
    folded = token.casefold()
    if folded in _FUNCTION:
        return folded
    case = "UP" if token.isupper() else "CAP" if token.istitle() else "LOW" if token.islower() else "MIX"
    return f"<{case}{min(len(token), 10)}>"


@dataclass
class _Distribution:
    bins: np.ndarray
    mass: np.ndarray
    norm: float

    @classmethod
    def build(cls, counts):
        bins = np.asarray(sorted(counts), dtype=np.int32)
        mass = np.fromiter((counts[i] for i in bins), dtype=np.float64, count=len(bins))
        mass /= max(mass.sum(), 1)
        return cls(bins, mass, float(np.linalg.norm(mass)))


@dataclass(eq=False)
class ProfileSummary:
    received: int
    posts: int
    words: int
    reason: str | None
    order_key: bytes
    # Keep a single original per usable post to remove shared writing before scoring.
    # The fitted classifier needs only the pair features. It never needs these texts.
    _originals: dict = field(repr=False)
    _distributions: tuple = field(repr=False)
    _mean: np.ndarray = field(repr=False)
    _std: np.ndarray = field(repr=False)
    _shingles: np.ndarray = field(repr=False)
    _truncated: bool = field(repr=False)


def _summarize(originals, received):
    char, word, shape = Counter(), Counter(), Counter()
    sketch = set()
    truncated = False
    total_words = 0
    total = np.zeros(_STYLE_DIM)
    squares = np.zeros_like(total)
    # Stable ordering makes summation and duplicate handling deterministic.
    order_digest = hashlib.blake2b(digest_size=16)
    for key in sorted(originals):
        order_digest.update(key)
        raw = originals[key]
        tokens = comparison_words(raw)
        total_words += len(tokens)
        for signature in _shingles(tokens):
            sketch.add(signature)
            if len(sketch) >= 2 * _SKETCH_SIZE:
                sketch = set(sorted(sketch)[:_SKETCH_SIZE])
                truncated = True
        text = mask_hashtags(raw)
        for width in (3, 4):
            char.update(_bucket(text[i:i + width]) for i in range(len(text) - width + 1))
        word.update(_bucket(token) for token in comparison_words(text))
        shapes = [_shape(token) for token in _TOKENS.findall(text)]
        # Bigrams end at the post boundary even when the next post begins immediately.
        shape.update(_bucket(a + "\x00" + b) for a, b in zip(shapes, shapes[1:]))
        style = np.asarray(_style(text))
        total += style
        squares += style * style

    n = len(originals)
    mean = total / max(1, n)
    std = np.sqrt(np.maximum(0., squares / max(1, n) - mean * mean))
    reason = None
    if n < MIN_POSTS:
        reason = "too_few_distinct_posts"
    elif total_words < MIN_WORDS:
        reason = "too_few_profile_words"
    truncated |= len(sketch) > _SKETCH_SIZE
    signatures = np.asarray(sorted(sketch)[:_SKETCH_SIZE], dtype=np.uint64)
    return ProfileSummary(received, n, total_words, reason, order_digest.digest(), originals,
                          tuple(_Distribution.build(c) for c in (char, word, shape)),
                          mean, std, signatures, truncated)


def usable_post_key(cleaned, tokens) -> bytes | None:
    """Canonical post fingerprint or None when the text lacks usable prose."""
    if not tokens or not any(char.isalpha() for char in cleaned):
        return None
    if len(tokens) >= 4 and len(set(tokens)) < 2:
        return None
    return hashlib.blake2b(" ".join(tokens).encode(), digest_size=16).digest()


def make_profile(texts: Iterable[str]) -> ProfileSummary:
    """Summarize unique original posts. Each supplied item must be a separate post."""
    if isinstance(texts, (str, bytes)):
        raise ValueError("pass an iterable of individual post strings")
    originals = {}
    received = 0
    for text in texts:
        received += 1
        if not isinstance(text, str):
            raise ValueError("each post must be a string")
        cleaned = clean_text(text)
        tokens = comparison_words(cleaned)
        identity = usable_post_key(cleaned, tokens)
        if identity is None:
            continue
        current = originals.get(identity)
        if current is None or cleaned < current:
            originals[identity] = cleaned
    # Use the same near-copy criterion as the corpus pipeline for long edits.
    if len(originals) > 1 and any(len(comparison_words(t)) >= 12 for t in originals.values()):
        candidates = (Post(key.hex(), "", "", "", originals[key]) for key in sorted(originals))
        retained = _deduplicate(candidates, Counter())
        originals = {bytes.fromhex(p.id): p.text for p in retained}
    return _summarize(originals, received)


def limit_profile(profile, n):
    """Use a nested, deterministic sample independent of the comparison result."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("profile limit must be a positive integer")
    if profile.posts <= n:
        return profile
    keys = sorted(profile._originals)[:n]
    return _summarize({key: profile._originals[key] for key in keys}, profile.received)


def independent_profiles(left, right):
    """Remove canonical posts shared by both sides. Preserve the input counts."""
    if left._originals.keys().isdisjoint(right._originals):
        return left, right
    def remaining(a, b):
        return _summarize({key: text for key, text in a._originals.items() if key not in b._originals},
                          a.received)
    return remaining(left, right), remaining(right, left)


def pair_reason(left, right):
    left, right = independent_profiles(left, right)
    if left.reason or right.reason:
        # A fixed precedence retains symmetry when the sides fail different tests.
        if min(left.posts, right.posts) < MIN_POSTS:
            return "too_few_distinct_posts"
        return "too_few_profile_words"
    # A common sampling threshold keeps the containment estimate comparable
    # when one history has many more shingles than the other.
    a, b = left._shingles, right._shingles
    if not len(a) or not len(b):
        return None
    cuts = []
    if left._truncated:
        cuts.append(a[-1])
    if right._truncated:
        cuts.append(b[-1])
    if cuts:
        cutoff = min(cuts)
        a, b = a[a <= cutoff], b[b <= cutoff]
    shared = len(np.intersect1d(a, b, assume_unique=True))
    if shared >= 4 and shared >= .6 * min(len(a), len(b)):
        return "substantial_copied_passages"
    return None


def _similarities(a, b):
    _, ai, bi = np.intersect1d(a.bins, b.bins, return_indices=True, assume_unique=True)
    x, y = a.mass[ai], b.mass[bi]
    product = x * y
    cosine = product.sum() / (a.norm * b.norm) if a.norm and b.norm else 0.
    return [cosine, np.sqrt(product).sum(), np.minimum(x, y).sum()]


def pair_features(left, right) -> np.ndarray:
    """Symmetric fixed-length vector over content, punctuation, shape and style.

    Hash distributions pool occurrences within individual posts. Style moments
    give each independent post equal weight. Sample counts are returned as
    evidence metadata separately from these features.
    """
    left, right = independent_profiles(left, right)
    similarity = []
    for a, b in zip(left._distributions, right._distributions):
        similarity.extend(_similarities(a, b))
    return np.concatenate([similarity, np.abs(left._mean - right._mean),
                           np.minimum(left._mean, right._mean), np.abs(left._std - right._std)])
