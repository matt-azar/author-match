"""Compare distributions of individual post styles across complete histories."""

from __future__ import annotations

from itertools import islice
from weakref import WeakKeyDictionary

import numpy as np

from .model import _FUNCTION_WORDS, _style
from .profile_features import (FEATURE_COUNT as POOLED_FEATURE_COUNT, ProfileSummary,
                               independent_profiles, pair_features as pooled_pair_features)
from .text import mask_hashtags


FEATURE_VERSION = 2
_D = len(_style(""))
_FUNCTION_START = _D - len(_FUNCTION_WORDS)
# Each coordinate group describes one post. The joint view can distinguish
# authors whose punctuation and formality co-vary, even if their marginals agree.
_VIEWS = (slice(0, _FUNCTION_START), slice(_FUNCTION_START, _D), slice(0, _D))
_BANDWIDTHS = (0.35, 1.0)
_FREQUENCIES = 64
_BATCH = 128
FEATURE_COUNT = POOLED_FEATURE_COUNT + 3 * len(_VIEWS) * len(_BANDWIDTHS)


def _projections():
    # The legacy RandomState has stable seeded draws across NumPy releases.
    # These are fixed quadrature directions, not learned or sampled at inference.
    rng = np.random.RandomState(23861)
    return tuple(rng.standard_normal((view.stop - view.start, _FREQUENCIES))
                 for view in _VIEWS)


_PROJECTIONS = _projections()
# Training repeatedly compares the same profiles. Cache only while the caller
# holds that profile, so completed comparisons don't leave a permanent text cache.
_EMBEDDINGS = WeakKeyDictionary()


def _embedded_mean(profile: ProfileSummary) -> np.ndarray:
    """Empirical characteristic functions of style, aggregated over all posts."""
    cached = _EMBEDDINGS.get(profile)
    if cached is not None:
        return cached
    sums = np.zeros((len(_VIEWS), len(_BANDWIDTHS), 2, _FREQUENCIES))
    keys = iter(sorted(profile._originals))
    while batch := list(islice(keys, _BATCH)):
        x = np.array([_style(mask_hashtags(profile._originals[key])) for key in batch])
        # Stabilize the different natural units: log word count, mean word
        # length, and nonnegative rates. Square roots preserve rare habits.
        x[:, 0] /= 3
        x[:, 2] /= 4
        np.sqrt(x[:, 3:], out=x[:, 3:])
        for i, (view, projection) in enumerate(zip(_VIEWS, _PROJECTIONS)):
            angles = x[:, view] @ projection
            for j, bandwidth in enumerate(_BANDWIDTHS):
                t = angles / bandwidth
                sums[i, j, 0] += np.cos(t).sum(axis=0)
                sums[i, j, 1] += np.sin(t).sum(axis=0)
    sums /= max(1, profile.posts) * np.sqrt(_FREQUENCIES)
    _EMBEDDINGS[profile] = sums
    return sums


def _within(norm, n):
    # Exclude each post's self-comparison. The sine/cosine representation
    # has exactly unit squared norm for each post. For a singleton, only its
    # observed point mass is available.
    return (n * norm - 1) / (n - 1) if n > 1 else norm


def pair_features(left: ProfileSummary, right: ProfileSummary) -> np.ndarray:
    """Pooled features plus six non-linear comparisons of the two style distributions.

    Each kernel (three style views at two bandwidths) returns the estimated
    mean cross-post affinity, squared distance between the empirical kernel
    means, and its within-history bias correction. All comparisons are symmetric.
    """
    left, right = independent_profiles(left, right)
    pooled = pooled_pair_features(left, right)
    a, b = _embedded_mean(left), _embedded_mean(right)
    axes = (-2, -1)
    cross = (a * b).sum(axis=axes)
    distance = ((a - b) ** 2).sum(axis=axes)
    within_left = _within((a * a).sum(axis=axes), left.posts)
    within_right = _within((b * b).sum(axis=axes), right.posts)
    corrected_distance = within_left + within_right - 2 * cross
    return np.concatenate((pooled, np.stack((cross, distance, corrected_distance), axis=-1).ravel()))
