"""Learn discriminative shared character n-grams from disjoint proxy writers."""

from functools import lru_cache

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from author_match.adaptive_trials import sample_adaptive_trials
from author_match.profile_features import independent_profiles, make_profile, adaptive_pair_reason
from author_match.text import mask_hashtags


class PatternEvidence:
    def __init__(self, posts, seed):
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False,
                                          sublinear_tf=True, min_df=3, max_features=48000, dtype=np.float32)
        self.vectorizer.fit(mask_hashtags(p.text) for p in posts)
        trials = sample_adaptive_trials(posts, label_basis="account_proxy", seed=seed)
        profile = lru_cache(maxsize=256)(make_profile)
        rows, labels = [], []
        for t in trials:
            a, b = independent_profiles(profile(t.left), profile(t.right))
            if adaptive_pair_reason(a, b):
                continue
            rows.append(self.row(a, b))
            labels.append(t.same)
        self.model = LogisticRegression(C=2, class_weight="balanced", solver="liblinear",
                                        random_state=seed).fit(sparse.vstack(rows), labels)
        self.training_trials = len(rows)
        self.pool.cache_clear()

    @lru_cache(maxsize=384)
    def pool(self, posts):
        vectors = self.vectorizer.transform([mask_hashtags(p) for p in posts])
        return normalize(sparse.csr_matrix(vectors.sum(axis=0)))

    def row(self, a, b):
        left, right = [self.pool(tuple(p._originals[k] for k in sorted(p._originals))) for p in (a, b)]
        shared = left.multiply(right).tocsr()
        cosine = float(shared.sum())
        # Square roots retain weak but repeated shared patterns. Each column
        # receives its own supervised weight; overall cosine is a separate cue.
        shared.data = np.sqrt(shared.data)
        return sparse.hstack((shared, sparse.csr_matrix([[cosine]])), format="csr")

    def score(self, a, b):
        row = self.row(a, b)
        return [float(self.model.decision_function(row)[0]), float(row[0, -1])]
