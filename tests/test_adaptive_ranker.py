import json
import unittest
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from author_match.adaptive_ranker import AdaptiveRanker, fit_ranker


class RankerTests(unittest.TestCase):
    def test_export_matches_both_sklearn_estimators_on_new_rows(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(300, 5))
        y = (x[:, 0] * x[:, 1] + x[:, 2] > 0)
        linear = LogisticRegression(C=0.5, class_weight="balanced", max_iter=600)
        trees = GradientBoostingClassifier(n_estimators=64, learning_rate=0.06, max_depth=2,
                                           min_samples_leaf=7, init="zero", random_state=19)
        with patch("author_match.adaptive_ranker.LogisticRegression", return_value=linear), \
                patch("author_match.adaptive_ranker.GradientBoostingClassifier", return_value=trees):
            trained = fit_ranker(x, y, 19)
        ranker = AdaptiveRanker(**json.loads(json.dumps(asdict(trained))))
        new = rng.normal(size=(51, 5))
        expected = (linear.decision_function(StandardScaler().fit(x).transform(new))
                    + trees.decision_function(new)) * 0.5
        np.testing.assert_allclose(ranker.margins(new), expected, atol=1e-12)
        np.testing.assert_allclose([ranker.margin(row) for row in new], expected, atol=1e-12)

    def test_invalid_tree_cannot_cycle_or_reference_unknown_features(self):
        for nodes in ([[0, 1., 0, 1], [1.]], [[3, 1., 1, 2], [0.], [1.]], [[1.], [2.]]):
            with self.assertRaises(ValueError):
                AdaptiveRanker([1.], 0., [nodes])
