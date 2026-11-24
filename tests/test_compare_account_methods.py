import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from author_match.profile_probability import posterior
from author_match.profile_trials import ProfileTrial

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_account_methods import _metrics, compare_predictions, draw_windows, histories, lexical_matches


class AccountComparisonTests(unittest.TestCase):
    def test_weighted_losses_and_ties_match_reference(self):
        y = np.array([True, False, True, False, False, True])
        evidence = np.array([-1, -1, 3, 1, 1, 1])
        counts = np.array([[1, 1, 1, 1, 1, 1], [3, 2, 1, 0, 1, 0]])
        prior = 0.1
        metrics = _metrics(y, evidence, counts, prior)
        for i, count in enumerate(counts):
            weights = np.zeros(len(y))
            weights[y] = count[y] * prior / count[y].sum()
            weights[~y] = count[~y] * (1 - prior) / count[~y].sum()
            probs = posterior(evidence, prior)
            self.assertAlmostEqual(metrics["brier"][i], brier_score_loss(y, probs, sample_weight=weights))
            self.assertAlmostEqual(metrics["log_loss"][i], log_loss(y, probs, sample_weight=weights))
            self.assertAlmostEqual(metrics["roc_auc"][i], roc_auc_score(y, evidence, sample_weight=weights))

    def test_paired_resamples_preserve_entire_windows_and_same_model_has_zero_gain(self):
        counts = draw_windows(["a", "b", "a", "b"], seed=275, replicates=80)
        np.testing.assert_equal(counts[:, 0], counts[:, 2])
        np.testing.assert_equal(counts[:, 1], counts[:, 3])
        np.testing.assert_equal(counts[:, 0] + counts[:, 1], 2)
        labels = [False, False, True, True]
        scores = [0.0, -0.4, -0.1, 0.8]
        out = compare_predictions(labels, scores, scores, counts, 0.5)
        self.assertEqual(out["usable_bootstrap_replicates"], 80)
        for metric in out["metrics"].values():
            self.assertEqual(metric["candidate_gain"], 0)
            self.assertEqual(metric["paired_gain_interval_95"], [0, 0])
        better = compare_predictions(labels, [0] * 4, [-1, -1, 1, 1], counts, 0.5)
        self.assertTrue(all(m["candidate_gain"] > 0 for m in better["metrics"].values()))

    def test_one_post_stress_does_not_depend_on_label_or_post_order(self):
        first = ProfileTrial(("a", "b", "c"), ("d", "e"), True, "p", "p")
        shuffled = ProfileTrial(("b", "c", "a"), ("e", "d"), False, "p", "q")
        self.assertEqual(histories(first, "original"), (first.left, first.right))
        self.assertEqual(histories(first, "one_vs_one"), histories(shuffled, "one_vs_one"))
        left, right = histories(first, "one_vs_history")
        self.assertEqual(len(left), 1)
        self.assertEqual(right, first.right)

    def test_topic_matching_maximizes_pairs_and_preserves_anchor_and_length(self):
        trials = [ProfileTrial(("anchor",), (str(i),), same, "p", "p" if same else "q")
                  for i, same in enumerate((True, True, False, False, False))]
        trials.extend([
            ProfileTrial(("different anchor",), ("5",), False, "p", "q"),
            ProfileTrial(("anchor",), ("6", "7"), False, "p", "q"),
        ])
        # Greedy nearest matching would consume (1, 2) and leave 0 unmatched.
        windows = ["one"] * 4 + ["two", "one", "one"]
        matches = lexical_matches(trials, windows, [0.0, 0.04, 0.03, 0.08, 0.0, 0.0, 0.0])
        self.assertEqual(set(matches), {(0, 2), (1, 3)})
