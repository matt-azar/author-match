import json
import unittest
from types import SimpleNamespace
from collections import namedtuple
from dataclasses import asdict

import numpy as np

from author_match.metrics import LengthPolicy, Thresholds, calibrate, decision, fit_policy, report, thresholds_for


Post = namedtuple("Post", "id author text")
Pair = namedtuple("Pair", "left right same condition")


class CalibrationTests(unittest.TestCase):
    def test_calibration_respects_inclusive_ties_and_both_budgets(self):
        scores = [0.01] * 95 + [0.8] * 4 + [0.9] + [0.2] * 5 + [0.8] * 90 + [0.99] * 5
        labels = [False] * 100 + [True] * 100
        reasons = [None] * len(scores)
        thresholds = calibrate(scores, labels, reasons)
        self.assertEqual(thresholds, Thresholds(0.99, 0.01))
        decisions = [decision(score, None, thresholds) for score in scores]
        self.assertLessEqual(decisions[:100].count("same_author"), 1)
        self.assertLessEqual(decisions[100:].count("different_author"), 5)
        self.assertEqual(decision(0.99, "copied", thresholds), "inconclusive")
        self.assertEqual(decision(0.8, None, thresholds), "inconclusive")
        json.dumps(asdict(thresholds), allow_nan=False)

    def test_resolution_checks_eligible_opposing_class(self):
        self.assertEqual(calibrate([0.1] * 99 + [0.9], [False] * 99 + [True], [None] * 100),
                         Thresholds(None, None))
        reasons = ["short"] + [None] * 199
        thresholds = calibrate([0.1] * 100 + [0.9] * 100, [False] * 100 + [True] * 100, reasons)
        self.assertEqual(thresholds, Thresholds(None, 0.1))

    def test_overlap_has_one_unambiguous_decision_and_handles_no_evidence(self):
        thresholds = calibrate([0.1, 0.4, 0.6, 0.8, 0.2, 0.45, 0.65, 0.9],
                               [0] * 4 + [1] * 4, [None] * 8, 0.75, 0.75)
        self.assertLess(thresholds.lower, thresholds.upper)
        self.assertEqual(decision(thresholds.lower, None, thresholds), "different_author")
        self.assertEqual(decision(thresholds.upper, None, thresholds), "same_author")
        self.assertEqual(calibrate([], [], []), Thresholds(None, None))
        self.assertEqual(calibrate([0.5] * 200, [0] * 100 + [1] * 100, [None] * 200),
                         Thresholds(None, None))
        for bad_scores, bad_labels, bad_reasons in (([0.1], [], []), ([float("nan")], [1], [None]),
                                                    ([0.2], [2], [None])):
            with self.assertRaises(ValueError):
                calibrate(bad_scores, bad_labels, bad_reasons)

    def test_invalid_thresholds_and_rate(self):
        for thresholds in [(0.5, 0.5), (0.1, 0.9), (None, float("inf"))]:
            with self.assertRaises(ValueError):
                Thresholds(*thresholds)
        for rate in (-0.01, 1, float("nan")):
            with self.assertRaises(ValueError):
                calibrate([], [], [], rate, 0.05)


class LengthPolicyTests(unittest.TestCase):
    def test_global_and_local_evidence_are_both_needed_for_each_call(self):
        policy = LengthPolicy(Thresholds(0.8, 0.3), {
            "<10": Thresholds(None, None),
            "10-24": Thresholds(0.7, 0.1),
            "25-49": Thresholds(0.95, 0.4),
            "50-99": Thresholds(None, 0.2),
        })
        self.assertEqual(thresholds_for(9, policy), Thresholds(None, None))
        self.assertEqual(thresholds_for(24, policy), Thresholds(0.8, 0.1))
        self.assertEqual(thresholds_for(25, policy), Thresholds(0.95, 0.3))
        self.assertEqual(thresholds_for(50, policy), Thresholds(None, 0.2))
        self.assertEqual(thresholds_for(100, policy), Thresholds(None, None))
        self.assertEqual(thresholds_for(8, policy.overall), policy.overall)

    def test_sparse_band_abstains_even_if_overall_threshold_accepts(self):
        posts, pairs, scores, reasons = [], [], [], []
        for length, n in [(50, 120), (25, 20), (9, 4)]:
            for i in range(2 * n):
                same = i < n
                left, right = f"{length}-{i}-l", f"{length}-{i}-r"
                author = str(i)
                posts.extend([Post(left, author, "word " * length),
                              Post(right, author if same else right, "word " * 100)])
                pairs.append(Pair(left, right, same, "separate"))
                scores.append(0.9 if same else 0.1)
                reasons.append(None)
        policy = fit_policy(posts, pairs, scores, reasons)
        self.assertEqual(policy.overall, Thresholds(0.9, 0.1))
        self.assertEqual(thresholds_for(50, policy), Thresholds(0.9, 0.1))
        self.assertEqual(thresholds_for(25, policy), Thresholds(None, 0.1))
        self.assertEqual(thresholds_for(9, policy), Thresholds(None, None))
        self.assertEqual(policy.calibration_support["10-24"]["eligible_different"], 0)
        result = report(posts, pairs, scores, reasons, policy, bootstrap_samples=0)
        self.assertEqual(result["by_length"]["<10"]["coverage"], 0.0)
        self.assertEqual(result["by_length"]["25-49"]["recall"], 0.0)
        self.assertEqual(result["by_length"]["25-49"]["coverage"], 0.5)
        self.assertEqual(result["by_length"]["50-99"]["coverage"], 1.0)
        self.assertEqual(result["thresholds"]["effective_by_length"]["<10"], {"upper": None, "lower": None})
        reverse = [Pair(p.right, p.left, p.same, p.condition) for p in pairs]
        self.assertEqual(policy, fit_policy(posts, reverse, scores, reasons))
        self.assertEqual(result, report(posts, reverse, scores, reasons, policy, bootstrap_samples=0))
        json.dumps(result, allow_nan=False)


class ReportTests(unittest.TestCase):
    def test_abstentions_in_denominators_and_shorter_post_length(self):
        lengths = [9, 10, 24, 25, 49, 50, 99, 100]
        posts = [Post(str(i), str(i // 2), "word " * n) for i, n in enumerate(lengths)]
        pairs = [Pair("0", "1", True, "cross"), Pair("2", "3", True, "within"),
                 Pair("4", "6", False, "within"), Pair("5", "7", False, "cross")]
        output = report(posts, pairs, [0.8, 0.9, 0.9, 0.1], ["copied", None, None, None],
                        Thresholds(0.8, 0.2), bootstrap_samples=50)
        self.assertEqual(output["overall"]["recall"], 0.5)
        self.assertEqual(output["overall"]["false_match_rate"], 0.5)
        self.assertEqual(output["overall"]["false_reject_rate"], 0.0)
        self.assertEqual(output["overall"]["coverage"], 0.75)
        self.assertEqual(output["overall"]["error_among_decided"], 1 / 3)
        self.assertEqual(output["overall"]["evidence_exclusions"], {"copied": 1})
        self.assertEqual(output["by_condition"]["cross"]["authors"], 3)
        self.assertEqual([group["pairs"] for group in output["by_length"].values()], [1, 1, 1, 1, 0])
        self.assertIsNone(output["by_length"]["100+"]["coverage"])
        self.assertIsNone(output["by_length"]["<10"]["roc_auc"])
        self.assertEqual(output["by_condition"]["cross"]["roc_auc"], 1)
        self.assertIsNone(output["by_condition"]["cross"]["roc_auc_eligible"])
        self.assertEqual(output["overall"]["eligible_pairs"], 3)
        self.assertEqual(output["by_time_gap"], {})
        json.dumps(output, allow_nan=False)

    def test_time_gap_boundaries_and_missing_timestamps(self):
        posts = [SimpleNamespace(id=str(i), author=str(i), text="abc", timestamp=t)
                 for i, t in enumerate((0, 86400, 30 * 86400, 180 * 86400, None))]
        pairs = [Pair(str(i), "0", False, "unknown") for i in range(1, 5)]
        result = report(posts, pairs, [0.4] * 4, [None] * 4, Thresholds(None, None), bootstrap_samples=0)
        self.assertEqual({k: v["pairs"] for k, v in result["by_time_gap"].items()},
                         {"<30d": 1, "30-179d": 1, "180d+": 1, "unknown": 1})

    def test_bootstrap_determinism_and_empty_report(self):
        posts = [Post(str(i), str(i // 2), "yeah") for i in range(20)]
        pairs = [Pair(str(i), str(i + 1), i % 2 == 0, "thread") for i in range(19)]
        scores = np.arange(19) / 18
        args = (posts, pairs, scores, [None] * 19, Thresholds(0.75, 0.25))
        a = report(*args, bootstrap_samples=40, seed=123)
        self.assertEqual(a, report(*args, bootstrap_samples=40, seed=123))
        self.assertIn("recall", a["uncertainty"]["intervals"])
        self.assertLessEqual(a["uncertainty"]["intervals"]["coverage"][0],
                             a["uncertainty"]["intervals"]["coverage"][1])
        empty = report([], [], [], [], Thresholds(None, None))
        self.assertIsNone(empty["overall"]["recall"])
        self.assertEqual(empty["overall"]["pairs"], 0)
        json.dumps(empty, allow_nan=False)
