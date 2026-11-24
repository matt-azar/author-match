import json
import unittest
from dataclasses import asdict, replace
from math import exp, log

import numpy as np

from author_match.profile_probability import (
    AccountModel, eval_report, fit_account_model, posterior, predict_log_lr, quality_intervals, score, window_quality_intervals,
)


class AccountProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(64)
        cls.train_y = np.tile([0, 1], 100)
        cls.cal_y = np.tile([0, 1], 60)
        cls.train = rng.normal(size=(len(cls.train_y), 2)) + cls.train_y[:, None] * [2.5, -0.25]
        cls.cal = rng.normal(size=(len(cls.cal_y), 2)) + cls.cal_y[:, None] * [2.5, -0.25]
        cls.model = fit_account_model(
            cls.train, cls.train_y, cls.cal, cls.cal_y,
            train_authors=range(100), calibration_authors=range(100, 150),
        )

    def test_prior_odds_and_neutral_evidence(self):
        self.assertEqual(posterior(0, 0.0031), 0.0031)
        for p in (0.001, 0.2, 0.5, 0.95):
            self.assertAlmostEqual(posterior(log(4), p), 4 * p / (1 - p + 4 * p))
        for invalid in (None, 0, 1, 2, -1, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                posterior(1.2, invalid)
        with self.assertRaises(ValueError):
            posterior(np.nan, 0.5)

    def test_portable_model_and_same_evidence_under_different_priors(self):
        self.assertEqual(self.model, AccountModel(**json.loads(json.dumps(asdict(self.model), allow_nan=False))))
        self.assertEqual(self.model.calibration_status, "fitted")
        self.assertEqual(self.model.train_author_count, 100)
        self.assertEqual(self.model.calibration_author_count, 50)
        a = score(self.model, [2.5, 0], 0.02)
        b = score(self.model, [2.5, 0], 0.5)
        self.assertEqual(a["log_likelihood_ratio"], b["log_likelihood_ratio"])
        self.assertGreater(b["probability"], a["probability"])
        self.assertAlmostEqual(a["probability"] / (1 - a["probability"]),
                               exp(a["log_likelihood_ratio"]) * 0.02 / 0.98)
        self.assertAlmostEqual(a["log_likelihood_ratio"], predict_log_lr(self.model, [[2.5, 0]])[0])
        with self.assertRaises(ValueError):
            replace(self.model, coefficients=[])
        with self.assertRaises(ValueError):
            replace(self.model, max_abs_log_lr=1e300)
        with self.assertRaises(ValueError):
            replace(self.model, calibration_status="no_score_variation")

    def test_no_extrapolation_increases_evidence_and_sparse_calibration_is_bounded(self):
        far = score(self.model, [100, 0])
        farther = score(self.model, [1e6, 0])
        self.assertTrue(far["extrapolated"])
        self.assertTrue(farther["extrapolated"])
        self.assertEqual(far, farther)
        self.assertLessEqual(abs(far["log_likelihood_ratio"]), log(51))
        self.assertGreater(far["log_likelihood_ratio"], 0)
        model = fit_account_model(self.train, self.train_y, [[0, 0], [2.5, 0]], [0, 1])
        self.assertLessEqual(score(model, [1e6, 0])["probability"], 2 / 3)
        self.assertGreaterEqual(score(model, [-1e6, 0])["probability"], 1 / 3)

    def test_repeated_positive_owners_limit_calibration_strength(self):
        model = fit_account_model(self.train, self.train_y, self.cal, self.cal_y,
                                  train_authors=range(100), calibration_authors=range(100, 150),
                                  calibration_effective_people=4)
        self.assertEqual(model.effective_calibration_size, 4)
        self.assertEqual(model.calibration_author_count, 50)
        self.assertEqual(self.model.effective_calibration_size, 50)
        self.assertAlmostEqual(model.max_abs_log_lr, log(5))
        supported = score(model, [2.5, 0], 0.5)
        self.assertLess(supported["log_likelihood_ratio"], score(self.model, [2.5, 0])["log_likelihood_ratio"])
        self.assertLessEqual(abs(supported["log_likelihood_ratio"]), log(5))
        for invalid in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                fit_account_model(self.train, self.train_y, self.cal, self.cal_y,
                                  calibration_effective_people=invalid)

    def test_no_variation_and_reversed_calibration_produce_neutral_evidence(self):
        for x, labels, status in (
            (np.zeros((4, 2)), [0, 0, 1, 1], "no_score_variation"),
            (self.cal, 1 - self.cal_y, "nonpositive_association"),
        ):
            model = fit_account_model(self.train, self.train_y, x, labels)
            self.assertEqual(model.calibration_status, status)
            self.assertEqual(score(model, [100, 0], prior=0.1)["probability"], 0.1)
            self.assertEqual(score(model, [-100, 0], prior=0.9)["probability"], 0.9)

    def test_input_validation_and_author_holdout(self):
        with self.assertRaises(ValueError):
            fit_account_model(self.train, self.train_y, self.cal, self.cal_y,
                              train_authors=[1, 2], calibration_authors=[2, 3])
        with self.assertRaises(ValueError):
            fit_account_model(self.train, self.train_y, self.cal, self.cal_y, train_authors=[1])
        with self.assertRaises(ValueError):
            fit_account_model(self.train, self.train_y, self.cal, np.zeros(len(self.cal_y)))
        with self.assertRaises(ValueError):
            fit_account_model(self.train, self.train_y, self.cal[:, :1], self.cal_y)
        with self.assertRaises(ValueError):
            predict_log_lr(self.model, [[np.inf, 1]])
        with self.assertRaises(ValueError):
            score(self.model, [[1, 2]], prior=0.2)
        with self.assertRaises(ValueError):
            score(self.model, [1, 2, 3], prior=0.2)

    def test_prior_weighted_metrics_and_reliability(self):
        y = [1, 1, 1, 0]
        p = 0.2
        data = eval_report(y, [0] * 4, p, bins=5)
        self.assertEqual(data["pairs"], 4)
        self.assertEqual((data["same"], data["different"]), (3, 1))
        self.assertEqual(data["roc_auc"], 0.5)
        self.assertAlmostEqual(data["brier"], data["prior_only"]["brier"])
        self.assertAlmostEqual(data["log_loss"], data["prior_only"]["log_loss"])
        self.assertAlmostEqual(data["expected_calibration_error"], 0)
        bucket = next(b for b in data["reliability"] if b["pairs"])
        self.assertAlmostEqual(bucket["observed_same_fraction"], p)
        self.assertAlmostEqual(bucket["mean_probability"], p)
        self.assertAlmostEqual(bucket["target_weight"], 1)
        separated = eval_report(y, [log(10)] * 3 + [-log(10)], p)
        self.assertLess(separated["brier"], data["brier"])
        self.assertLess(separated["log_loss"], data["log_loss"])
        self.assertEqual(separated["roc_auc"], 1.0)
        self.assertEqual(eval_report([0, 1], [-1000, 1000], p)["log_loss"], 0.0)
        with self.assertRaises(ValueError):
            eval_report([1], [0], p)
        with self.assertRaises(ValueError):
            eval_report(y, [0] * 4, p, bins=0)
        json.dumps(separated, allow_nan=False)

    def test_uncertainty_resamples_distinct_people_including_shared_endpoints(self):
        people = np.arange(20)
        left = np.repeat(people, 2)
        right = np.column_stack((people, (people + 1) % 20)).ravel()
        labels = left == right
        scores = np.random.default_rng(15).normal(size=len(left)) + labels
        args = (labels, scores, left, right, 0.1)
        result = quality_intervals(*args, seed=47, bootstrap_samples=100)
        self.assertEqual(result, quality_intervals(labels, scores, right, left, 0.1, seed=47, bootstrap_samples=100))
        self.assertEqual(result["people"], 20)
        self.assertGreaterEqual(result["usable_repetitions"], 80)
        for name in ("brier_improvement", "log_loss_improvement", "roc_auc"):
            lo, hi = result["intervals_95"][name]
            self.assertLess(lo, hi)
        # Replicating each trial does not multiply the number of independent people.
        doubled = quality_intervals(np.tile(labels, 2), np.tile(scores, 2), np.tile(left, 2), np.tile(right, 2),
                                    0.1, seed=47, bootstrap_samples=100)
        for name in result["intervals_95"]:
            np.testing.assert_allclose(result["intervals_95"][name], doubled["intervals_95"][name])
        json.dumps(result, allow_nan=False)

    def test_uncertainty_omits_constant_estimates_and_one_class(self):
        left, right = ["a", "b", "a", "b"], ["a", "b", "b", "a"]
        labels = [1, 1, 0, 0]
        neutral = quality_intervals(labels, [0] * 4, left, right, 0.3, seed=43)
        self.assertEqual(neutral["intervals_95"], {})
        self.assertEqual(quality_intervals([1], [0], ["a"], ["a"])["intervals_95"], {})
        with self.assertRaises(ValueError):
            quality_intervals([0], [0], ["a"], ["a"])
        with self.assertRaises(ValueError):
            quality_intervals(labels, [0] * 4, left, right, bootstrap_samples=-1)

    def test_window_intervals_match_explicit_whole_record_resampling(self):
        # Mix trial counts within windows. A cumulative record contains multiple
        # people and repeated pairs, each of which must get one common multiplier.
        groups = np.repeat(np.arange(4), [4, 3, 4, 2])
        y = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
        scores = np.array([2, 0, 2, -1, 1, 0, 0, -2, -1, 0, 2, -1, 2])
        seed, repetitions, prior = 17, 60, 0.1
        result = window_quality_intervals(y, scores, groups, prior, seed=seed, bootstrap_samples=repetitions)
        expected = {"roc_auc": [], "log_loss_improvement": [], "brier_improvement": []}
        rng = np.random.default_rng(seed)
        for _ in range(repetitions):
            sampled = rng.integers(4, size=4)
            trials = np.concatenate([np.flatnonzero(groups == window) for window in sampled])
            stats = eval_report(y[trials], scores[trials], prior)
            expected["roc_auc"].append(stats["roc_auc"])
            for name in ("log_loss", "brier"):
                expected[name + "_improvement"].append(stats["prior_only"][name] - stats[name])
        self.assertEqual(result["windows"], 4)
        self.assertEqual(result["usable_repetitions"], repetitions)
        for name, values in expected.items():
            np.testing.assert_allclose(result["intervals_95"][name], np.quantile(values, [0.025, 0.975]))
        json.dumps(result, allow_nan=False)

    def test_window_intervals_sparse_missing_and_degenerate(self):
        for windows in (["only"], ["", "b"], [None, "b"], [True, True]):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                window_quality_intervals([0, 1], [1, -1], windows, 0.1)
        single = window_quality_intervals([1, 0], [1, -1], ["one", "one"])
        self.assertEqual(single["intervals_95"], {})
        self.assertEqual(single["usable_repetitions"], 0)
        self.assertEqual(window_quality_intervals([], [], [])["windows"], 0)
        self.assertEqual(window_quality_intervals([0, 1] * 3, [0] * 6, ["a", "a", "b", "b", "c", "c"])
                         ["intervals_95"], {})
