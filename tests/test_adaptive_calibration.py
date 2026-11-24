import json
import unittest

import numpy as np

from author_match.adaptive_calibration import AdaptiveCalibration, fit_adaptive_calibration


class AdaptiveCalibrationTest(unittest.TestCase):
    @staticmethod
    def trials():
        rows, scores = [], []
        for owner in range(18):
            a, b = 4 + owner % 5, 6 + owner % 8
            sizes = [a, b, 10 * a, 12 * b]
            rows.extend([[*sizes, 1, owner, owner], [*sizes, 0, owner, 100 + owner]])
            scores.extend([0.9 + owner / 30, -0.6 - owner / 32])
        return rows, scores

    def fitted(self):
        rows, scores = self.trials()
        return AdaptiveCalibration.from_dict(json.loads(json.dumps(fit_adaptive_calibration(scores, rows))))

    def test_monotone_and_prior_invariant_likelihood(self):
        model = self.fitted()
        sizes = [6, 12, 60, 144]
        probabilities = [model.estimate(x, sizes, 0.20)["probability"] for x in [-1, -.4, .4, 1.0]]
        self.assertEqual(sorted(probabilities), probabilities)
        self.assertLess(probabilities[0], .2)
        self.assertGreater(probabilities[-1], .2)
        p10 = model.estimate(1.0, sizes, 0.1)
        p50 = model.estimate(1.0, sizes, 0.5)
        self.assertEqual(p10["log_likelihood_ratio"], p50["log_likelihood_ratio"])
        self.assertAlmostEqual(p10["probability"] / (1 - p10["probability"]),
                               np.exp(p50["log_likelihood_ratio"]) / 9)

    def test_swap_orientation_and_writer_replication_invariance(self):
        model = self.fitted()
        sizes, swapped = [5, 9, 61, 73], [9, 5, 73, 61]
        orig = model.estimate(.8, sizes, .1)
        rev = model.estimate(.8, swapped, .1)
        for field in orig:
            self.assertAlmostEqual(orig[field], rev[field])
        rows, scores = self.trials()
        # Repeated trials from one positive writer and one negative pair must
        # not make calibration acquire more independent people or more evidence.
        rows += [rows[0], rows[1]] * 80
        scores += [scores[0], scores[1]] * 80
        duplicate = AdaptiveCalibration.from_dict(fit_adaptive_calibration(scores, rows))
        dup = duplicate.estimate(.8, sizes, .1)
        for field in orig:
            self.assertAlmostEqual(orig[field], dup[field], places=7)

    def test_weak_neighborhood_continuously_returns_to_prior(self):
        model = self.fitted()
        normal = model.estimate(1., [5, 8, 50, 96], .1)
        near = model.estimate(1., [5, 8, 50, 97], .1)
        far = model.estimate(1., [1, 1, 1, 1], .1)
        self.assertLess(abs(near["support_weight"] - normal["support_weight"]), .001)
        self.assertLess(far["support_weight"], normal["support_weight"])
        self.assertAlmostEqual(far["probability"], .1, places=8)
        self.assertTrue(model.estimate(1e6, [100000, 10000, 900000, 90000], .1)["extrapolated"])

    def test_reusing_one_negative_writer_cannot_certify_many_independent_pairs(self):
        rows, scores = self.trials()
        for row in rows:
            if not row[4]:
                row[5] = 1000
        model = AdaptiveCalibration.from_dict(fit_adaptive_calibration(scores, rows))
        estimate = model.estimate(1., [4, 8, 41, 85], .1)
        self.assertGreater(estimate["negative_writer_support"], 1.)
        self.assertLessEqual(estimate["effective_calibration_pairs"], 1.)

    def test_direction_and_constant_fail_to_supply_evidence(self):
        rows, scores = self.trials()
        for values in ([1.] * len(scores), -np.array(scores)):
            model = AdaptiveCalibration.from_dict(fit_adaptive_calibration(values, rows))
            self.assertNotEqual(model.status, "fitted")
            estimate = model.estimate(112., [4, 4, 21, 31], .006)
            self.assertEqual(estimate["probability"], .006)
            self.assertEqual(estimate["log_likelihood_ratio"], 0.)

    def test_malformed_model_and_inputs(self):
        rows, scores = self.trials()
        payload = fit_adaptive_calibration(scores, rows)
        for field, invalid in (("version", True), ("raw_min", float("nan")), ("scale", 0.),
                               ("slopes", [1., -1.]), ("bandwidth", [1., 0.]),
                               ("rows", [[2, 5, 20, 40, 1, 0, 1]])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                AdaptiveCalibration.from_dict({**payload, field: invalid})
        model = AdaptiveCalibration.from_dict(payload)
        for sizes in ([6, 0, 42, 18], [0.5, 2, 20, 41], [10, 3]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                model.estimate(.9, sizes, .1)
        with self.assertRaises(ValueError):
            model.estimate(float("nan"), [4, 3, 20, 18], .1)
        with self.assertRaises(ValueError):
            model.estimate(.5, [4, 3, 20, 18], 0.)
