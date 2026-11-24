from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_suspected_pairs import fit, predict, signals
from author_match.profile_features import make_profile


class SuspectedPairExperiments(unittest.TestCase):
    def test_compression_and_boundary_scores_are_symmetric(self):
        a = make_profile(["so that happened...", "i saw it too!!"])
        b = make_profile(["Something happened to me too."])
        np.testing.assert_allclose(signals(a, b), signals(b, a))

    def test_degenerate_inputs_carry_neutral_evidence(self):
        x = np.array([[0.], [0.2], [1.], [1.1], [np.nan]])
        model = fit(x, np.array([0, 0, 1, 1, 0]), np.array([1, 2, 1, 2, 2]), [0])
        pred = predict(model, x)
        self.assertEqual(pred[-1], 0.)
        self.assertGreater(pred[3], pred[0])
