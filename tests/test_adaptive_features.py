import unittest

import numpy as np

from author_match import adaptive_features
from author_match.profile_features import FEATURE_COUNT as POOLED_WIDTH, make_profile


def distribution_features(left, right):
    return adaptive_features.pair_features(make_profile(left), make_profile(right))[POOLED_WIDTH:].reshape(-1, 3)


class AdaptiveFeatureTests(unittest.TestCase):
    def test_joint_post_habits_can_differ_with_identical_marginals(self):
        # Same punctuation and capitalization frequencies; the all-caps author
        # can use exclamation marks specifically, or reserve them for lowercase.
        anchor = ["PLUM PEAR!!", "mint cake."]
        same_registers = ["LAMB RICE!!", "corn milk."]
        reversed_registers = ["LAMB RICE.", "corn milk!!"]
        aligned = distribution_features(anchor, same_registers)
        reversed_ = distribution_features(anchor, reversed_registers)
        np.testing.assert_allclose(aligned[:, 1], 0, atol=1e-13)
        self.assertGreater(reversed_[-1, 1], 0.01)

    def test_symmetric_order_independent_and_copy_invariant(self):
        left = ["maybe it will rain", "funny that you should ask!!", "HELLO again"]
        right = ["i'm running late", "try next week", "The blue one is on the porch.",
                 "look next to the cabinet"]
        shared = "the two people read the exact same post"
        original = distribution_features(left, right)
        np.testing.assert_array_equal(original, distribution_features(right[::-1], left[::-1]))
        np.testing.assert_array_equal(original, distribution_features([*left, shared, left[0]], [*right, shared]))
        self.assertTrue(np.isfinite(original).all())

    def test_correction_agrees_with_explicit_distinct_post_pairs(self):
        left = ["I need that book", "looks great!!", "maybe tomorrow..."]
        right = ["Ok?", "Can you find your place?"]
        a, b = make_profile(left), make_profile(right)
        def post_vectors(posts):
            return [adaptive_features._embedded_mean(make_profile([post])) for post in posts]

        def dot(x, y):
            return (x * y).sum(axis=(-2, -1))

        def expected_within(pool):
            return np.mean([dot(x, y) for i, x in enumerate(pool) for j, y in enumerate(pool)
                            if i != j], axis=0)
        za, zb = post_vectors(left), post_vectors(right)
        cross = np.mean([dot(x, y) for x in za for y in zb], axis=0)
        expected = expected_within(za) + expected_within(zb) - 2 * cross
        result = adaptive_features.pair_features(a, b)[POOLED_WIDTH:].reshape(-1, 3)
        np.testing.assert_allclose(result[:, 0], cross.ravel(), atol=1e-14)
        np.testing.assert_allclose(result[:, 2], expected.ravel(), atol=1e-14)

    def test_all_batches_contribute(self):
        many = [f"my order number {i} was filled" for i in range(260)]
        other = ["someone tell me when", "please don't hurry", "another weekend away"]
        profile = make_profile(many)
        self.assertEqual(profile.posts, len(many))
        before = distribution_features(many, other)
        after = distribution_features([*many, "WHAT?!?! impressive!!!"], other)
        self.assertGreater(np.linalg.norm(after - before), 1e-6)

    def test_singleton_and_empty_are_finite_with_constant_width(self):
        for left, right in (([], []), ([], ["hello"]), (["hello"], ["ok", "maybe"])):
            values = adaptive_features.pair_features(make_profile(left), make_profile(right))
            self.assertEqual(values.shape, (adaptive_features.FEATURE_COUNT,))
            self.assertTrue(np.isfinite(values).all())
