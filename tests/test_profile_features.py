import unittest

import numpy as np

from author_match.profile_features import FEATURE_COUNT, independent_profiles, limit_profile, make_profile, pair_features, pair_reason


class ProfileFeaturesTests(unittest.TestCase):
    def test_clean_deduplicate_count_and_ignore_boundaries(self):
        posts = [
            "> These words come from someone else entirely.\nTomorrow this city will finally have sunshine!!!",
            "Tomorrow this city will finally have sunshine",
            "I still need to water all the tomatoes before breakfast.",
            "> Someone else's entire post", "", "1234",
        ]
        profile = make_profile(iter(posts))
        self.assertEqual((profile.received, profile.posts, profile.words), (6, 2, 17))
        self.assertIsNone(profile.reason)
        reversed_profile = make_profile(reversed(posts))
        np.testing.assert_array_equal(pair_features(profile, make_profile([])),
                                      pair_features(reversed_profile, make_profile([])))
        # No char trigram or shape bigram can span the two posts.
        tiny = make_profile(["a", "b"])
        self.assertEqual(len(tiny._distributions[0].bins), 0)
        self.assertEqual(len(tiny._distributions[2].bins), 0)

    def test_material_shared_posts_are_removed_from_both_sides(self):
        duplicate = "Late last night we went back to the same college cafe!"
        left = ["I have been thinking about buying a bicycle again",
                "Usually I take the train through the valley on Mondays"]
        right = ["The cat was sitting outside all afternoon in the sunshine",
                 "Somebody left a stray grocery bag in that green driveway"]
        a = make_profile([duplicate, *left])
        b = make_profile([duplicate.upper(), *right])
        x, y = independent_profiles(a, b)
        self.assertEqual((x.received, y.received, x.posts, y.posts), (3, 3, 2, 2))
        np.testing.assert_array_equal(pair_features(a, b), pair_features(make_profile(left), make_profile(right)))
        np.testing.assert_array_equal(pair_features(a, b), pair_features(b, a))
        self.assertIsNone(pair_reason(a, b))
        self.assertIsNotNone(pair_reason(a, a))

    def test_edited_copied_paragraphs_do_not_create_matches(self):
        base = ["somebody painted the entire wooden structure white immediately after the police arrived",
                "the gravel path ends abruptly outside a small stone shed beside the back gate"]
        edited = [s + " yesterday anyway" for s in base]
        left, right = make_profile(base), make_profile(edited)
        self.assertEqual(pair_reason(left, right), "substantial_copied_passages")
        self.assertEqual(pair_reason(left, right), pair_reason(right, left))

    def test_unequal_sizes_all_contribute(self):
        a = make_profile(f"on day {n} the ferry was running behind schedule again" for n in range(130))
        b = make_profile(["coffee and a book make my morning commute go by so quickly",
                          "on sundays i avoid the station because they keep the doors locked"])
        self.assertEqual((a.received, a.posts, b.posts), (130, 130, 2))
        self.assertIsNone(pair_reason(a, b))
        features = pair_features(a, b)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features.ndim, 1)
        np.testing.assert_array_equal(features, pair_features(b, a))
        self.assertEqual(features.shape, pair_features(make_profile([]), make_profile([])).shape)

    def test_stable_nested_cap_and_near_copy_removal(self):
        edited = "the last time i actually went to the old market there was no way to cross the main street"
        self.assertEqual(make_profile([edited, edited + " outside"]).posts, 1)
        posts = [f"design {i} uses the handle differently" for i in range(25)]
        full = make_profile(reversed(posts))
        capped = limit_profile(full, 11)
        self.assertEqual((capped.received, capped.posts), (25, 11))
        self.assertIs(limit_profile(capped, 12), capped)
        a, b = limit_profile(capped, 4), limit_profile(make_profile(posts), 4)
        self.assertEqual(a._originals, b._originals)
        self.assertEqual(a.order_key, b.order_key)
        self.assertNotEqual(a.order_key, capped.order_key)
        self.assertEqual(pair_features(a, b).shape, (FEATURE_COUNT,))
        with self.assertRaises(ValueError):
            limit_profile(full, 0)

    def test_too_little_evidence_and_input_validation(self):
        self.assertEqual(make_profile(["nice", "cool"]).reason, "too_few_profile_words")
        self.assertEqual(make_profile(["lol"] * 100).reason, "too_few_distinct_posts")
        self.assertIsNotNone(pair_reason(make_profile(["nice", "cool"]), make_profile(["wow"])))
        for invalid in ("one post", ["another post", 3]):
            with self.assertRaises(ValueError):
                make_profile(invalid)
