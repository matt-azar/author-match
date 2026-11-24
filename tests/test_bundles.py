import random
import unittest

import numpy as np

from author_match.bundles import (
    BundlePolicy, _sample, calibrate_bundles, compare_bundles, evaluate_bundles,
)
from author_match.data import Post
from author_match.metrics import LengthPolicy, Thresholds
from author_match.text import word_count


class StyleMarker:
    def scores(self, posts, pairs):
        first = {p.id: p.text.split(" ")[0] for p in posts}
        return np.asarray([0.9 if first[p.left] == first[p.right] else 0.1 for p in pairs])


def corpus(author_count=24, offset=0):
    rng = random.Random(123 + offset)
    vocabulary = [f"word{k}" for k in range(240)]
    result = []
    for i in range(offset, offset + author_count):
        for j in range(10):
            # Threads are shared across different writers as well as author-specific.
            thread = f"common-{j}" if j < 2 else f"thread-{i}-{j}"
            text = f"marker{i} " + " ".join(rng.sample(vocabulary, 13))
            result.append(Post(f"{i}-{j}", str(i), thread, str(i % 3), text))
    return result


class BundleTests(unittest.TestCase):
    def test_matching_is_reproducible_and_bounded(self):
        posts = corpus()
        for mode in ("1x5", "5x5"):
            sides, trials = _sample(posts, mode, 77)
            self.assertEqual(trials, _sample(reversed(posts), mode, 77)[1])
            self.assertGreater(len(trials), 0)
            negatives = [p for p in trials if not p.same]
            self.assertEqual(len(negatives), len({p.left for p in negatives}))
            self.assertEqual(len(negatives), len({p.right for p in negatives}))
            for pair in trials:
                left, right = sides[pair.left], sides[pair.right]
                threads = [p.thread for p in left.posts + right.posts]
                self.assertEqual(len(threads), len(set(threads)))
                self.assertEqual(pair.same, left.summary.author == right.summary.author)

    def test_calibration_and_live_scoring_use_same_units(self):
        posts = corpus()
        model = StyleMarker()
        for mode in ("1x5", "5x5"):
            policy, calibration = calibrate_bundles(model, posts, mode, false_match_rate=.2, false_reject_rate=.2)
            self.assertEqual(calibration["overall"]["same_pairs"], calibration["overall"]["different_pairs"])
            self.assertLessEqual(calibration["overall"]["same_pairs"], 24)
            test = corpus(22, offset=100)
            result = evaluate_bundles(model, test, policy, bootstrap_samples=0)
            self.assertEqual(result["overall"]["recall"], 1)
            self.assertEqual(result["overall"]["false_match_rate"], 0)
            sides, trials = _sample(test, mode, 42)
            for pair in trials[:6]:
                left = [p.text for p in sides[pair.left].posts]
                right = [p.text for p in sides[pair.right].posts]
                match = compare_bundles(model, left, right, policy)
                self.assertEqual(match["decision"], "same_author" if pair.same else "different_author")
                self.assertEqual(match["score"], compare_bundles(model, right, left, policy)["score"])
                self.assertEqual(match["words"], [sum(map(word_count, left)), sum(map(word_count, right))])
            with self.assertRaises(ValueError):
                compare_bundles(model, ["another post"], ["one post"], policy)

    def test_copied_insufficient_and_empty_calibration(self):
        model = StyleMarker()
        policy, summary = calibrate_bundles(model, [], "1x5")
        self.assertEqual(summary["overall"]["pairs"], 0)
        texts = [p.text for p in corpus(1)]
        self.assertEqual(compare_bundles(model, texts[:1], texts[1:6], policy)["reason"], "no_supported_threshold")
        high = Thresholds(.5, None)
        active = BundlePolicy("1x5", LengthPolicy(high, {"10-24": high}))
        duplicates = texts[1:5] + texts[1:2]
        self.assertEqual(compare_bundles(model, texts[:1], duplicates, active)["reason"], "copied_overlap")
        self.assertEqual(compare_bundles(model, texts[:1], texts[:5], active)["reason"], "copied_overlap")
        weak = ["ah", "right", "what", *texts[1:3]]
        self.assertEqual(compare_bundles(model, texts[:1], weak, active)["reason"], "insufficient_independent_text")
