from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from author_match.data import Pair, Post
from author_match.bundles import BundlePolicy, compare_bundles
from author_match.experiment import Artifact, compare
from author_match.metrics import LengthPolicy, Thresholds
from author_match.model import METHODS, Verifier
from author_match.portable import export_cpu, load_cpu


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.json"
        self.posts = [
            Post("a", "x", "1", "k", "i have sooo many questions rn!! 😂"),
            Post("b", "x", "2", "k", "omggg does anyone even know?? 😂"),
            Post("c", "y", "1", "k", "My question appears to have been answered."),
            Post("d", "y", "2", "k", "I would prefer a formal explanation tomorrow."),
        ]
        self.pairs = [Pair(a, b, same, "test") for a, b, same in (
            ("a", "b", True), ("c", "d", True), ("a", "d", False), ("b", "c", False),
        )]
        self.policy = LengthPolicy(
            Thresholds(0.8, 0.1), {"<10": Thresholds(None, 0.05)},
            {"<10": {"same_pairs": 121, "different_pairs": 139, "eligible_same": 120, "eligible_different": 136}},
        )

    def artifact(self, method, posts=None):
        model = Verifier(method).fit(self.posts if posts is None else posts, self.pairs)
        return Artifact(model, self.policy, {"method": method, "author_match_version": "test-version",
                                            "format_version": 1, "thresholds": asdict(self.policy)},
                        fit_authors=("0" * 32, "e" * 32))

    def test_portable_models_reproduce_native_scores_and_decisions(self):
        for method in METHODS:
            original = self.artifact(method)
            for suffix in (".json", ".json.gz"):
                with self.subTest(method=method, suffix=suffix):
                    path = self.path.with_suffix(suffix)
                    export_cpu(original, path)
                    loaded = load_cpu(path)
                    self.assertEqual(loaded.thresholds, original.thresholds)
                    self.assertEqual(loaded.provenance, original.provenance)
                    self.assertEqual(loaded.fit_authors, original.fit_authors)
                    np.testing.assert_array_equal(loaded.verifier.scores(self.posts, self.pairs),
                                                  original.verifier.scores(self.posts, self.pairs))
                    for left, right in (
                        (self.posts[0].text, self.posts[1].text),
                        ("a different post about today's interesting news", "Wait, is that generally true?"),
                        ("💭", "yep"), (self.posts[0].text, self.posts[0].text),
                    ):
                        self.assertEqual(compare(loaded, left, right), compare(original, left, right))
                        self.assertEqual(compare(loaded, left, right)["score"], compare(loaded, right, left)["score"])
                    before = path.read_bytes()
                    export_cpu(loaded, path)
                    self.assertEqual(path.read_bytes(), before)

    def test_empty_vocabulary(self):
        posts = [Post(p.id, p.author, p.thread, p.community, "") for p in self.posts]
        for method in METHODS:
            original = self.artifact(method, posts)
            export_cpu(original, self.path)
            loaded = load_cpu(self.path)
            self.assertEqual(compare(loaded, self.posts[0].text, self.posts[1].text),
                             compare(original, self.posts[0].text, self.posts[1].text))

    def test_calibrated_bundles_survive_export(self):
        original = self.artifact("style")
        original.bundle_policies = {"1x5": BundlePolicy("1x5", self.policy)}
        export_cpu(original, self.path)
        loaded = load_cpu(self.path)
        self.assertEqual(loaded.bundle_policies, original.bundle_policies)
        left = ["My name is already listed on there."]
        right = [p.text for p in self.posts] + ["Was there a reason for mentioning that yesterday?"]
        self.assertEqual(compare_bundles(loaded.verifier, left, right, loaded.bundle_policies["1x5"]),
                         compare_bundles(original.verifier, left, right, original.bundle_policies["1x5"]))

    def test_rejects_invalid_checkpoint_parameters(self):
        export_cpu(self.artifact("style"), self.path)
        good = json.loads(self.path.read_text())
        invalid = [
            (("format",), "other"), (("method",), "unrecognized"),
            (("vocabulary",), ["abc", "abc"]), (("idf",), []),
            (("linear", "scale"), [0] * len(good["linear"]["scale"])),
            (("linear", "mean"), [1]), (("linear", "intercept"), float("nan")),
            (("policy", "overall", "upper"), "0.5"),
            (("policy", "calibration_support", "<10", "eligible_same"), 500),
            (("bundle_policies",), {"1x1": good["policy"]}),
            (("fit_authors",), ["1" * 32, "0" * 32]),
            (("fit_authors",), ["A" * 32]),
            (("provenance", "thresholds", "overall", "lower"), 0.3),
        ]
        for keys, replacement in invalid:
            with self.subTest(keys=keys):
                payload = deepcopy(good)
                ref = payload
                for key in keys[:-1]:
                    ref = ref[key]
                ref[keys[-1]] = replacement
                self.path.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    load_cpu(self.path)
        self.path.write_bytes(b"\x80\x04pickle payload")
        with self.assertRaises(ValueError):
            load_cpu(self.path)

    def test_only_fitted_cpu_models_are_exportable(self):
        for invalid in (object(), Verifier("word")):
            with self.assertRaisesRegex(ValueError, "fitted"):
                export_cpu(Artifact(invalid, self.policy, {}), self.path)
