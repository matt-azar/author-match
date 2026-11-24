from dataclasses import asdict
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from author_match.profile_io import ProfileArtifact, load_profile_model, read_history, save_profile_model
from author_match.profile_features import FEATURE_COUNT, FEATURE_VERSION
from author_match.profile_probability import fit_account_model, score


class HistoryInputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)

    def put(self, name, text):
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_preserved_body_and_explicitly_grouped_histories(self):
        body = "  Waittt!! 😂\nYES?? "
        self.assertEqual(read_history(self.put("x.json", json.dumps([body, {"text": "next"}]))), [body, "next"])
        wrapper = {"handle": "writer", "posts": [body, {"text": "hi"}, {"author": "writer", "text": "there"}]}
        p = self.put("wrapper.json", json.dumps(wrapper))
        self.assertEqual(read_history(p), [body, "hi", "there"])
        self.assertEqual(read_history(p, "writer"), [body, "hi", "there"])
        with self.assertRaisesRegex(ValueError, "missing"):
            read_history(p, "unmatched")

    def test_mixed_jsonl_requires_selection(self):
        posts = [
            {"author": "A", "text": "post 1"}, {"handle": "B", "text": "post 2"},
            {"text": "unlabeled"}, {"user": "A", "text": "post 3"},
        ]
        path = self.dir / "mixed.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8-sig") as f:
            f.write("\n".join(json.dumps(row) for row in posts))
        with self.assertRaisesRegex(ValueError, "mixed handles"):
            read_history(path)
        self.assertEqual(read_history(path, "A"), ["post 1", "post 3"])
        self.assertEqual(read_history(path, "B"), ["post 2"])
        with self.assertRaisesRegex(ValueError, "missing"):
            read_history(path, "a")
        numeric = self.put("numeric.json", '[{"handle":42,"text":"hello"}]')
        self.assertEqual(read_history(numeric, "42"), ["hello"])
        path.write_bytes(path.read_bytes()[:-6])
        with self.assertRaisesRegex(ValueError, "compressed"):
            read_history(path, "A")

    def test_ambiguous_or_invalid_inputs_raise(self):
        cases = [
            {"handle": "x", "posts": [{"handle": "y", "text": "wrong"}]},
            [{"text": "unknown"}, {"handle": "known", "text": "known"}],
            [{"author": "a", "handle": "b", "text": "which account?"}],
            {"posts": "not an array"}, [{"text": None}], [3], {"text": "ok", "handle": []}, 42,
        ]
        for i, data in enumerate(cases):
            with self.subTest(data=data), self.assertRaises(ValueError):
                read_history(self.put(f"{i}.json", json.dumps(data)))
        with self.assertRaisesRegex(ValueError, "duplicate field"):
            read_history(self.put("two.json", '{"handle":"a","handle":"b","text":"x"}'))
        with self.assertRaisesRegex(ValueError, "line 2"):
            read_history(self.put("broken.jsonl", '{"text":"first"}\nnot json'))

    def test_empty_inputs_and_text_directory(self):
        for name, text in (("a.txt", "  "), ("b.json", ""), ("c.jsonl", ""), ("d.json", "[]")):
            self.assertEqual(read_history(self.put(name, text)), [])
        root = self.dir / "single_account"
        root.mkdir()
        self.assertEqual(read_history(root), [])
        (root / "b.txt").write_text("next", encoding="utf-8")
        (root / "a.txt").write_text("the first post", encoding="utf-8-sig")
        self.assertEqual(read_history(root), ["the first post", "next"])
        self.assertEqual(read_history(root / "b.txt"), ["next"])
        with self.assertRaisesRegex(ValueError, "missing"):
            read_history(root, "unknown")
        wrapped = self.put("empty.json", '{"handle":"a","posts":[]}')
        self.assertEqual(read_history(wrapped, "a"), [])
        with self.assertRaisesRegex(ValueError, "missing"):
            read_history(wrapped, "b")


class ProfileModelIOTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "account.json"
        x = [[v] + [0.0] * (FEATURE_COUNT - 1) for v in (0.2, 0.3, 0.8, 0.9)]
        model = fit_account_model(x, [0, 0, 1, 1], x, [0, 0, 1, 1])
        rows = [[2, 3, 90, 60, 0, 0, 1], [2, 4, 40, 55, 0, 2, 3],
                [3, 3, 52, 65, 1, 1, 1], [3, 4, 70, 81, 1, 2, 2]]
        self.artifact = ProfileArtifact(model, "linked_handles", {"rows": rows, "min_people": 6},
                                        {"author_match_version": "test", "feature_version": FEATURE_VERSION},
                                        ("a" * 32, "d" * 32))

    def test_plain_and_gzip_predictions_and_metadata_survive(self):
        for path in (self.path, self.path.with_suffix(".json.gz")):
            save_profile_model(path, self.artifact)
            loaded = load_profile_model(path)
            self.assertEqual(asdict(self.artifact), asdict(loaded))
            example = [0.8] + [0] * (FEATURE_COUNT - 1)
            self.assertEqual(score(self.artifact.model, example, 0.02), score(loaded.model, example, 0.02))
            self.assertEqual(loaded.support["rows"][0][:4], [2, 3, 90, 60])
            original = path.read_bytes()
            save_profile_model(path, loaded)
            self.assertEqual(original, path.read_bytes())

    def test_rejects_inconsistent_shape_counts_ids_metadata_and_schema(self):
        save_profile_model(self.path, self.artifact)
        raw = self.path.read_text()
        bad = [
            ("model", "coefficients", [1.0]),
            ("model", "intercept", float("inf")),
            ("support", "rows", [[2, 1, 2, 3, 1, 0, 0]]),
            ("support", "rows", [[2, 2, 50, 25, 1, 0, 0]]),
            ("support", "rows", [[2, 2, 2, 3, 1, 0, -1]]),
            ("support", "rows", [[2, 2, 2, 3, True, 0, 0]]),
            ("support", "min_people", False),
            ("provenance", "feature_version", FEATURE_VERSION + 1),
            ("provenance", "feature_version", True),
            (None, "provenance", []), (None, "fit_people", ["person-A"]),
            (None, "fit_people", ["b" * 32, "a" * 32]),
            (None, "label_basis", "guessed"), (None, "format", "author-match/cpu-v1"),
        ]
        for group, key, value in bad:
            payload = json.loads(raw)
            (payload if group is None else payload[group])[key] = value
            self.path.write_text(json.dumps(payload))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                load_profile_model(self.path)
        payload = json.loads(raw)
        for key in ("feature_means", "feature_scales", "coefficients"):
            payload["model"][key].pop()
        self.path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "width"):
            load_profile_model(self.path)

    def test_invalid_save_preserves_existing_checkpoint(self):
        save_profile_model(self.path, self.artifact)
        old = self.path.read_bytes()
        self.artifact.provenance["invalid"] = float("nan")
        with self.assertRaises(ValueError):
            save_profile_model(self.path, self.artifact)
        self.assertEqual(self.path.read_bytes(), old)
