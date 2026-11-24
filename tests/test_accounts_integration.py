"""End-to-end account tests using entirely invented people, aliases, and writing."""

import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from author_match.accounts import _Support
from author_match.cli import main


_VOCABULARY = (
    "there still really some quite little almost then when more down outside "
    "market red house lunch walk blue road bus table green book read turn "
    "paper think hours friends could should never just hill round next"
).split()


def fictional_history(writer, alias, count=4):
    # Style habits persist across aliases; topic words are sampled afresh.
    rng = random.Random((writer + 1) * 11 + alias)
    style = writer % 4
    posts = []
    for _ in range(count):
        words = rng.choices(_VOCABULARY, k=13)
        if style == 0:
            text = "idk, " + " ".join(words) + " lol!!"
        elif style == 1:
            text = " ".join(words).upper() + "??"
        elif style == 2:
            text = "Perhaps " + " ".join(words[:6]) + ". " + " ".join(words[6:]).capitalize() + "."
        else:
            text = " ".join(words[:6]) + "\n\n" + " ".join(words[6:]) + " 🙃"
        posts.append(text)
    return posts


def call(*args):
    out, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(errors):
        status = main([str(a) for a in args])
    return status, out.getvalue(), errors.getvalue()


def run(*args):
    status, out, errors = call(*args)
    if status:
        raise AssertionError(errors)
    return json.loads(out)


class AccountCommandsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.base = Path(tmp.name)
        cls.corpus = cls.base / "invented.jsonl"
        cls.model = cls.base / "linked.json.gz"
        with cls.corpus.open("w", encoding="utf-8") as f:
            for writer in range(96):
                for alias in range(2):
                    handle = f"alias-{writer}-{alias}"
                    for idx, text in enumerate(fictional_history(writer, alias)):
                        pid = f"{handle}/{idx}"
                        row = dict(id=pid, text=text, handle=handle, person_id=f"person-{writer}",
                                   author=f"person-{writer}", thread=pid, community="invented",
                                   split_group=f"window-{writer // 8}")
                        f.write(json.dumps(row) + "\n")
        run("train-accounts", cls.corpus, cls.model, "--description", "invented aliases for a test")

    def compare(self, left, right, prior=0.5, model=None):
        a, b = self.base / "left.json", self.base / "right.json"
        a.write_text(json.dumps(left), encoding="utf-8")
        b.write_text(json.dumps(right), encoding="utf-8")
        return run("compare-accounts", a, b, "--model", model or self.model, "--prior", prior)

    def test_compare_uses_writing_and_explicit_prior(self):
        a = fictional_history(128, 0, 2)
        b = fictional_history(128, 1)
        c = fictional_history(129, 0)
        same = self.compare(a, b)
        different = self.compare(a, c)
        self.assertEqual(same["status"], "estimated")
        self.assertEqual(different["status"], "estimated")
        self.assertEqual(same["label_basis"], "linked_handles")
        self.assertEqual(same["link_evidence"], "provided_owner_labels")
        self.assertIsNone(same["same_account_proxy_probability"])
        self.assertGreater(same["estimated_same_person_probability"], different["estimated_same_person_probability"])
        self.assertEqual([side["used_posts"] for side in same["histories"]], [2, 4])

        reversed_inputs = self.compare(b[::-1], a[::-1])
        self.assertEqual(reversed_inputs["log_likelihood_ratio"], same["log_likelihood_ratio"])
        self.assertEqual(reversed_inputs["histories"], same["histories"][::-1])
        prior = 0.01
        rarer = self.compare(a, b, prior)
        self.assertEqual(rarer["log_likelihood_ratio"], same["log_likelihood_ratio"])
        odds = prior / (1 - prior) * same["likelihood_ratio"]
        self.assertAlmostEqual(rarer["estimated_same_person_probability"], odds / (1 + odds))

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["compare-accounts", "a", "b", "--model", str(self.model)])

    def test_duplicate_and_shared_writing_supplies_no_additional_evidence(self):
        a, b = fictional_history(128, 0), fictional_history(129, 0)
        shared = "Please remember the meeting moved to Monday morning by popular request."
        original = self.compare(a, b)
        repeated = self.compare(a * 2 + [shared], b + [shared])
        self.assertEqual(original["status"], "estimated")
        self.assertEqual(repeated["log_likelihood_ratio"], original["log_likelihood_ratio"])
        self.assertEqual([h["used_posts"] for h in repeated["histories"]], [4, 4])
        for left, right in [(a, a), ([], b), (["yes"] * 40, b)]:
            denied = self.compare(left, right)
            self.assertEqual(denied["status"], "inconclusive")
            self.assertIsNone(denied["estimated_same_person_probability"])
            self.assertIsNone(denied["likelihood_ratio"])

    def test_no_extrapolation_to_unsupported_writing_volume(self):
        rng = random.Random(9)
        long_posts = [" ".join(rng.choices(_VOCABULARY, k=700)) for _ in range(2)]
        result = self.compare(long_posts, fictional_history(132, 1, 2))
        self.assertEqual(result["reason"], "unsupported_history_sizes")
        self.assertIsNone(result["estimated_same_person_probability"])
        self.assertEqual([h["used_posts"] for h in result["histories"]], [0, 0])

    def test_evaluate_reload_and_reject_external_identity_overlap(self):
        report = self.base / "eval.json"
        result = run("evaluate-accounts", self.corpus, self.model, "--split", "test", "--prior", "0.2",
                     "--bootstrap-samples", 0, "--output", report)
        self.assertEqual(json.loads(report.read_text(encoding="utf-8")), result)
        self.assertEqual(result["evaluation_label_basis"], "linked_handles")
        self.assertGreater(result["returned_estimates"], 0)
        scored = result["all_pairs_using_prior_when_inconclusive"]
        self.assertGreater(min(scored["same"], scored["different"]), 0)
        self.assertAlmostEqual(scored["prior_only"]["brier"], 0.16)
        status, _, error = call("evaluate-accounts", self.corpus, self.model, "--split", "external")
        self.assertEqual(status, 2)
        self.assertIn("writers or handles", error)

    def test_proxy_training_keeps_person_probability_unavailable(self):
        prepared, proxy = self.base / "prepared", self.base / "proxy.json"
        run("prepare", self.corpus, prepared)
        run("train-accounts", prepared, proxy)
        self.assertTrue(proxy.read_bytes().startswith(b"{"))
        value = self.compare(fictional_history(128, 0), fictional_history(128, 1), model=proxy)
        self.assertEqual(value["label_basis"], "account_proxy")
        self.assertEqual(value["link_evidence"], "split_account_history")
        self.assertEqual(value["status"], "estimated")
        self.assertIsNotNone(value["same_account_proxy_probability"])
        self.assertIsNone(value["estimated_same_person_probability"])

    def test_comparable_histories_keep_word_totals_paired_with_post_counts(self):
        rows = [[2, 20, 1000, 50, same, i, i if same else i + 1]
                for same in (0, 1) for i in range(8)]
        support = _Support({"min_people": 6, "rows": rows})
        def history(n, words):
            return SimpleNamespace(posts=n, words=words)
        expected = support.counts(history(2, 1000), history(20, 50))
        self.assertGreaterEqual(min(expected.values()), 6)
        self.assertEqual(expected, support.counts(history(20, 50), history(2, 1000)))
        self.assertEqual(max(support.counts(history(2, 50), history(20, 1000)).values()), 0)
