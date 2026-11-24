import pickle
import unittest
from dataclasses import dataclass

import numpy as np

from author_match.model import METHODS, Verifier
from author_match.text import clean_text, evidence_reason, mask_hashtags, word_count


@dataclass
class Post:
    id: str
    text: str


@dataclass
class Pair:
    left: str
    right: str
    same: bool


POSTS = [
    Post("a", "waittt,   really!! 😂 is that a thing here??"),
    Post("b", "yeahhh,   honestly!! 😂 when did that start??"),
    Post("c", "I typically prefer the original version. It was well written."),
    Post("d", "I would suggest the shorter route. It seemed quite direct."),
]
PAIRS = [
    Pair("a", "b", True), Pair("c", "d", True),
    Pair("a", "c", False), Pair("d", "b", False),
]


class TextTests(unittest.TestCase):
    def test_preserves_style_masks_metadata_and_drops_attributed_text(self):
        raw = (
            "> copy THIS!\n&gt; quoted line\n"
            "u/Somebody look at [this](https://example.org/secret)!!\r\n"
            "soooo   TRUE 😂  @other123.\n-- \nunique signature"
        )
        cleaned = "@ look at this!!\nsoooo   TRUE 😂  @."
        self.assertEqual(clean_text(raw), cleaned)
        self.assertEqual(clean_text(cleaned), cleaned)

    def test_normalization_is_idempotent(self):
        for raw in (
            "[> quoted](https://example.org)", "[https://first.net](https://second.net)",
            " `hi`  ", "&#x26;gt; nested quote", "hey &lt;blockquote&gt;borrowed&lt;/blockquote&gt;",
            "  www.example.org/hello!!", "a\n```python\nprint('code')\n```\nb",
            "@right. u/someone /r/SomeCommunity", "RT @somebody: writing by someone else",
        ):
            self.assertEqual(clean_text(clean_text(raw)), clean_text(raw))

    def test_quoted_repost_and_fenced_code(self):
        self.assertEqual(clean_text("RT @user: A lot of copied text"), "")
        self.assertEqual(clean_text("&gt; only a quote"), "")
        self.assertEqual(clean_text("hello\n```py\nsecret\n```\nworld"), "hello\nworld")
        self.assertEqual(clean_text("visit (https://example.com/path)."), "visit ().")

    def test_multilingual_surface_features(self):
        self.assertEqual(word_count("i’d say café 🌮!"), 3)
        self.assertEqual(mask_hashtags("hmm #Pizza #café"), "hmm # #")

    def test_abstention_is_symmetric(self):
        long_text = "The exact same sentence appears here. It also has another ending."
        for a, b, reason in [
            ("", long_text, "empty"),
            ("yeah same", long_text, "too_short"),
            ("😂😂😂", long_text, "too_short"),
            ("exact same sentence appears", long_text, "copied_overlap"),
            ("you really like this?", "YOU REALLY LIKE THIS!!", "copied_overlap"),
            ("no no no no no", long_text, "repeated_words"),
            ("1 2 3 4", long_text, "no_prose"),
            (POSTS[0].text, POSTS[1].text, None),
        ]:
            self.assertEqual(evidence_reason(a, b), reason)
            self.assertEqual(evidence_reason(b, a), reason)


class ModelTests(unittest.TestCase):
    def test_each_method_can_fit_score_reload_and_reverse(self):
        rev = [Pair(p.right, p.left, p.same) for p in PAIRS]
        lookup = {p.id: p.text for p in POSTS}
        for method in METHODS:
            with self.subTest(method=method):
                model = Verifier(method).fit(POSTS, PAIRS)
                result = model.scores(POSTS, PAIRS)
                self.assertTrue(np.all((0 <= result) & (result <= 1)))
                np.testing.assert_array_equal(result, model.scores(POSTS, rev))
                np.testing.assert_allclose(
                    result, [model.score_pair(lookup[p.left], lookup[p.right]) for p in PAIRS],
                    rtol=1e-7, atol=1e-8,
                )
                np.testing.assert_array_equal(result, pickle.loads(pickle.dumps(model)).scores(POSTS, PAIRS))
                self.assertEqual(model.scores([], []).shape, (0,))
                self.assertTrue(np.isfinite(model.score_pair("", "🔥")))

    def test_empty_training_vocabulary_is_a_valid_zero_similarity(self):
        for method in METHODS:
            model = Verifier(method).fit([Post(p.id, "") for p in POSTS], PAIRS)
            score = model.score_pair("a later much longer message", "another message")
            self.assertTrue(np.isfinite(score))
            if method != "style":
                self.assertEqual(score, 0)

    def test_word_model_uses_only_training_vocabulary(self):
        model = Verifier("word").fit(POSTS, PAIRS)
        self.assertEqual(model.score_pair("newword", "newword"), 0)
        self.assertGreater(model.score_pair(POSTS[0].text, POSTS[0].text), 0.99)

    def test_validation(self):
        with self.assertRaises(ValueError):
            Verifier("other")
        with self.assertRaisesRegex(ValueError, "fit"):
            Verifier().score_pair("one", "two")
        with self.assertRaisesRegex(ValueError, "at least one post"):
            Verifier().fit([], [])
        with self.assertRaisesRegex(ValueError, "same-author and different-author"):
            Verifier().fit(POSTS, PAIRS[:1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Verifier().fit([POSTS[0], POSTS[0]], PAIRS)
        model = Verifier(method="char").fit(POSTS, PAIRS)
        with self.assertRaisesRegex(ValueError, "unknown post"):
            model.scores(POSTS, [Pair("a", "missing", False)])
