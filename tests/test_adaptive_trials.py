from collections import Counter, defaultdict
from dataclasses import replace
from math import log1p
import unittest

from author_match.adaptive_trials import sample_adaptive_trials
from author_match.profile_trials import AccountPost
from author_match.text import word_count


def account_posts(owners=18, aliases=2, count=15):
    posts = []
    for owner in range(owners):
        for alias in range(aliases):
            for i in range(count):
                h = f"h{owner}a{alias}"
                identifier = f"{h}post{i}"
                topic = f"topic{i % 2}"
                words = " another" * (i % 6)
                text = f"sample {identifier} some writing today{words}"
                posts.append(AccountPost(identifier, str(owner), h, f"thread{i}", text, topic,
                                         f"window{owner // 6}"))
    return posts


class AdaptiveTrialsTest(unittest.TestCase):
    def test_matched_exposure_context_and_distinct_threads(self):
        posts = account_posts()
        rows = {p.text: p for p in posts}
        trials = sample_adaptive_trials(posts, label_basis="linked_handles", seed=13)
        self.assertEqual(trials, sample_adaptive_trials(reversed(posts), label_basis="linked_handles", seed=13))
        self.assertNotEqual(trials, sample_adaptive_trials(posts, label_basis="linked_handles", seed=3))
        groups = defaultdict(lambda: defaultdict(list))
        seen = set()
        uses = Counter()
        for t in trials:
            left, right = [[rows[s] for s in side] for side in (t.left, t.right)]
            uses.update({t.left_person, t.right_person})
            self.assertEqual({p.person for p in left}, {t.left_person})
            self.assertEqual({p.person for p in right}, {t.right_person})
            self.assertEqual(t.same, t.left_person == t.right_person)
            self.assertNotEqual(left[0].handle, right[0].handle)
            self.assertEqual({p.split_group for p in left + right}, {left[0].split_group})
            self.assertTrue({p.thread for p in left}.isdisjoint(p.thread for p in right))
            ids = frozenset(frozenset(p.id for p in side) for side in (left, right))
            self.assertNotIn(ids, seen)
            seen.add(ids)
            layout = (t.left, len(right), tuple(sorted(Counter(p.community for p in right).items())))
            groups[layout][t.same].append(log1p(sum(word_count(p.text) for p in right)))
        self.assertLessEqual(max(uses.values()), 18)
        self.assertGreater(len(groups), 20)
        for pair in groups.values():
            self.assertEqual(len(pair[False]), len(pair[True]))
            for pos, neg in zip(sorted(pair[True]), sorted(pair[False])):
                self.assertLessEqual(abs(pos - neg), log1p(.5) + 1e-12)
        self.assertTrue(any(min(len(t.left), len(t.right)) == 1 and
                            max(len(t.left), len(t.right)) >= 5 for t in trials))

    def test_genuinely_sparse_handles_and_short_text(self):
        posts = [replace(p, thread=None, text=f"hello {p.id}")
                 for p in account_posts(owners=12, count=1)]
        trials = sample_adaptive_trials(posts, label_basis="linked_handles")
        self.assertGreater(len(trials), 6)
        self.assertTrue(all(len(t.left) == len(t.right) == 1 for t in trials))
        self.assertEqual({t.left_person for t in trials if t.same}, {p.person for p in posts})

    def test_same_handle_requires_known_disjoint_threads(self):
        posts = account_posts(aliases=1)
        rows = {p.text: p for p in posts}
        trials = sample_adaptive_trials(posts, label_basis="account_proxy")
        self.assertTrue(any(t.same for t in trials))
        for t in trials:
            left, right = [[rows[s] for s in side] for side in (t.left, t.right)]
            self.assertEqual(left[0].handle == right[0].handle, t.same)
            self.assertTrue({p.thread for p in left}.isdisjoint(p.thread for p in right))
        absent = [replace(p, thread=None) for p in posts]
        self.assertEqual(sample_adaptive_trials(absent, label_basis="account_proxy"), [])

    def test_unmatchable_controls_and_invalid_basis(self):
        posts = account_posts(owners=2, count=5)
        separate = [replace(p, split_group=p.person) for p in posts]
        self.assertEqual(sample_adaptive_trials(separate, label_basis="linked_handles"), [])
        topics = [replace(p, community=p.person) for p in posts]
        self.assertEqual(sample_adaptive_trials(topics, label_basis="linked_handles"), [])
        lengthy = [replace(p, text=p.text + " extra" * (100 if p.person == "1" else 0)) for p in posts]
        self.assertEqual(sample_adaptive_trials(lengthy, label_basis="linked_handles"), [])
        with self.assertRaisesRegex(ValueError, "label_basis"):
            sample_adaptive_trials(posts, label_basis="unknown")
        with self.assertRaisesRegex(ValueError, "one observed handle"):
            sample_adaptive_trials(posts, label_basis="account_proxy")
