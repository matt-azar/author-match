from collections import Counter, defaultdict
from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from author_match.text import word_count

from author_match.profile_trials import (
    AccountPost, partition_people, proxy_posts, read_linked_posts, sample_trials,
)
from author_match.data import Post


def corpus(people=14, handles=2, posts_per_handle=19, missing_thread=False):
    rows = []
    for person in range(people):
        for handle in range(handles):
            name = f"{person}_{handle}"
            for n in range(posts_per_handle):
                pid = f"row-{person}-{handle}-{n}"
                thread = None if missing_thread else f"thread-{n % 9}"
                rows.append(AccountPost(pid, str(person), name, thread, f"entry {pid} first{n} word{person} note{handle} today{n}"))
    return rows


class ProfileTrialsTest(unittest.TestCase):
    def check_pairs(self, rows, basis):
        trials = sample_trials(rows, label_basis=basis)
        self.assertEqual(trials, sample_trials(reversed(rows), label_basis=basis))
        self.assertGreater(len(trials), 4)
        by_text = {p.text: p for p in rows}
        layouts = defaultdict(Counter)
        use = Counter()
        seen = set()
        for t in trials:
            left, right = [[by_text[x] for x in side] for side in (t.left, t.right)]
            for side, owner in ((left, t.left_person), (right, t.right_person)):
                self.assertGreaterEqual(len(side), 2)
                self.assertGreaterEqual(sum(word_count(p.text) for p in side), 15)
                self.assertEqual({p.person for p in side}, {owner})
                self.assertEqual(len({p.handle for p in side}), 1)
            self.assertEqual(t.left_person == t.right_person, t.same)
            if t.same:
                self.assertEqual(left[0].handle == right[0].handle, basis == "account_proxy")
            self.assertTrue({p.id for p in left}.isdisjoint(p.id for p in right))
            self.assertTrue({p.thread for p in left if p.thread}.isdisjoint(p.thread for p in right if p.thread))
            key = frozenset((frozenset(t.left), frozenset(t.right)))
            self.assertNotIn(key, seen)
            seen.add(key)
            layouts[t.same][(t.left_person, t.left, len(t.right))] += 1
            use.update({t.left_person, t.right_person})
        self.assertEqual(layouts[True], layouts[False])
        self.assertLessEqual(max(use.values()), 18)
        return trials

    def test_proxy_trials_match_exposure_and_handle_unknown_threads(self):
        rows = corpus(handles=1)
        rows.append(AccountPost("lost", "0", "0_0", None, "unknown source thread"))
        trials = self.check_pairs(rows, "account_proxy")
        self.assertFalse(any("unknown source thread" in side for t in trials for side in (t.left, t.right)))
        self.assertTrue(any(len(t.left) != len(t.right) for t in trials))
        self.assertEqual(sample_trials(corpus(handles=1, missing_thread=True), label_basis="account_proxy"), [])

    def test_verified_aliases_and_full_history_sizes(self):
        rows = corpus(posts_per_handle=123, missing_thread=True)
        trials = self.check_pairs(rows, "linked_handles")
        sizes = {(len(t.left), len(t.right)) for t in trials}
        self.assertIn((2, 2), sizes)
        self.assertIn((123, 123), sizes)
        self.assertIn((2, 123), sizes)
        self.assertIn((123, 2), sizes)

    def test_linked_handles_with_shared_threads(self):
        self.check_pairs(corpus(), "linked_handles")
        self.assertEqual(sample_trials(corpus(handles=1), label_basis="linked_handles"), [])
        with self.assertRaisesRegex(ValueError, "one observed handle"):
            sample_trials(corpus(), label_basis="account_proxy")

    def test_disjoint_splits_across_aliases(self):
        rows = corpus(people=40, posts_per_handle=3)
        split = partition_people(rows, seed=5)
        self.assertEqual(split, partition_people(reversed(rows), seed=5))
        seen = set()
        for name in ("train", "dev", "calibration", "test"):
            people = {p.person for p in split[name]}
            self.assertTrue(seen.isdisjoint(people))
            self.assertGreaterEqual(len(people), 2)
            seen.update(people)
        self.assertEqual(len(seen), 40)

    def test_import_validates_ids_drops_ambiguous_text_and_preserves_newlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "verified.jsonl.gz"
            def write(rows):
                with gzip.open(file, "wt", encoding="utf-8") as out:
                    for row in rows:
                        out.write(json.dumps(row) + "\n")
            a = dict(person_id="1", handle="One", text="first line\nSecond line!")
            b = dict(person_id="1", handle="Two", text="FIRST LINE second line...")
            c = dict(person_id="2", handle="Three", text="different body")
            write([a, a, b, c, {**c, "text": "> quoted only"}])
            clean = read_linked_posts(file)
            self.assertEqual([p.text for p in clean], ["different body"])
            self.assertIsNone(clean[0].thread)
            self.assertTrue(clean[0].id.startswith("derived:"))
            write([a, a])
            self.assertEqual([p.text for p in read_linked_posts(file)], [a["text"]])
            write([a, {**a, "person_id": "other"}])
            with self.assertRaisesRegex(ValueError, "multiple people"):
                read_linked_posts(file)
            write([{**a, "id": "shared"}, {**c, "id": "shared"}])
            with self.assertRaisesRegex(ValueError, "conflicting records"):
                read_linked_posts(file)
            write([dict(handle="one", text="missing label")])
            with self.assertRaisesRegex(ValueError, r":1: person_id"):
                read_linked_posts(file)

    def test_sparse_calibration_histories_and_prose_minimum(self):
        rows = corpus(people=6, posts_per_handle=3, missing_thread=True)
        trials = self.check_pairs(rows, "linked_handles")
        self.assertEqual({t.left_person for t in trials if t.same}, {str(i) for i in range(6)})
        thin = [AccountPost(p.id, p.person, p.handle, None, "w" + p.id.replace("-", ""))
                for p in rows]
        self.assertEqual(sample_trials(thin, label_basis="linked_handles"), [])

    def test_known_reference_community_is_balanced_when_available(self):
        rows = [AccountPost(p.id, p.person, p.handle, p.thread, p.text,
                            "topic" + str(int(p.person) % 2))
                for p in corpus(people=8, posts_per_handle=5, missing_thread=True)]
        trials = sample_trials(rows, label_basis="linked_handles")
        ref = {p.text: p.community for p in rows}
        self.assertGreater(len(trials), 0)
        positives = Counter((p.left, len(p.right), ref[p.right[0]]) for p in trials if p.same)
        negatives = Counter((p.left, len(p.right), ref[p.right[0]]) for p in trials if not p.same)
        self.assertGreater(sum((positives & negatives).values()), .8 * sum(positives.values()))

    def test_transfer_windows_stay_intact_within_window_controls(self):
        # Each window has one linked writer and several single-handle controls.
        rows = [replace(p, split_group="day" + str(int(p.person) // 3),
                        community="community" + str(int(p.person) // 3))
                for p in corpus(people=30, posts_per_handle=6, missing_thread=True)
                if int(p.person) % 3 == 0 or p.handle.endswith("_0")]
        splits = partition_people(rows, transfer=True)
        self.assertEqual(splits, partition_people(reversed(rows), transfer=True))
        self.assertFalse(splits["train"] or splits["dev"])
        phases = {s: {p.split_group for p in pp} for s, pp in splits.items()}
        self.assertEqual(len(phases["calibration"]), 6)
        self.assertEqual(len(phases["test"]), 4)
        self.assertTrue(phases["calibration"].isdisjoint(phases["test"]))
        for split in ("calibration", "test"):
            groups = {p.person: p.split_group for p in splits[split]}
            trials = self.check_pairs(splits[split], "linked_handles")
            for t in trials:
                self.assertEqual(groups[t.left_person], groups[t.right_person])
            self.assertEqual({t.left_person for t in trials if t.same},
                             {p.person for p in splits[split] if int(p.person) % 3 == 0})
        with self.assertRaisesRegex(ValueError, "incompatible split groups"):
            partition_people(rows + [replace(rows[0], id="additional", split_group="other")])
        with self.assertRaisesRegex(ValueError, "two independent windows"):
            partition_people([p for p in rows if p.split_group == "day0"], transfer=True)

    def test_proxy_adapter_and_basis_validation(self):
        rows = proxy_posts([Post("id", "user", "thread", "community", "source message")])
        self.assertEqual(rows, [AccountPost("id", "user", "user", "thread", "source message", "community")])
        with self.assertRaisesRegex(ValueError, "label_basis"):
            sample_trials(rows, label_basis="unverified")
