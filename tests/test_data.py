import gzip
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from author_match.data import SPLITS, length_band, load_split, prepare
from author_match.text import word_count


def sample():
    rng = random.Random(100)
    vocab = [f"w{i}" for i in range(150)]
    result = []
    for i in range(72):
        for j in range(7):
            # Two shared conversations, five independent threads per writer.
            thread = f"shared-{i % 9}-{j}" if j < 2 else f"own-{i}-{j}"
            body = " ".join(rng.sample(vocab, 14 if j % 3 else 7))
            result.append(dict(id=f"p-{i}-{j}", speaker=f"a{i}", conversation_id=thread,
                               meta={"subreddit": f"c{j % 3}"}, text=body))
    return result


def write(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class DataTest(unittest.TestCase):
    def test_independence_matching_and_limits(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "utterances.jsonl"
            write(file, sample())
            folder = root / "pairs"
            manifest = prepare(file, folder, pairs_per_author=10)
            seen_authors, seen_threads = set(), set()
            for split in SPLITS:
                posts, pairs = load_split(folder, split)
                by_id = {p.id: p for p in posts}
                authors = {p.author for p in posts}
                threads = {p.thread for p in posts}
                self.assertTrue(authors.isdisjoint(seen_authors))
                self.assertTrue(threads.isdisjoint(seen_threads))
                seen_authors.update(authors)
                seen_threads.update(threads)
                self.assertEqual(len({(p.left, p.right) for p in pairs}), len(pairs))
                self.assertGreater(len(pairs), 0)
                self.assertEqual(sum(p.same for p in pairs) * 2, len(pairs))
                counts, bands = {}, {False: [], True: []}
                for p in pairs:
                    a, b = by_id[p.left], by_id[p.right]
                    self.assertEqual(p.same, a.author == b.author)
                    if p.same:
                        self.assertNotEqual(a.thread, b.thread)
                    for name in {a.author, b.author}:
                        counts[name] = counts.get(name, 0) + 1
                    bands[p.same].append(length_band(min(word_count(a.text), word_count(b.text))))
                self.assertEqual(sorted(bands[True]), sorted(bands[False]))
                self.assertLessEqual(max(counts.values()), 10)
            again = prepare(file, folder, pairs_per_author=10)
            self.assertEqual(again, manifest)
            with self.assertRaisesRegex(ValueError, "different or incomplete"):
                prepare(file, folder, pairs_per_author=9)
            with self.assertRaisesRegex(ValueError, "split must be"):
                load_split(folder, "../oops")

    def test_source_order_does_not_change_records(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "source.jsonl"
            rows = sample()
            write(path, rows)
            one = prepare(path, root / "one")
            write(path, reversed(rows))
            two = prepare(path, root / "two")
            self.assertEqual(one["file_sha256"], two["file_sha256"])

    def test_duplicates_identity_and_gzip_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "utterances.jsonl.gz"
            long = " ".join(f"token{n}" for n in range(50))
            rows = [dict(id=str(i), author=author, thread=str(i), text=body)
                    for i, (author, body) in enumerate([
                        ("Alice", long), ("Bob", long.replace("token20", "slightlydifferent")),
                        ("Alice", "my OWN comment"), ("ALICE", "My own COMMENT"),
                        ("[deleted]", "can't be used"), ("AutoModerator", "admin's automated reply"),
                        ("junk", "[removed]"), ("person", "> only quoted words"),
                        ("org", "formal group message"), ("other", "totally different"),
                    ])]
            with gzip.open(file, "wt", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            (root / "conversations.json").write_text(json.dumps({
                str(i): {"meta": {"subreddit": "topic"}} for i in range(len(rows))
            }))
            folder = root / "out"
            manifest = prepare(root, folder, excluded_accounts=["ORG"])
            posts = [p for split in SPLITS for p in load_split(folder, split)[0]]
            self.assertEqual({p.id for p in posts}, {"2", "9"})
            self.assertEqual({p.community for p in posts}, {"topic"})
            self.assertEqual(manifest["counts"]["conflicting_copy_clusters"], 1)
            self.assertEqual(manifest["counts"]["near_copy_rows"], 1)

    def test_short_copies_use_the_comparison_word_sequence(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            messages = [
                ("a", "the patch finally fixed it for me!"),
                ("b", "THE PATCH FINALLY FIXED IT FOR ME..."),
                ("c", "I can’t make that today!!"),
                ("c", "i CAN'T make that today"),
                ("d", "yeah!"), ("e", "yeah?"), ("f", "🙂"), ("g", "😂"),
            ]
            source = root / "posts.jsonl"
            write(source, [dict(id=str(i), author=author, thread=str(i), text=text)
                           for i, (author, text) in enumerate(messages)])
            out = root / "external"
            manifest = prepare(source, out, external=True)
            posts, _ = load_split(out, "external")
            self.assertEqual(manifest["settings"]["format_version"], 2)
            self.assertEqual(manifest["counts"]["conflicting_copy_clusters"], 1)
            self.assertEqual({post.text for post in posts}, {
                "I can’t make that today!!", "yeah!", "yeah?", "🙂", "😂",
            })

    def test_community_holdout(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "posts.jsonl"
            write(file, sample())
            out = root / "data"
            manifest = prepare(file, out, holdout_communities=("C1",))
            self.assertEqual(manifest["settings"]["holdout_communities"], ["c1"])
            train = [p for s in SPLITS[:3] for p in load_split(out, s)[0]]
            test, pairs = load_split(out, "test")
            self.assertTrue(train and test and pairs)
            self.assertTrue({p.author for p in test}.isdisjoint(p.author for p in train))
            self.assertTrue({p.thread for p in test}.isdisjoint(p.thread for p in train))
            self.assertEqual({p.community for p in test}, {"c1"})
            self.assertNotIn("c1", {p.community for p in train})
            with self.assertRaisesRegex(ValueError, "missing"):
                prepare(file, root / "missing", holdout_communities=("missing",))

    def test_external_data_is_one_evaluation_only_corpus(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, out = root / "posts.jsonl", root / "external"
            rows = sample()
            write(source, rows + [rows[0]])
            m = prepare(source, out, external=True)
            self.assertIs(m["settings"]["external"], True)
            self.assertEqual(set(m["splits"]), {"external"})
            self.assertEqual({p.name for p in out.iterdir()}, {
                "external.posts.jsonl", "external.pairs.jsonl", "manifest.json",
            })
            posts, pairs = load_split(out, "external")
            self.assertEqual({p.id for p in posts}, {r["id"] for r in rows})
            self.assertTrue(pairs)
            self.assertEqual(len(pairs), 2 * sum(p.same for p in pairs))
            self.assertEqual(m, prepare(source, out, external=True))
            with self.assertRaisesRegex(ValueError, "does not contain"):
                load_split(out, "train")
            with self.assertRaisesRegex(ValueError, "different or incomplete"):
                prepare(source, out)
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                prepare(source, root / "invalid", external=True, holdout_communities=["c0"])
            (out / "external.pairs.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_split(out, "external")
            normal = prepare(source, root / "regular")
            self.assertNotIn("external", normal["settings"])
            with self.assertRaisesRegex(ValueError, "does not contain"):
                load_split(root / "regular", "external")

    def test_error_and_corruption_do_not_replace_existing_dataset(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "posts.jsonl"
            rows = sample()
            write(source, [rows[0], {**rows[0], "text": "conflicting"}])
            with self.assertRaisesRegex(ValueError, "conflicting records"):
                prepare(source, root / "dataset")
            self.assertFalse((root / "dataset").exists())
            write(source, rows)
            prepare(source, root / "dataset")
            artifact = root / "dataset" / "train.pairs.jsonl"
            artifact.write_text("damaged")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_split(root / "dataset", "train")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                prepare(source, root / "dataset")
            self.assertEqual(artifact.read_text(), "damaged")
