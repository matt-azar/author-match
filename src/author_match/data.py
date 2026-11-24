"""Prepare independent author-verification trials from individual social posts."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .text import MIN_WORDS, clean_text, comparison_words, word_count


SPLITS = ("train", "dev", "calibration", "test")
SPLIT_WEIGHTS = (0.55, 0.15, 0.15, 0.15)
UNKNOWN = ""
FORMAT_VERSION = 2
_BAD_ACCOUNTS = {"[deleted]", "[removed]", "deleted", "removed", "automoderator", "reddit"}


@dataclass(frozen=True, slots=True)
class Post:
    id: str
    author: str
    thread: str
    community: str
    text: str
    timestamp: float | None = None


@dataclass(frozen=True, slots=True)
class Pair:
    left: str
    right: str
    same: bool
    condition: str


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _objects(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
            yield line_no, row


def _community_map(file):
    if not file.is_file():
        return {}
    with file.open(encoding="utf-8") as stream:
        conversations = json.load(stream)
    return {
        thread: entry.get("meta", {}).get("subreddit") or entry.get("subreddit")
        for thread, entry in conversations.items()
        if isinstance(entry, dict)
    }


def _read_posts(path, sidecar, excluded_accounts):
    rows = {}
    counts = Counter()
    communities = _community_map(sidecar) if sidecar else {}
    fallback = path.parent.name.removeprefix("subreddit-") if path.parent.name.startswith("subreddit-") else ""
    for line, row in _objects(path):
        counts["input_rows"] += 1
        account = row.get("author", row.get("speaker", row.get("user")))
        if isinstance(account, dict):
            account = account.get("id", account.get("name"))
        if not isinstance(account, (str, int)) or not str(account).strip():
            counts["missing_account"] += 1
            continue
        author = str(account).strip()
        key = author.casefold()
        if key in excluded_accounts or key.endswith(("_bot", "-bot")) or row.get("is_bot") is True:
            counts["excluded_account"] += 1
            continue
        thread = row.get("thread") or row.get("conversation_id") or row.get("root")
        post_id = row.get("id")
        if post_id is None or thread is None or not str(thread).strip() or not str(post_id).strip():
            raise ValueError(f"{path.name}:{line}: each post needs its own id and a thread or conversation_id")
        body = row.get("text")
        if not isinstance(body, str):
            raise ValueError(f"{path.name}:{line}: text must be a string")
        if body.strip().casefold() in {"[removed]", "[deleted]"}:
            counts["removed_text"] += 1
            continue
        body = clean_text(body)
        if not body.strip():
            counts["empty_after_cleaning"] += 1
            continue
        meta = row.get("meta") or {}
        if not isinstance(meta, dict):
            raise ValueError(f"{path.name}:{line}: meta must be an object")
        subreddit = (row.get("community") or row.get("subreddit") or meta.get("subreddit")
                     or meta.get("community") or communities.get(str(thread)) or fallback or UNKNOWN)
        if (row.get("platform") == "reddit" or row.get("subreddit") or meta.get("subreddit")
                or communities.get(str(thread)) or fallback):
            author = key
            subreddit = str(subreddit).casefold()
        stamp = row.get("timestamp")
        if stamp is not None:
            try:
                stamp = float(stamp)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path.name}:{line}: expected a numeric timestamp") from exc
            if not float("-inf") < stamp < float("inf"):
                raise ValueError(f"{path.name}:{line}: expected a finite timestamp")
        post = Post(str(post_id), author, str(thread), str(subreddit), body, stamp)
        if post.id in rows:
            if rows[post.id] != post:
                raise ValueError(f"{path.name}:{line}: conflicting records for post ID {post.id!r}")
            counts["repeated_post_id"] += 1
            continue
        rows[post.id] = post
    return [rows[k] for k in sorted(rows)], counts


def _deduplicate(posts, counts):
    """Collapse canonical copies; discard clusters that have conflicting authors.

    At the comparison word minimum, group identical word sequences across
    casing, spacing, and punctuation variations. Keep punctuation and emoji
    in the canonical key for shorter replies. At 12+ words, four-word anchors shortlist near copies for a trigram
    Jaccard check. This heuristic may miss paraphrases, shorter edits, and
    copies whose anchors all changed or occur above the bucket frequency cap.
    """
    canonical = {}
    anchors = defaultdict(list)
    ambiguous = set()
    retained = []
    for post in posts:
        words = comparison_words(post.text)
        normal = " ".join(words) if len(words) >= MIN_WORDS else " ".join(post.text.casefold().split())
        digest = hashlib.blake2b(normal.encode(), digest_size=16).digest()
        target = canonical.get(digest)
        if target is None:
            keys = []
            if len(words) >= 12:
                grams = list(zip(words, words[1:], words[2:], words[3:]))
                # Lexicographic extremes provide position-independent anchors.
                keys = sorted(set(grams))
                keys = keys[:2] + keys[-2:]
                candidates = sorted({i for k in keys for i in anchors.get(k, ())
                                     if len(anchors[k]) <= 32})
                current = None
                for i in candidates:
                    previous_words = comparison_words(retained[i].text)
                    if min(len(words), len(previous_words)) < 0.8 * max(len(words), len(previous_words)):
                        continue
                    if current is None:
                        current = set(zip(words, words[1:], words[2:]))
                    previous = set(zip(previous_words, previous_words[1:], previous_words[2:]))
                    if len(current & previous) >= 0.82 * len(current | previous):
                        target = i
                        break
            if target is None:
                target = len(retained)
                retained.append(post)
                for k in set(keys):
                    bucket = anchors[k]
                    if len(bucket) <= 32:
                        bucket.append(target)
                canonical[digest] = target
                continue
            counts["near_copy_rows"] += 1
            canonical[digest] = target
        else:
            counts["exact_copy_rows"] += 1
        if post.author != retained[target].author:
            ambiguous.add(target)
    counts["conflicting_copy_clusters"] = len(ambiguous)
    return [p for i, p in enumerate(retained) if i not in ambiguous]


def _split_posts(posts, rng, names=SPLITS, weights=SPLIT_WEIGHTS):
    by_author = defaultdict(list)
    for post in posts:
        by_author[post.author].append(post)
    # Two-thread authors can generate positives. Single-thread authors can
    # follow their thread later without introducing any author leakage.
    active = [a for a, pp in by_author.items() if len({p.thread for p in pp}) > 1]
    rng.shuffle(active)
    owners = {}
    start = 0
    for i, name in enumerate(names):
        stop = round(sum(weights[:i + 1]) * len(active))
        owners.update(dict.fromkeys(active[start:stop], name))
        start = stop
    votes = defaultdict(Counter)
    for post in posts:
        if post.author in owners:
            votes[post.thread][owners[post.author]] += 1
    thread_owners = {}
    for thread in sorted({p.thread for p in posts}):
        candidates = votes.get(thread)
        if candidates:
            highest = max(candidates.values())
            thread_owners[thread] = rng.choice([s for s in names if candidates[s] == highest])
        else:
            thread_owners[thread] = rng.choices(names, weights=weights)[0]
    splits = {name: [] for name in names}
    for p in posts:
        split = thread_owners[p.thread]
        if owners.get(p.author, split) == split:
            splits[split].append(p)
    return splits


def _partition(posts, rng, held_communities, counts):
    if not held_communities:
        return _split_posts(posts, rng)
    # Separate experiment: all test posts come from reserved communities.
    held = [p for p in posts if p.community.casefold() in held_communities]
    found = {p.community.casefold() for p in held}
    if found != held_communities:
        missing = ", ".join(sorted(held_communities - found))
        raise ValueError(f"no usable posts for requested holdout communities: {missing}")
    other = [p for p in posts if p.community.casefold() not in held_communities]
    held_authors = {p.author for p in held}
    other_authors = {p.author for p in other}
    shared = sorted(held_authors & other_authors)
    rng.shuffle(shared)
    # Keep writers represented only in the new communities and reserve a
    # seeded subset of cross-community writers. Their other work is dropped.
    test_authors = (held_authors - other_authors) | set(shared[:max(1, round(0.15 * len(shared)))])
    test = [p for p in held if p.author in test_authors]
    test_threads = {p.thread for p in test}
    core = [p for p in other if p.author not in test_authors and p.thread not in test_threads]
    counts["dropped_community_holdout_conflicts"] = len(posts) - len(core) - len(test)
    core_weights = tuple(w / sum(SPLIT_WEIGHTS[:3]) for w in SPLIT_WEIGHTS[:3])
    splits = _split_posts(core, rng, SPLITS[:3], core_weights)
    splits["test"] = test
    return splits


def length_band(length):
    if length < 10:
        return "<10"
    if length < 25:
        return "10-24"
    if length < 50:
        return "25-49"
    if length < 100:
        return "50-99"
    return "100+"


def _condition(a, b):
    if a.thread == b.thread:
        return "same_thread"
    if not a.community or not b.community:
        return "unknown_community"
    return "same_community" if a.community == b.community else "cross_community"


def _sample_pairs(posts, rng, cap):
    by_author = defaultdict(list)
    by_community = defaultdict(list)
    by_thread = defaultdict(list)
    bands = {p.id: length_band(word_count(p.text)) for p in posts}
    for p in posts:
        by_author[p.author].append(p)
        by_community[p.community, bands[p.id]].append(p)
        by_thread[p.thread, bands[p.id]].append(p)
    active = [a for a, pp in by_author.items() if len({p.thread for p in pp}) > 1]
    rng.shuffle(active)
    trials = []
    usage = Counter()
    seen = set()

    def pair_key(a, b):
        return (a.id, b.id) if a.id < b.id else (b.id, a.id)

    def draw(author, target_cross, try_thread):
        pool = by_author[author]
        fallback = None
        # Sample bounded proposals so high-volume accounts don't form quadratic pools.
        for _ in range(12):
            a, b = rng.sample(pool, 2)
            pos_key = pair_key(a, b)
            if a.thread == b.thread or pos_key in seen:
                continue
            if fallback is None:
                fallback = a, b, pos_key
            if (_condition(a, b) == "cross_community") == target_cross:
                fallback = a, b, pos_key
                break
        if fallback is None:
            return False
        a, b, pos_key = fallback
        # Holding the anchor fixed, replace the positive partner with a
        # different writer from its community and length band.
        pools = []
        if try_thread and a.community == b.community:
            pools.append(by_thread.get((a.thread, bands[b.id]), ()))
        pools.append(by_community[b.community, bands[b.id]])
        for neg_pool in pools:
            for c in rng.sample(neg_pool, min(24, len(neg_pool))):
                neg_key = pair_key(a, c)
                if c.author == author or usage[c.author] >= cap or neg_key in seen:
                    continue
                trials.extend((Pair(*pos_key, True, _condition(a, b)),
                               Pair(*neg_key, False, _condition(a, c))))
                seen.update((pos_key, neg_key))
                usage[author] += 2
                usage[c.author] += 1
                return True
        return False

    for turn in range(cap // 2):
        for i, author in enumerate(active):
            if usage[author] <= cap - 2:
                draw(author, (turn + i) % 2 == 0, (turn + i) % 3 == 0)
    rng.shuffle(trials)
    return trials


def _write_records(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _audit(posts, pairs):
    authors = {p.id: p.author for p in posts}
    lengths = {p.id: word_count(p.text) for p in posts}
    appearances = Counter()
    for pair in pairs:
        appearances.update({authors[pair.left], authors[pair.right]})
    return {
        "posts": len(posts),
        "authors": len(set(authors.values())),
        "threads": len({p.thread for p in posts}),
        "known_communities": len({p.community for p in posts} - {UNKNOWN}),
        "pairs": len(pairs),
        "paired_authors": len(appearances),
        "max_pairs_per_author": max(appearances.values(), default=0),
        "positive_pairs": sum(p.same for p in pairs),
        "negative_pairs": sum(not p.same for p in pairs),
        "conditions": dict(Counter(("same:" if p.same else "different:") + p.condition for p in pairs)),
        "length_bands": dict(Counter(length_band(min(lengths[p.left], lengths[p.right])) for p in pairs)),
    }


def prepare(input_path: Path, output_dir: Path, *, seed: int = 42, pairs_per_author: int = 12,
            excluded_accounts=(), holdout_communities=(), external: bool = False) -> dict:
    """Prepare an immutable training benchmark or a separate external evaluation corpus.

    Set external=True only for an independently sourced corpus that was held
    out of model development. Preparation cannot infer cross-corpus identity.
    """
    if pairs_per_author < 2:
        raise ValueError("pairs_per_author must be at least 2")
    path = Path(input_path)
    if path.is_dir():
        path = path / "utterances.jsonl"
        if not path.is_file() and Path(str(path) + ".gz").is_file():
            path = Path(str(path) + ".gz")
    sidecar = path.parent / "conversations.json" if path.name.startswith("utterances.jsonl") else None
    if sidecar and not sidecar.is_file():
        sidecar = None
    fingerprint = {
        "format_version": FORMAT_VERSION, "source_sha256": _digest(path), "seed": seed,
        "conversation_metadata_sha256": _digest(sidecar) if sidecar else None,
        "source_community_fallback": path.parent.name.removeprefix("subreddit-")
        if path.parent.name.startswith("subreddit-") else None,
        "pairs_per_author": pairs_per_author,
        "holdout_communities": sorted({str(c).strip().casefold() for c in holdout_communities}),
        "excluded_accounts": sorted({str(a).casefold() for a in excluded_accounts}),
    }
    split_names = SPLITS
    if external:
        if fingerprint["holdout_communities"]:
            raise ValueError("external evaluation cannot be combined with a community holdout")
        fingerprint["external"] = True
        split_names = ("external",)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_dir() and any(output.iterdir()):
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("settings") == fingerprint and set(manifest.get("file_sha256", {})) == {
                f"{s}.{kind}.jsonl" for s in split_names for kind in ("posts", "pairs")
            } and all(
                (output / name).is_file() and _digest(output / name) == checksum
                for name, checksum in manifest["file_sha256"].items()
            ):
                return manifest
        raise ValueError(f"{output} already contains different or incomplete data; use a new output directory")
    rng = random.Random(seed)
    posts, counts = _read_posts(path, sidecar, _BAD_ACCOUNTS | set(fingerprint["excluded_accounts"]))
    posts = _deduplicate(posts, counts)
    if not posts:
        raise ValueError("no usable posts after cleaning and deduplication")
    splits = {"external": posts} if external else _partition(
        posts, rng, set(fingerprint["holdout_communities"]), counts
    )
    counts["dropped_thread_conflicts"] = len(posts) - sum(map(len, splits.values())) - counts["dropped_community_holdout_conflicts"]
    counts["usable_before_split"] = len(posts)
    manifest = {"settings": fingerprint, "source": path.name, "counts": dict(counts),
                "splits": {}, "file_sha256": {}}
    # Stage beside the target so the complete directory can be renamed in
    # one filesystem operation. An existing nonempty directory is never replaced.
    with TemporaryDirectory(dir=output.parent) as scratch:
        staged = Path(scratch) / "dataset"
        staged.mkdir()
        for name, split_posts in splits.items():
            pairs = _sample_pairs(split_posts, rng, pairs_per_author)
            manifest["splits"][name] = _audit(split_posts, pairs)
            for kind, records in (("posts", split_posts), ("pairs", pairs)):
                file = staged / f"{name}.{kind}.jsonl"
                _write_records(file, records)
                manifest["file_sha256"][file.name] = _digest(file)
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
        if output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        staged.rename(output)
    return manifest


def load_split(directory: Path, split: str) -> tuple[list[Post], list[Pair]]:
    if split not in (*SPLITS, "external"):
        raise ValueError(f"split must be one of {', '.join((*SPLITS, 'external'))}")
    base = Path(directory)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if split not in manifest.get("splits", {}):
        raise ValueError(f"dataset does not contain split {split!r}")
    for kind in ("posts", "pairs"):
        file = base / f"{split}.{kind}.jsonl"
        if not file.is_file() or manifest.get("file_sha256", {}).get(file.name) != _digest(file):
            raise ValueError(f"prepared split checksum mismatch: {file.name}")
    posts = [Post(**row) for _, row in _objects(base / f"{split}.posts.jsonl")]
    pairs = [Pair(**row) for _, row in _objects(base / f"{split}.pairs.jsonl")]
    return posts, pairs
