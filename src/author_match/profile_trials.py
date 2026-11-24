"""Build account-history trials while keeping identity labels outside the text."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from math import log1p
from pathlib import Path
import random

from .data import SPLITS, SPLIT_WEIGHTS, _objects
from .profile_features import MIN_POSTS, MIN_WORDS, usable_post_key
from .text import clean_text, comparison_words, word_count


LABEL_BASES = ("account_proxy", "linked_handles")
_MAX_APPEARANCES = 18
_ROUNDS = 6


@dataclass(frozen=True, slots=True)
class AccountPost:
    id: str
    person: str
    handle: str
    thread: str | None
    text: str
    community: str | None = None
    split_group: str | None = None


@dataclass(frozen=True, slots=True)
class ProfileTrial:
    left: tuple[str, ...]
    right: tuple[str, ...]
    same: bool
    left_person: str
    right_person: str


def _id(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty ID")
    return str(value).strip()


def _validated(posts):
    """Reject inconsistent labels. Collapse repeats, excluding cross-handle copies."""
    owners = {}
    records = {}
    person_groups = {}
    for p in posts:
        person = _id(p.person, "person_id")
        handle = _id(p.handle, "handle")
        pid = _id(p.id, "id")
        thread = None if p.thread in (None, "") else _id(p.thread, "thread")
        if owners.setdefault(handle, person) != person:
            raise ValueError(f"handle {handle!r} is assigned to multiple people")
        if not isinstance(p.text, str):
            raise ValueError("text must be a string")
        community = getattr(p, "community", None)
        community = _id(community, "community") if community not in ("", None) else None
        group = getattr(p, "split_group", None)
        group = _id(group, "split_group") if group not in ("", None) else None
        if group is not None and person_groups.setdefault(person, group) != group:
            raise ValueError(f"person {person!r} belongs to incompatible split groups")
        record = AccountPost(pid, person, handle, thread, clean_text(p.text), community, group)
        if records.setdefault(pid, record) != record:
            raise ValueError(f"conflicting records for post ID {pid!r}")
    canonical = {}
    ambiguous = set()
    for pid in sorted(records):
        p = records[pid]
        if not p.text or p.text.casefold() in ("[removed]", "[deleted]"):
            continue
        tokens = comparison_words(p.text)
        key = usable_post_key(p.text, tokens)
        if key is None:
            continue
        original = canonical.setdefault(key, p)
        if original.handle != p.handle:
            ambiguous.add(key)
    return [replace(p, split_group=person_groups[p.person])
            if not p.split_group and p.person in person_groups else p
            for key, p in canonical.items() if key not in ambiguous]


def read_linked_posts(path: Path) -> list[AccountPost]:
    """Read JSONL(.gz) rows with person_id, handle, text, and optional id/thread/community/split_group.

    A person_id groups known aliases. Provide globally scoped IDs when joining
    platforms. Absent post IDs are derived deterministically from row contents.
    """
    path = Path(path)
    posts = []
    for line, row in _objects(path):
        try:
            person = _id(row.get("person_id"), "person_id")
            handle = _id(row.get("handle"), "handle")
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            thread = row.get("thread")
            thread = None if thread in ("", None) else _id(thread, "thread")
            if row.get("id") is None:
                key = json.dumps([person, handle, thread, text], ensure_ascii=False).encode()
                pid = "derived:" + hashlib.sha256(key).hexdigest()
            else:
                pid = _id(row["id"], "id")
        except ValueError as e:
            raise ValueError(f"{path.name}:{line}: {e}") from e
        posts.append(AccountPost(pid, person, handle, thread, text, row.get("community"), row.get("split_group")))
    return _validated(posts)


def proxy_posts(posts) -> list[AccountPost]:
    """Adapt single-post benchmark records, with one observed account per identity."""
    return [AccountPost(p.id, p.author, p.author, p.thread, p.text, p.community) for p in posts]


def partition_people(posts, *, seed: int = 42, transfer: bool = False) -> dict[str, list[AccountPost]]:
    """Keep people, their aliases and any labeled observation windows together.

    Transfer reserves approximately 60% for calibration and 40% for testing.
    With explicit groups, balance by linked people and total population while
    assigning entire groups; exact fractions depend on the available blocks.
    """
    posts = _validated(posts)
    handles = defaultdict(set)
    blocks = defaultdict(set)
    for p in posts:
        handles[p.person].add(p.handle)
        block = ("window", p.split_group) if p.split_group else ("person", p.person)
        blocks[block].add(p.person)
    names, weights = (("calibration", "test"), (0.6, 0.4)) if transfer else (SPLITS, SPLIT_WEIGHTS)
    rng = random.Random(seed)
    owners = {}
    if any(p.split_group for p in posts):
        groups = list(sorted(blocks))
        rng.shuffle(groups)
        linked = {p for p, hs in handles.items() if len(hs) >= 2}
        def capacity(group):
            members = blocks[group]
            n_linked = len(members & linked)
            return bool(n_linked and len(members) > 1), n_linked, len(members)
        groups.sort(key=capacity, reverse=True)
        # Seed separate partitions with blocks containing both a positive
        # owner and a distinct within-window negative control, when available.
        viable = sum(capacity(g)[0] for g in groups)
        if transfer and viable < 2:
            raise ValueError("grouped transfer requires two independent windows with linked writers and controls")
        seed_names = sorted(names, key=lambda n: weights[names.index(n)], reverse=True)
        linked_used = Counter()
        people_used = Counter()
        for i, group in enumerate(groups):
            _, n_linked, n_people = capacity(group)
            if i < min(viable, len(names)):
                name = seed_names[i]
            else:
                def cost(name):
                    w = weights[names.index(name)]
                    # Change in squared deviation from each target share.
                    x = (linked_used[name] / max(1, len(linked))) - w
                    a = n_linked / max(1, len(linked))
                    y = (people_used[name] / len(handles)) - w
                    b = n_people / len(handles)
                    return 2 * ((x + a) ** 2 - x ** 2) + (y + b) ** 2 - y ** 2
                name = min(names, key=cost)
            owners.update(dict.fromkeys(blocks[group], name))
            linked_used[name] += n_linked
            people_used[name] += n_people
    else:
        for multiple in (True, False):
            people = sorted(p for p, hs in handles.items() if (len(hs) > 1) == multiple)
            rng.shuffle(people)
            start = 0
            for i, name in enumerate(names):
                stop = round(sum(weights[:i + 1]) * len(people))
                owners.update(dict.fromkeys(people[start:stop], name))
                start = stop
    splits = {name: [] for name in SPLITS}
    for p in posts:
        splits[owners[p.person]].append(p)
    return splits


def _history_sizes(left, right, phase, rng):
    """Cover sparse, asymmetric, medium and full histories; max size follows data."""
    if phase == 0:
        return MIN_POSTS, MIN_POSTS
    if phase == 1:
        return rng.randint(MIN_POSTS, min(left, 4)), rng.randint(MIN_POSTS, min(right, 4))
    if phase == 2:
        return MIN_POSTS, right
    if phase == 3:
        return left, MIN_POSTS
    if phase == 4:
        return (rng.randint(min(left, 5), min(left, 24)),
                rng.randint(min(right, 5), min(right, 24)))
    return left, right


def _draw_history(pool, count, rng, word_counts):
    sample = rng.sample(pool, count)
    size = sum(word_counts[p.id] for p in sample)
    if size < MIN_WORDS:
        used = {p.id for p in sample}
        rest = [p for p in pool if p.id not in used]
        rng.shuffle(rest)
        for p in rest:
            sample.append(p)
            size += word_counts[p.id]
            if size >= MIN_WORDS:
                break
    return sample if size >= MIN_WORDS else []


def _side_pools(first, second, rng):
    # Assign each known thread shared by the handles to exactly one side. An
    # individual profile can contain several contributions to one conversation.
    common = sorted((first.keys() & second.keys()) - {None})
    rng.shuffle(common)
    left_threads = set(common[:(len(common) + 1) // 2])
    if len(common) == 1 and not (second.keys() - set(common)) and first.keys() - set(common):
        left_threads.clear()
    right_threads = set(common) - left_threads
    left = [p for t, group in first.items() if t not in right_threads for p in group]
    right = [p for t, group in second.items() if t not in left_threads for p in group]
    return left, right


def sample_trials(posts, *, label_basis: str, seed: int = 42) -> list[ProfileTrial]:
    """Generate paired positives and negatives from one person-disjoint partition.

    An account_proxy positive uses one handle across disjoint known threads.
    A linked_handles positive uses two handles of the same labeled person.
    Missing threads are allowed only for the latter basis; known threads are
    disjoint across trial sides for both labels. Each negative shares its
    positive's anchor and side counts. Candidate reference histories are
    selected for similar community mix and total words, within available data.
    """
    if label_basis not in LABEL_BASES:
        raise ValueError(f"label_basis must be one of {', '.join(LABEL_BASES)}")
    posts = _validated(posts)
    threads = defaultdict(lambda: defaultdict(list))
    people = defaultdict(set)
    for p in posts:
        people[p.person].add(p.handle)
        if p.thread is not None or label_basis == "linked_handles":
            threads[p.handle][p.thread].append(p)
    if label_basis == "account_proxy" and any(len(hs) != 1 for hs in people.values()):
        raise ValueError("account_proxy requires one observed handle per person")
    word_counts = {p.id: word_count(p.text) for p in posts}
    by_handle = {}
    for h, groups in threads.items():
        pool = [p for group in groups.values() for p in group]
        if len(pool) >= MIN_POSTS and sum(word_counts[p.id] for p in pool) >= MIN_WORDS:
            by_handle[h] = pool
    by_window = defaultdict(list)
    for h, pool in by_handle.items():
        by_window[None].append(h)
        if pool[0].split_group:
            by_window[pool[0].split_group].append(h)
    catalogs = {}
    for window, hs in by_window.items():
        hs.sort(key=lambda h: (len(by_handle[h]), h))
        catalogs[window] = hs, [len(by_handle[h]) for h in hs]
    eligible = sorted(person for person, hs in people.items() if
                      (len(hs) >= 2 if label_basis == "linked_handles" else len(threads[next(iter(hs))]) >= 2))
    rng = random.Random(seed)
    rng.shuffle(eligible)
    result = []
    appearances = Counter()
    seen = set()

    def best_negative(anchor_person, anchor, positive):
        anchor_threads = {p.thread for p in anchor if p.thread is not None}
        # Include references with enough material even when highly active
        # handles are rare. Also sample from the full set as a fallback.
        window = positive[0].split_group if label_basis == "linked_handles" else None
        handle_index, sizes = catalogs[window]
        start = bisect_left(sizes, len(positive))
        candidates = rng.sample(range(start, len(sizes)), min(12, len(sizes) - start))
        candidates += rng.sample(range(len(sizes)), min(12, len(sizes)))
        best_key, best = None, None
        a_ids = tuple(sorted(p.id for p in anchor))
        # Prefixes are random because the positive was sampled before matching.
        sums = [0]
        for p in positive:
            sums.append(sums[-1] + word_counts[p.id])
        for idx in dict.fromkeys(candidates):
            h = handle_index[idx]
            pool = by_handle[h]
            person = pool[0].person
            if person == anchor_person or appearances[person] >= _MAX_APPEARANCES:
                continue
            if not anchor_threads.isdisjoint(threads[h]):
                pool = [p for p in pool if p.thread not in anchor_threads]
            count = min(len(positive), len(pool))
            if count < MIN_POSTS or sums[count] < MIN_WORDS:
                continue
            candidate = rng.sample(pool, count)
            size = sum(word_counts[p.id] for p in candidate)
            if size < MIN_WORDS:
                continue
            # Match counts, then community membership, then total words.
            # Community IDs serve only to construct comparable trials.
            own_mix = Counter(p.community for p in positive[:count])
            neg_mix = Counter(p.community for p in candidate)
            mismatch = count - sum((own_mix & neg_mix).values())
            if window and mismatch:
                continue
            key = (len(positive) - count, mismatch, abs(log1p(size) - log1p(sums[count])))
            n_ids = tuple(sorted(p.id for p in candidate))
            p_ids = tuple(sorted(p.id for p in positive[:count]))
            n_key = tuple(sorted((a_ids, n_ids)))
            p_key = tuple(sorted((a_ids, p_ids)))
            if (best_key is None or key < best_key) and n_key not in seen and p_key not in seen:
                best_key, best = key, (positive[:count], candidate, person, p_key, n_key)
        return best

    for round_no in range(_ROUNDS):
        for i, person in enumerate(eligible):
            if appearances[person] > _MAX_APPEARANCES - 2:
                continue
            handles = sorted(h for h in people[person] if h in by_handle)
            if label_basis == "linked_handles":
                if len(handles) < 2:
                    continue
                h1, h2 = rng.sample(handles, 2)
            else:
                if not handles:
                    continue
                h1 = h2 = handles[0]
            left, right = _side_pools(threads[h1], threads[h2], rng)
            if min(len(left), len(right)) < MIN_POSTS:
                continue
            phase = (i + round_no) % _ROUNDS
            nl, nr = _history_sizes(len(left), len(right), phase, rng)
            anchor = _draw_history(left, nl, rng, word_counts)
            positive = _draw_history(right, nr, rng, word_counts)
            if not anchor or not positive:
                continue
            match = best_negative(person, anchor, positive)
            if match is None:
                continue
            own, other, different_person, positive_key, negative_key = match
            seen.update((positive_key, negative_key))
            left_text = tuple(p.text for p in anchor)
            result.append(ProfileTrial(left_text, tuple(p.text for p in own), True, person, person))
            result.append(ProfileTrial(left_text, tuple(p.text for p in other), False, person, different_person))
            appearances[person] += 2
            appearances[different_person] += 1
    rng.shuffle(result)
    return result
