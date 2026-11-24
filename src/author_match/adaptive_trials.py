"""Matched account trials with independently varying amounts of writing on each side."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from math import exp, log, log1p
import random

from .profile_trials import LABEL_BASES, ProfileTrial, _side_pools, _validated
from .text import word_count


_ROUNDS_PER_OWNER = 6
_MAX_OWNER_APPEARANCES = 18


def _budget(available, rng):
    # Some histories arrive intact. Thinning the others on a log scale gives
    # a 100-post account opportunities to supply 1, 10 or nearly 100 posts.
    if available == 1 or rng.random() < 0.2:
        return available
    return min(available, int(exp(rng.uniform(0, log(available + 1)))))


def _group_by_community(posts):
    groups = defaultdict(list)
    for p in posts:
        groups[p.community].append(p)
    return groups


def _key(posts):
    return tuple(sorted(p.id for p in posts))


def _pair_key(left, right):
    return tuple(sorted((left, _key(right))))


def _control(anchor, positive, catalog, by_handle, uses, lengths, seen, rng):
    # Look for similarly large accounts first, but allow the reference budget
    # to shrink in BOTH labels if a comparable control has fewer posts.
    handles, sizes = catalog
    start = bisect_left(sizes, len(positive))
    indices = rng.sample(range(start, len(handles)), min(12, len(handles) - start))
    indices += rng.sample(range(len(handles)), min(12, len(handles)))
    own_groups = _group_by_community(positive)
    forbidden = {p.thread for p in anchor if p.thread is not None}
    a_key = _key(anchor)
    best = cost = None
    for index in dict.fromkeys(indices):
        pool = by_handle[handles[index]]
        owner = pool[0].person
        if owner == anchor[0].person or uses[owner] >= _MAX_OWNER_APPEARANCES:
            continue
        groups = _group_by_community(
            p for p in pool if p.community in own_groups and p.thread not in forbidden)
        positive_ref, negative_ref = [], []
        for community, own in own_groups.items():
            other = groups[community]
            n = min(len(other), len(own))
            positive_ref.extend(own[:n])
            negative_ref.extend(rng.sample(other, n))
        n = len(negative_ref)
        if not n:
            continue
        pos_words = sum(lengths[p.id] for p in positive_ref)
        neg_words = sum(lengths[p.id] for p in negative_ref)
        # Both classes have identical anchor, reference post count and
        # community histogram. Reject a poor word-volume match instead of
        # making a much shorter negative an easy training example.
        error = abs(log1p(pos_words) - log1p(neg_words))
        rank = (len(positive) - n, error)
        if error > log(1.5) or (cost is not None and rank >= cost):
            continue
        pk = _pair_key(a_key, positive_ref)
        nk = _pair_key(a_key, negative_ref)
        if pk in seen or nk in seen:
            continue
        cost, best = rank, (positive_ref, negative_ref, owner, pk, nk)
        if n == len(positive) and pos_words == neg_words:
            break
    return best


def sample_adaptive_trials(posts, *, label_basis: str, seed: int = 42) -> list[ProfileTrial]:
    """Sample from one preassigned split. Post counts range from one to full history.

    Each positive is accompanied by a count- and community-matched negative
    sharing its complete anchor. Known threads are disjoint across sides, and
    controls stay in the owner's recording window. Same-account proxies
    require known, distinct threads; explicit links require distinct handles.
    """
    if label_basis not in LABEL_BASES:
        raise ValueError(f"label_basis must be one of {', '.join(LABEL_BASES)}")
    linked = label_basis == "linked_handles"
    posts = _validated(posts)
    handles = defaultdict(set)
    threads = defaultdict(lambda: defaultdict(list))
    by_handle = defaultdict(list)
    for post in posts:
        handles[post.person].add(post.handle)
        if linked or post.thread is not None:
            threads[post.handle][post.thread].append(post)
            by_handle[post.handle].append(post)
    if not linked and any(len(names) != 1 for names in handles.values()):
        raise ValueError("account_proxy requires one observed handle per person")
    lengths = {p.id: word_count(p.text) for p in posts}
    windows = defaultdict(list)
    for h, pool in by_handle.items():
        windows[pool[0].split_group].append(h)
    catalogs = {}
    for window, names in windows.items():
        names.sort(key=lambda h: (len(by_handle[h]), h))
        catalogs[window] = (names, [len(by_handle[h]) for h in names])

    eligible = sorted(person for person, names in handles.items()
                      if (len(names) >= 2 if linked else len(threads[next(iter(names))]) >= 2))
    rng = random.Random(seed)
    rng.shuffle(eligible)
    uses, seen, trials = Counter(), set(), []
    for epoch in range(_ROUNDS_PER_OWNER):
        for i, person in enumerate(eligible):
            if uses[person] + 2 > _MAX_OWNER_APPEARANCES:
                continue
            names = sorted(handles[person])
            first, second = rng.sample(names, 2) if linked else (names[0], names[0])
            left, right = _side_pools(threads[first], threads[second], rng)
            if not left or not right:
                continue
            full = (i + epoch) % _ROUNDS_PER_OWNER == 0
            anchor = rng.sample(left, len(left) if full else _budget(len(left), rng))
            positive = rng.sample(right, len(right) if full else _budget(len(right), rng))
            selected = _control(anchor, positive, catalogs[anchor[0].split_group],
                                by_handle, uses, lengths, seen, rng)
            if selected is None:
                continue
            own, other, different, pk, nk = selected
            seen.update((pk, nk))
            a = tuple(p.text for p in anchor)
            trials.append(ProfileTrial(a, tuple(p.text for p in own), True, person, person))
            trials.append(ProfileTrial(a, tuple(p.text for p in other), False, person, different))
            uses[person] += 2
            uses[different] += 1
    rng.shuffle(trials)
    return trials
