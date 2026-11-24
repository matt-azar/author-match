"""Calibrate multiple independent posts per writer as one verification trial."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
import random

import numpy as np

from .data import Pair, Post
from .metrics import LengthPolicy, decision, fit_policy, report, thresholds_for
from .text import clean_text, evidence_reason, word_count

MODES = {"1x5": (1, 5), "5x5": (5, 5)}
AGGREGATION = "median of eligible single-post pair scores"


@dataclass(frozen=True)
class BundlePolicy:
    mode: str
    thresholds: LengthPolicy


@dataclass(frozen=True)
class _Side:
    posts: tuple
    threads: frozenset
    communities: frozenset
    # Report a combined word count. Only the individual posts go through the verifier.
    summary: Post

    @classmethod
    def build(cls, posts, side_id):
        posts = tuple(posts)
        summary = Post(side_id, posts[0].author, "", "", "\n".join(p.text for p in posts))
        return cls(posts, frozenset(p.thread for p in posts),
                   frozenset(p.community for p in posts if p.community), summary)


def _community_condition(left, right):
    if not left.communities or not right.communities:
        return "unknown_community"
    if left.communities & right.communities:
        return "shared_community"
    return "separate_communities"


def _sample(posts, mode, seed):
    """Pair each qualifying author with itself and at most one different reference.

    A reference may be used once as a negative and once as a positive. All posts
    in a trial come from distinct conversations. Calibration counts trials,
    never the component comparisons within a trial.
    """
    if mode not in MODES:
        raise ValueError(f"bundle mode must be one of {', '.join(MODES)}")
    a, b = MODES[mode]
    rng = random.Random(seed)
    authors = defaultdict(lambda: defaultdict(list))
    for post in sorted(posts, key=lambda p: p.id):
        if post.thread:
            authors[post.author][post.thread].append(post)

    sides = {}
    left_ids, right_ids = [], []
    community_refs = defaultdict(list)
    lengths = {}
    for author, threads in authors.items():
        if len(threads) < a + b:
            continue
        thread_ids = rng.sample(list(threads), a + b)
        samples = [rng.choice(threads[t]) for t in thread_ids]
        i = len(left_ids)
        left, right = _Side.build(samples[:a], f"L{i}"), _Side.build(samples[a:], f"R{i}")
        left_ids.append(left.summary.id)
        right_ids.append(right.summary.id)
        sides[left.summary.id] = left
        sides[right.summary.id] = right
        lengths[right.summary.id] = word_count(right.summary.text)
        for community in sorted(right.communities):
            community_refs[community].append(right.summary.id)

    available = set(right_ids)
    trials = []
    indices = list(range(len(left_ids)))
    rng.shuffle(indices)
    for i in indices:
        left, own = sides[left_ids[i]], sides[right_ids[i]]
        condition = _community_condition(left, own)
        # Candidate length matching makes it harder to identify the label
        # from a prolific writer's unusually long posts.
        candidates = rng.sample(right_ids, min(64, len(right_ids)))
        for community in sorted(left.communities):
            refs = community_refs[community]
            candidates.extend(rng.sample(refs, min(16, len(refs))))
        best = None
        for key in dict.fromkeys(candidates):
            if key == own.summary.id or key not in available or left.threads & sides[key].threads:
                continue
            priority = (_community_condition(left, sides[key]) != condition,
                        abs(lengths[own.summary.id] - lengths[key]))
            if best is None or priority < best[0]:
                best = priority, key
        if best is None:
            continue
        other_id = best[1]
        available.remove(other_id)
        trials.append(Pair(left.summary.id, own.summary.id, True, condition))
        trials.append(Pair(left.summary.id, other_id, False, _community_condition(left, sides[other_id])))
    return sides, trials


def _dependent_posts(side):
    for a, b in combinations(side.posts, 2):
        if a.text.casefold().strip() == b.text.casefold().strip() or evidence_reason(a.text, b.text) == "copied_overlap":
            return True
    return False


def _score(verifier, sides, trials):
    scores, reasons = [], []
    duplicates = {key: _dependent_posts(side) for key, side in sides.items()}
    for start in range(0, len(trials), 128):
        block = trials[start:start + 128]
        components, component_reasons, offsets, originals = [], [], [], {}
        for trial in block:
            left, right = sides[trial.left], sides[trial.right]
            offsets.append(len(components))
            for a, b in product(left.posts, right.posts):
                originals[a.id] = a
                originals[b.id] = b
                components.append(Pair(a.id, b.id, trial.same, trial.condition))
                reason = evidence_reason(a.text, b.text)
                if a.text and a.text.casefold() == b.text.casefold():
                    reason = "copied_overlap"
                component_reasons.append(reason)
        offsets.append(len(components))
        estimates = verifier.scores(list(originals.values()), components)
        for trial, lo, hi in zip(block, offsets, offsets[1:]):
            flags = component_reasons[lo:hi]
            usable = [s for s, flag in zip(estimates[lo:hi], flags) if flag is None]
            reason = None
            if "copied_overlap" in flags or duplicates[trial.left] or duplicates[trial.right]:
                reason = "copied_overlap"
            elif len(usable) <= len(flags) // 2:
                reason = "insufficient_independent_text"
            # report() also describes the raw ranking for excluded trials.
            scores.append(float(np.median(usable if usable else estimates[lo:hi])))
            reasons.append(reason)
    return scores, reasons


def _observations(verifier, posts, mode, seed):
    sides, trials = _sample(posts, mode, seed)
    scores, reasons = _score(verifier, sides, trials)
    return [side.summary for side in sides.values()], trials, scores, reasons


def _report(observations, policy, bootstrap_samples, seed):
    summaries, pairs, scores, reasons = observations
    result = report(summaries, pairs, scores, reasons, policy.thresholds,
                    bootstrap_samples=bootstrap_samples, seed=seed)
    result.update(mode=policy.mode, aggregation=AGGREGATION,
                  length_unit="total cleaned words on the shorter side",
                  eligible_authors=len(summaries) // 2, sampling_seed=seed)
    return result


def calibrate_bundles(verifier, posts, mode, *, seed=42, false_match_rate=0.01, false_reject_rate=0.05):
    """Use only calibration writers. The base verifier is already fitted and fixed."""
    observations = _observations(verifier, posts, mode, seed)
    policy = BundlePolicy(mode, fit_policy(*observations, false_match_rate, false_reject_rate))
    return policy, _report(observations, policy, 0, seed)


def evaluate_bundles(verifier, posts, policy, *, seed=42, bootstrap_samples=200):
    observations = _observations(verifier, posts, policy.mode, seed)
    return _report(observations, policy, bootstrap_samples, seed)


def compare_bundles(verifier, left, right, policy):
    """Each list is supplied as independent posts from one writer in separate threads."""
    if isinstance(left, str) or isinstance(right, str) or policy.mode not in MODES:
        raise ValueError("provide one sequence of independent posts per side and a fitted bundle policy")
    shape = (len(left), len(right))
    if sorted(shape) != sorted(MODES[policy.mode]):
        raise ValueError(f"post counts {shape} do not match calibrated mode {policy.mode}")
    sides = {}
    for key, texts in (("L", left), ("R", right)):
        posts = [Post(f"{key}{i}", key, "", "", clean_text(text)) for i, text in enumerate(texts)]
        sides[key] = _Side.build(posts, key)
    scores, reasons = _score(verifier, sides, [Pair("L", "R", False, "unknown_community")])
    score, reason = scores[0], reasons[0]
    counts = [word_count(sides[key].summary.text) for key in ("L", "R")]
    thresholds = thresholds_for(min(counts), policy.thresholds)
    outcome = decision(score, reason, thresholds)
    if outcome == "inconclusive" and reason is None:
        reason = ("no_supported_threshold" if thresholds.upper is None and thresholds.lower is None
                  else "insufficient_score_evidence")
    return {"decision": outcome, "reason": reason, "score": None if reasons[0] else score,
            "mode": policy.mode, "aggregation": AGGREGATION, "posts": list(shape), "words": counts}
