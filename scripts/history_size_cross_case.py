"""Check 5/20/50 budgets with other investigated writers as time/topic-matched controls."""

import argparse
import csv
import hashlib
import io
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from author_match.accounts import compare_accounts, _source_digest
from author_match.experiment import write_json
from author_match.profile_io import load_profile_model
from author_match.text import comparison_words
from experiment_history_size import BUDGETS, REV, git, histories, summarize


def vocabulary(posts):
    return Counter(w for entry in posts for w in comparison_words(entry[2]) if w not in ENGLISH_STOP_WORDS)


def cosine(x, y):
    norm = np.sqrt(sum(i*i for i in x.values()) * sum(i*i for i in y.values()))
    return sum(i*y[k] for k, i in x.items()) / norm if norm else 0.


def pick(h, used, rng):
    owners = [u for label, u in h if label and u not in used and len(h[label, u]) >= 50]
    rng.shuffle(owners)
    for a in owners:
        left = h[1, a][:50]
        pages = {r[1][0] for r in left}
        keys = {r[3] for r in left}
        for b in owners:
            if a == b:
                continue
            right = [r for r in h[1, b] if r[0] >= left[-1][0] + 60 and r[1][0] not in pages and r[3] not in keys]
            if len(right) >= 50:
                return a, b, left, rng.sample(right, 50)
    return None


def description(left, right):
    return (left, right, vocabulary(left), vocabulary(right), {r[1][0] for r in left},
            {r[1][0] for r in right}, np.quantile([r[0] for r in right], [0, .25, .5, .75, 1]))


def partner(i, records, available):
    left, right, av, bv, ap, bp, dates = records[i]
    best, chosen = None, None
    for j in available:
        if i == j:
            continue
        l2, r2, av2, bv2, ap2, bp2, d2 = records[j]
        if (min(r[0] for r in r2) < max(r[0] for r in left) + 60 or
            min(r[0] for r in right) < max(r[0] for r in l2) + 60 or ap & bp2 or ap2 & bp):
            continue
        days = max(abs(dates - d2))
        if days > 92:
            continue
        gap = max(abs(cosine(av, bv) - cosine(av, bv2)), abs(cosine(av2, bv2) - cosine(av2, bv)))
        if gap > .05:
            continue
        cost = (gap, days, j)
        if best is None or cost < best:
            best, chosen = cost, j
    return chosen


def repeated_metrics(draws):
    # Each row is one resampling of the same cases. Both classes' scores
    # travel together when a case is included twice in a bootstrap sample.
    if not draws[0]:
        return {}
    size = len(draws[0])
    rng = np.random.default_rng(7331)
    weights = rng.multinomial(size, np.full(size, 1 / size), size=1000)
    previous = first = None
    output = {}
    for n in BUDGETS:
        scores = np.array([[case[n] for case in draw] for draw in draws])
        gaps = scores[:, :, 0, None] - scores[:, None, :, 1]
        wins = (gaps > 0) + 0.5 * (gaps == 0)
        aucs = wins.mean(axis=(1, 2))
        joint = wins.mean(axis=0)
        boot = np.einsum("bi,ij,bj->b", weights, joint, weights) / size**2
        entry = {"mean_auc": float(aucs.mean()), "draw_auc_range": [float(aucs.min()), float(aucs.max())],
                 "mean_brier": float(((scores - np.array([1, 0]))**2).mean()),
                 "mean_within_case_concordance": float(np.trace(joint) / size)}
        if first is None:
            first = boot
        else:
            entry["paired_auc_change_from_5_95pct"] = np.quantile(boot - first, [.025, .975]).tolist()
            entry["paired_auc_change_from_previous_95pct"] = np.quantile(boot - previous, [.025, .975]).tolist()
        previous = boot
        output[n] = entry
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "model", "output"):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    if args.model.resolve() == args.output.resolve():
        p.error("choose a separate output")
    stats = csv.DictReader(io.StringIO(git(args.source, "show", REV + ":stats.csv").decode()))
    paths = set(git(args.source, "ls-tree", "-rz", "--name-only", REV, "data").decode().split("\0"))
    eligible = {"data/" + r["name"] for r in stats if float(r["duration"]) >= 60*86400
              and int(r["num_positives"]) >= 100} & paths
    records, used = [], set()
    for path in sorted(eligible, key=lambda s:hashlib.sha256(s.encode()).digest()):
        rows = csv.DictReader(io.StringIO(git(args.source, "show", REV + ":" + path).decode("utf-8-sig"), newline=""))
        positive = [r for r in rows if r["sock"] == "1"]
        if sum(v >= 50 for v in Counter(r["user"] for r in positive).values())<2:
            continue
        rng = random.Random(hashlib.sha256(path.encode()).digest())
        selected = pick(histories(positive), used, rng)
        if selected:
            a, b, left, right = selected
            used.update((a, b))
            rng.shuffle(left)
            rng.shuffle(right)
            records.append(description(left, right))
    model = load_profile_model(args.model)
    available = set(range(len(records)))
    draws = [[] for _ in range(20)]
    fallback = Counter()
    shortfalls = Counter()
    # Use one anchor per disjoint investigation pair so the case bootstrap
    # never treats a reused reference or writer as an independent case.
    for i in range(len(records)):
        if i not in available:
            continue
        j = partner(i, records, available)
        if j is None:
            continue
        available.difference_update((i, j))
        left, right = records[i][:2]
        negative = records[j][1]
        rng = random.Random(73451 + i)
        sampled_histories = [list(h) for h in (left, right, negative)]
        for r, draw in enumerate(draws):
            if r:
                for history in sampled_histories:
                    rng.shuffle(history)
            result = {}
            for n in BUDGETS:
                if n == 50 and r:
                    result[n] = draws[0][-1][n]
                    continue
                anchor = [entry[2] for entry in sampled_histories[0][:n]]
                result[n] = []
                for reference in sampled_histories[1:]:
                    pred = compare_accounts(model, anchor, [entry[2] for entry in reference[:n]], prior=.5)
                    if not r:
                        if pred["status"] != "estimated":
                            fallback[n] += 1
                        if any(h["used_posts"] < n for h in pred["histories"]):
                            shortfalls[n] += 1
                    result[n].append(pred["estimated_same_person_probability"] if pred["status"] == "estimated" else .5)
            draw.append(result)
    write_json(args.output, {"source_revision": REV, "checkpoint_sha256": _source_digest(args.model),
                            "screened": len(eligible), "eligible_positive_cases": len(records),
                            "prior_fallbacks": dict(fallback), "under_budget": dict(shortfalls),
                            "cohort": summarize(draws[0], 50), "nested_draws": len(draws),
                            "repeated_sampling": repeated_metrics(draws)})


if __name__ == "__main__":
    main()
