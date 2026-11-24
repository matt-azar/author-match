"""Frozen-model, within-investigation history-size experiment on Wiki-Socks."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
import hashlib
import io
import random
import re
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from author_match.accounts import compare_accounts, _source_digest
from author_match.experiment import write_json
from author_match.profile_io import load_profile_model
from author_match.text import clean_text, comparison_words

REV = "b488e891e0840bb7d89c2dc03280af18d403f064"
BUDGETS = (5, 20, 50)
MARKUP = re.compile(r"/\*.*?\*/|\[\[.*?\]\]|\{\{.*?\}\}|https?://\S+")
AUTO = re.compile(r"undo revision|undid revision|reverted edits? by|reverted to revision|using (?:twinkle|awb|huggle)|created page with|^redirected page|^moved page|^protected |^uploaded |^added .* using", re.I)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def histories(rows):
    labels = defaultdict(dict)
    users = {r["user"].casefold() for r in rows if r["user"]}
    for r in rows:
        raw = r["message"] or ""
        if AUTO.search(raw):
            continue
        body = clean_text(MARKUP.sub(" ", raw))
        words = comparison_words(body)
        if (len(words) < 4 or not r["page"] or not r["user"]
                or any(w in users for w in words) or r["user"].casefold() in body.casefold()):
            continue
        key = " ".join(words)
        stamp = r["timestamp"][:10]
        entry = (date.fromisoformat(stamp).toordinal(), (r["page"], stamp[:7]), body, key)
        history = labels[int(r["sock"]), r["user"]]
        if key not in history or entry < history[key]:
            history[key] = entry
    return {k: sorted(h.values()) for k, h in labels.items() if len(h) >= 5}


def choose(h, used, rng):
    positive = [u for label, u in h if label and u not in used]
    controls = [u for label, u in h if not label and u not in used and (1, u) not in h]
    rng.shuffle(positive)
    rng.shuffle(controls)
    for n in reversed(BUDGETS):
        for first in positive:
            anchor = h[1, first][:n]
            if len(anchor) < n:
                continue
            pages = {r[1][0] for r in anchor}
            excluded = {r[3] for r in anchor}
            earliest = anchor[-1][0] + 60
            def pool(records):
                bins = defaultdict(list)
                for day, tag, text, key in records:
                    if day >= earliest and tag[0] not in pages and key not in excluded:
                        bins[tag].append((text, key))
                return bins
            targets = [(u, pool(h[1, u])) for u in positive if u != first and len(h[1, u]) >= n]
            targets = [(u, b) for u, b in targets if sum(map(len, b.values())) >= n]
            if not targets:
                continue
            for other in controls:
                if len(h[0, other]) < n:
                    continue
                negative = pool(h[0, other])
                for second, genuine in targets:
                    common = genuine.keys() & negative.keys()
                    if sum(min(len(genuine[k]), len(negative[k])) for k in common) < n:
                        continue
                    shared = ({v[1] for b in genuine.values() for v in b} &
                              {v[1] for b in negative.values() for v in b})
                    pairs = []
                    for k in sorted(common):
                        g = [v[0] for v in genuine[k] if v[1] not in shared]
                        c = [v[0] for v in negative[k] if v[1] not in shared]
                        rng.shuffle(g)
                        rng.shuffle(c)
                        pairs.extend(zip(g, c))
                    if len(pairs) >= n:
                        rng.shuffle(pairs)
                        left = [r[2] for r in anchor]
                        rng.shuffle(left)
                        return (first, second, other), left, pairs[:n]
    return None


def metrics(matrix):
    y = np.tile([1, 0], len(matrix))
    p = matrix.ravel()
    return np.array([roc_auc_score(y, p), np.mean((p - y)**2),
                     np.mean((matrix[:, 0]>matrix[:, 1]) + .5*(matrix[:, 0] == matrix[:, 1]))])


def summarize(cases, maximum):
    cases = [c for c in cases if maximum in c]
    result = {"investigations": len(cases), "budgets": {}}
    if not cases:
        return result
    rng = np.random.default_rng(7331)
    draws = rng.integers(len(cases), size=(500, len(cases)))
    prev = None
    for n in BUDGETS:
        if n > maximum:
            break
        matrix = np.array([c[n] for c in cases])
        point = metrics(matrix)
        result["budgets"][n] = dict(zip(("auc", "brier_50pct_prior", "within_case_concordance"), point))
        if prev is not None and len(cases)>1:
            changes = np.array([metrics(matrix[i]) - metrics(prev[i]) for i in draws])
            result["budgets"][n]["auc_change_from_previous_95pct"] = np.quantile(changes[:, 0], [.025, .975]).tolist()
        prev = matrix
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--cases", type=int, default=10000)
    a = p.parse_args()
    if a.output.resolve() == a.model.resolve() or a.cases < 1:
        p.error("choose separate paths and a positive case limit")
    model = load_profile_model(a.model)
    paths = set(git(a.source, "ls-tree", "-rz", "--name-only", REV, "data").decode().split("\0"))
    stats = csv.DictReader(io.StringIO(git(a.source, "show", REV + ":stats.csv").decode()))
    eligible = {"data/" + r["name"] for r in stats if float(r["duration"]) >= 60*86400
                and int(r["num_positives"]) >= 100 and int(r["num_negatives"]) >= 50} & paths
    digest = lambda s: hashlib.sha256(s.encode()).digest()
    paths = sorted(eligible, key=digest)[:a.cases]
    used, cases = set(), []
    dropped = Counter()
    shortfalls = Counter()
    for index, path in enumerate(paths):
        if index and index % 1000 == 0:
            print(f"Screened {index}, retained {len(cases)}, maximum budgets {dict(Counter(max(c) for c in cases))}", flush=True)
        rows = list(csv.DictReader(io.StringIO(git(a.source, "show", REV + ":" + path).decode("utf-8-sig"), newline="")))
        trial = choose(histories(rows), used, random.Random(digest(path)))
        if trial is None:
            continue
        owners, left, pairs = trial
        used.update(owners)
        scores = {}
        for n in BUDGETS:
            if n > len(left):
                break
            estimates = []
            for j in (0, 1):
                pred = compare_accounts(model, left[:n], [r[j] for r in pairs[:n]], prior=.5)
                estimates.append(pred["estimated_same_person_probability"] if pred["status"] == "estimated" else .5)
                if any(side["used_posts"] < n for side in pred["histories"]):
                    shortfalls[str(n)] += 1
                if pred["status"] != "estimated":
                    dropped[str(n)] += 1
            scores[n] = estimates
        cases.append(scores)
    write_json(a.output, {"source_revision": REV, "checkpoint_sha256": _source_digest(a.model),
                          "trials_using_fewer_than_supplied": dict(shortfalls), "eligible_files": len(eligible), "screened": len(paths), "prior_fallbacks": dict(dropped),
                          "cohorts": {k:summarize(cases, k) for k in BUDGETS}})


if __name__ == "__main__":
    main()
