"""Compare text evidence for an already nominated pair, with window-disjoint model selection."""

import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path
import re
import zlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from author_match import adaptive_features
from author_match.accounts import _identities, _source_digest, _sizes, _support
from author_match.adaptive_trials import sample_adaptive_trials
from author_match.data import load_split
from author_match.experiment import write_json
from author_match.model import _FUNCTION_WORDS
from author_match.profile_features import adaptive_pair_reason, independent_profiles, make_profile
from author_match.profile_io import load_profile_model
from author_match.profile_trials import partition_people, proxy_posts, read_linked_posts
from pair_pattern_experiment import PatternEvidence
from compare_account_methods import compare_predictions, content_cosine, draw_windows, lexical_matches
from evaluate_account_transfer import excluded_writing, transfer_trials, person_weights

COMMON = frozenset(_FUNCTION_WORDS)
METHODS = {"ranker": [0], "compression": [1, 2], "boundary_habits": [3, 4, 5],
           "ranker_and_compression": [0, 1, 2], "ranker_and_habits": [0, 3, 4, 5],
           "combined": list(range(6)), "tfidf_cosine": [7], "learned_patterns": [6],
           "ranker_and_patterns": [0, 6, 7], "all_signals": list(range(8))}


def mask(word):
    w = word.group()
    if w.casefold() in COMMON:
        return w
    return ("A" if w.isupper() else "T" if w.istitle() else "w") + str(min(9, len(w)))


@lru_cache(maxsize=2048)
def encode(posts):
    raw = "\n".join(posts)
    masked = [re.sub(r"[^\W\d_]+", mask, p) for p in posts]
    habits = []
    for values in ([p[:6] for p in masked], [p[ - 6:] for p in masked],
                   [g for p in masked for g in zip(p, p[1:], p[2:])]):
        habits.append(Counter(values))
    streams = [s.encode() for s in (raw, "\n".join(masked))]
    return [(s, len(zlib.compress(s))) for s in streams], habits


def signals(a, b):
    def ordered(profile):
        return tuple(profile._originals[k] for k in sorted(profile._originals))
    (streams_a, habits_a), (streams_b, habits_b) = encode(ordered(a)), encode(ordered(b))
    values = []
    for (x, lx), (y, ly) in zip(streams_a, streams_b):
        joint = (len(zlib.compress(x + b"\n" + y)) + len(zlib.compress(y + b"\n" + x))) / 2
        values.append((lx + ly - joint) / max(lx, ly))
    for x, y in zip(habits_a, habits_b):
        norm = np.sqrt(sum(v * v for v in x.values()) * sum(v * v for v in y.values()))
        values.append(sum(v * y[k] for k, v in x.items()) / norm if norm else 0.)
    return values


def collect(model, trials, patterns):
    features, legacy = [], []
    cal = _support(model)
    profile = lru_cache(maxsize=256)(make_profile)
    for t in trials:
        a, b = independent_profiles(profile(t.left), profile(t.right))
        if adaptive_pair_reason(a, b):
            features.append([np.nan] * 8)
            legacy.append(0.)
            continue
        margin = model.model.margin(adaptive_features.pair_features(a, b))
        features.append([margin, *signals(a, b), *patterns.score(a, b)])
        legacy.append(cal.estimate(margin, _sizes(a, b), .5)["log_likelihood_ratio"])
    return np.array(features), np.array(legacy)


def fit(x, y, groups, cols):
    valid = np.isfinite(x[:, 0])
    x, y, groups = x[valid][:, cols], y[valid], groups[valid]
    # Give each label equal total weight and each window within that label an equal vote.
    counts = Counter(zip(groups, y))
    per_class = {v:len(set(groups[y == v])) for v in (0, 1)}
    w = np.array([len(set(groups)) / (counts[g, v] * per_class[v]) for g, v in zip(groups, y)])
    scaler = StandardScaler().fit(x, sample_weight=w)
    reg = LogisticRegression(C=1.).fit(scaler.transform(x), y, sample_weight=w)
    return scaler, reg, cols, np.log1p(len(set(groups)))


def predict(model, x):
    scaler, reg, cols, bound = model
    valid = np.isfinite(x[:, 0])
    scores = np.zeros(len(x))
    if valid.any():
        scores[valid] = np.clip(reg.decision_function(scaler.transform(x[valid][:, cols])), -bound, bound)
    return scores


def loss(y, z):
    return float(np.mean(np.logaddexp(0, (1 - 2 * y) * z)))


def matched(trials, groups):
    pairs = lexical_matches(trials, groups, [content_cosine(t.left, t.right) for t in trials])
    return np.array(pairs, dtype=int).ravel()


def operating(y, z, threshold):
    above = z > threshold
    return {"threshold_log_odds": float(threshold), "true_positive": int(sum(above & (y == 1))),
            "false_positive": int(sum(above & (y == 0))), "same": int(sum(y)), "different": int(sum(1 - y))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["linked", "checkpoint", "prepared", "external", "exclude_handles", "exclude_windows", "output"]:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    if a.output.resolve() in {path.resolve() for path in (a.linked, a.checkpoint, a.prepared, a.external, a.exclude_handles, a.exclude_windows)}:
        p.error("choose a separate output")
    model = load_profile_model(a.checkpoint)
    info = model.provenance
    if (info.get("method") != "adaptive" or not info.get("transfer")
            or info["source_sha256"] != _source_digest(a.linked)
            or info["pretrain_source_sha256"] != _source_digest(a.prepared)):
        p.error("expected the linked-source adaptive transfer checkpoint and its original prepared corpus")
    if (_source_digest(a.exclude_handles) != info["exclusions_sha256"]
            or _source_digest(a.exclude_windows) != info["excluded_windows_sha256"]):
        p.error("pretraining exclusions must match the frozen ranker")
    seed = info["seed"]
    linked = read_linked_posts(a.linked)
    blocked = set(a.exclude_handles.read_text().splitlines()) | {p.handle for p in linked} | {p.person for p in linked}
    windows = tuple(a.exclude_windows.read_text().splitlines())
    corpus = [p for p in proxy_posts(load_split(a.prepared, "train")[0])
              if p.handle not in blocked and p.person not in blocked and not p.id.startswith(windows)]
    patterns = PatternEvidence(corpus, seed)
    splits = partition_people(linked, seed=seed, transfer=True)
    def dataset(name, offset):
        posts = splits[name]
        group = {p.person:p.split_group for p in posts}
        trials = sample_adaptive_trials(posts, label_basis="linked_handles", seed=seed + offset)
        return trials, np.array([group[t.left_person] for t in trials])
    train, g = dataset("calibration", 1)
    x, _ = collect(model, train, patterns)
    y = np.array([int(t.same) for t in train])
    hard = matched(train, g)
    if not len(hard):
        raise ValueError("no lexical-matched trials for method selection")
    folds = list(GroupKFold(5).split(x, y, g))
    development = {}
    for name, cols in METHODS.items():
        pred = np.zeros(len(y))
        for tr, va in folds:
            pred[va] = predict(fit(x[tr], y[tr], g[tr], cols), x[va])
        all_loss, hard_loss = loss(y, pred), loss(y[hard], pred[hard])
        development[name] = {"auc": float(roc_auc_score(y, pred)), "log_loss": all_loss,
                           "matched_log_loss": hard_loss, "selection_loss": (all_loss + hard_loss) / 2,
                           "threshold": float(np.quantile(pred[y == 0], .9, method="higher"))}
    winner = min(development, key=lambda n:development[n]["selection_loss"])
    frozen = fit(x, y, g, METHODS[winner])
    report = {"checkpoint_sha256": _source_digest(a.checkpoint), "selection": "Equal weight on overall and lexical-matched out-of-fold log loss, five calibration-window folds",
            "pattern_training_pairs": patterns.training_trials, "development_trials": len(y), "development_windows": len(set(g)), "development_matched_trials": len(hard),
            "development": development, "winner": winner, "evaluation": {}}
    if _identities(splits["test"]) & set(model.fit_people):
        raise ValueError("test identity overlap")
    test, windows = dataset("test", 2)
    external = proxy_posts(load_split(a.external, "external")[0])
    if _identities(external) & set(model.fit_people):
        raise ValueError("external identity overlap")
    outside, _ = transfer_trials(external, excluded_writing([a.prepared, a.linked]), seed + 2)
    for name, trials, groups in (("rename", test, windows), ("external", [t for _, t in outside], None)):
        features, base = collect(model, trials, patterns)
        pred = predict(frozen, features)
        labels = np.array([int(t.same) for t in trials])
        subsets = {"all": np.arange(len(trials))}
        if groups is not None:
            subsets["lexical_matched"] = matched(trials, groups)
        for subset, ix in subsets.items():
            counts = (draw_windows(groups[ix], seed=seed, replicates=1000) if groups is not None
                    else person_weights(outside, seed, 1000))
            report["evaluation"][name + "_" + subset] = {
                "trials": len(ix), "same_owners": len({trials[i].left_person for i in ix if trials[i].same}),
                "metrics": [compare_predictions(labels[ix], base[ix], pred[ix], counts, prior) for prior in (.5, .1)],
                "frozen_operating_point": operating(labels[ix], pred[ix], development[winner]["threshold"])}
    write_json(a.output, report)


if __name__ == "__main__":
    main()
