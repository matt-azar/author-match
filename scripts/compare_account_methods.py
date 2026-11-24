"""Compare two frozen account models on identical IRC trials and whole-window resamples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from author_match.accounts import _identities, _source_digest, compare_accounts
from author_match.experiment import write_json
from author_match.model import _FUNCTION_WORDS
from author_match.profile_io import load_profile_model
from author_match.profile_probability import posterior
from author_match.profile_trials import partition_people, read_linked_posts, sample_trials
from author_match.text import comparison_words

_STOP = frozenset(_FUNCTION_WORDS)


def content_cosine(left, right):
    """Lexical-overlap diagnostic that does not estimate a probability."""
    def bag(posts):
        return Counter(w for post in posts for w in comparison_words(post) if w not in _STOP)

    a, b = bag(left), bag(right)
    norm = sqrt(sum(c * c for c in a.values()) * sum(c * c for c in b.values()))
    return sum(count * b[word] for word, count in a.items()) / norm if norm else 0.0


@dataclass
class Predictions:
    evidence: list
    abstentions: Counter
    all_posts_used: int = 0
    posts_used: int = 0
    independent_posts: int = 0

    def record(self, prediction):
        estimated = prediction["status"] == "estimated"
        self.evidence.append(prediction["log_likelihood_ratio"] if estimated else 0.0)
        if not estimated:
            self.abstentions[prediction["reason"]] += 1
        histories = prediction["histories"]
        self.all_posts_used += int(estimated and all(
            h["used_posts"] == h["independent_posts"] for h in histories))
        self.posts_used += sum(h["used_posts"] for h in histories)
        self.independent_posts += sum(h["independent_posts"] for h in histories)

    def coverage(self):
        estimated = len(self.evidence) - sum(self.abstentions.values())
        return {"estimates": estimated, "coverage": estimated / len(self.evidence),
                "estimates_using_all_independent_posts": self.all_posts_used,
                "fraction_of_independent_posts_used_over_all_trials": (
                    self.posts_used / self.independent_posts if self.independent_posts else None),
                "inconclusive_reasons": dict(self.abstentions)}


def histories(trial, scenario):
    # sample_trials already seeded and sampled the writing. A fixed digest
    # provides one repeatable thinning independent of pair label and model.
    def first_post(posts):
        return (min(posts, key=lambda p: sha256(p.encode("utf-8")).digest()),)

    if scenario == "original":
        return trial.left, trial.right
    if scenario == "one_vs_history":
        return first_post(trial.left), trial.right
    if scenario == "one_vs_one":
        return first_post(trial.left), first_post(trial.right)
    raise ValueError("unknown history scenario")


def draw_windows(groups, *, seed, replicates):
    """Keep every occurrence of both endpoints in its original recording window."""
    unique, codes = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(unique), [1 / len(unique)] * len(unique), size=replicates)
    return weights[:, codes]


def _metrics(y, evidence, counts, prior):
    """One row per window bootstrap draw. Replicate zero contains all original trials."""
    positive_mass = counts[:, y].sum(axis=1)
    negative_mass = counts[:, ~y].sum(axis=1)
    # Rank scores once, with half credit for ties. Each replicate accumulates
    # its own weighted number of negatives below each positive score.
    levels, rank = np.unique(evidence, return_inverse=True)
    auc = np.empty(len(counts))
    for i, weight in enumerate(counts):
        pos = np.bincount(rank[y], weights=weight[y], minlength=len(levels))
        neg = np.bincount(rank[~y], weights=weight[~y], minlength=len(levels))
        auc[i] = (pos @ (np.cumsum(neg) - neg / 2)) / (positive_mass[i] * negative_mass[i])
    prob = posterior(evidence, prior)
    squared_error = (prob - y) ** 2
    log_loss = np.logaddexp(0, np.where(y, -1, 1) * (evidence + logit(prior)))

    def weighted(values):
        return (prior * (counts[:, y] @ values[y]) / positive_mass
                + (1 - prior) * (counts[:, ~y] @ values[~y]) / negative_mass)

    return {"roc_auc": auc, "brier": weighted(squared_error), "log_loss": weighted(log_loss)}


def compare_predictions(labels, baseline, candidate, bootstrap_counts, prior):
    y = np.array(labels, dtype=bool)
    if not y.any() or y.all():
        raise ValueError("both classes are required")
    valid = (bootstrap_counts[:, y].sum(axis=1) > 0) & (bootstrap_counts[:, ~y].sum(axis=1) > 0)
    counts = np.vstack((np.ones(len(y), dtype=int), bootstrap_counts[valid]))
    values = [_metrics(y, np.asarray(e, dtype=float), counts, prior) for e in (baseline, candidate)]
    diffs = {}
    for name, earlier in values[0].items():
        later = values[1][name]
        # Uniform sign convention. A positive difference always favors candidate.
        gain = later - earlier if name == "roc_auc" else earlier - later
        interval = None
        if valid.sum() >= max(2, len(valid) * 0.8):
            interval = np.quantile(gain[1:], [0.025, 0.975]).tolist()
        diffs[name] = {"baseline": float(earlier[0]), "candidate": float(later[0]),
                       "candidate_gain": float(gain[0]), "paired_gain_interval_95": interval}
    return {"prior": prior, "usable_bootstrap_replicates": int(valid.sum()), "metrics": diffs,
            "always_prior": {"roc_auc": 0.5, "brier": prior * (1 - prior),
                             "log_loss": float(-prior * np.log(prior) - (1 - prior) * np.log1p(-prior))}}


def lexical_matches(trials, windows, lexical, *, tolerance=0.05):
    """Within a fixed anchor and reference length, match close lexical scores without replacement."""
    strata = defaultdict(list)
    for i, trial in enumerate(trials):
        key = windows[i], trial.left_person, tuple(sorted(trial.left)), len(trial.right)
        strata[key].append(i)
    matches = []
    for indices in strata.values():
        pos = sorted((i for i in indices if trials[i].same), key=lambda i: tuple(sorted(trials[i].right)))
        neg = sorted((i for i in indices if not trials[i].same), key=lambda i: tuple(sorted(trials[i].right)))
        if not pos or not neg:
            continue
        distance = np.abs(np.subtract.outer([lexical[i] for i in pos], [lexical[j] for j in neg]))
        # Prioritize the number of allowed matches, then their total distance.
        # Even the sum of every allowed distance is below one forbidden-edge penalty.
        costs = np.where(distance <= tolerance, distance, len(indices) + 1.0)
        pi, ni = linear_sum_assignment(costs)
        matches.extend((pos[i], neg[j]) for i, j in zip(pi, ni) if distance[i, j] <= tolerance)
    return matches


def topic_sensitivity(trials, windows, lexical, predictions, *, seed, replicates):
    tolerance = 0.05
    matches = lexical_matches(trials, windows, lexical, tolerance=tolerance)
    pairs = np.array(matches, dtype=int).reshape(-1, 2)
    selected = pairs.ravel()
    matched_windows = [windows[i] for i in selected]
    result = {
        "selection": "Same anchor posts, owner, recording window and reference post count. "
                     "One positive and one negative matched without replacement within the cosine tolerance. "
                     "Model scores were unavailable to matching.",
        "max_allowed_cosine_difference": tolerance, "matched_positive_negative_pairs": len(matches),
        "trials": len(selected), "windows": len(set(matched_windows)),
        "distinct_linked_people": len({trials[i].left_person for i in selected}),
        "caveat": "Exploratory subset of the same pilot. Matching controls this lexical-overlap measure. "
                  "Unmeasured topic cues and differences in the amount of text can remain.",
    }
    if not len(matches):
        return result
    labels = [trials[i].same for i in selected]
    cosines = np.asarray(lexical)
    gaps = np.abs(cosines[pairs[:, 0]] - cosines[pairs[:, 1]])
    result["mean_absolute_cosine_difference"] = float(gaps.mean())
    result["max_observed_cosine_difference"] = float(gaps.max())
    counts = draw_windows(matched_windows, seed=seed, replicates=replicates if result["windows"] > 1 else 0)
    scores = [np.array(p.evidence) for p in predictions]
    comparison = compare_predictions(labels, scores[0][selected], scores[1][selected], counts, 0.5)
    result["pooled_roc_auc"] = {"lexical_only": float(roc_auc_score(labels, cosines[selected])),
                                **comparison["metrics"]["roc_auc"]}
    result["within_matched_pair_concordance"] = {}
    for name, values in zip(("lexical_only", "baseline", "candidate"), (cosines, *scores)):
        gap = values[pairs[:, 0]] - values[pairs[:, 1]]
        result["within_matched_pair_concordance"][name] = float(np.mean((gap > 0) + 0.5 * (gap == 0)))
    return result


def run(data, baseline_path, candidate_path, *, priors=(0.5, 0.1), replicates=1000, seed=91283):
    if replicates < 0:
        raise ValueError("bootstrap replicates must be nonnegative")
    for prior in priors:
        posterior(0, prior)
    models = [load_profile_model(path) for path in (baseline_path, candidate_path)]
    source_hash = _source_digest(data)
    for model in models:
        if model.label_basis != "linked_handles" or model.provenance["source_sha256"] != source_hash:
            raise ValueError("both checkpoints must be fitted for this linked-handle source")
    config = models[0].provenance
    if any(models[1].provenance[k] != config[k] for k in ("seed", "transfer")):
        raise ValueError("both checkpoints must use the same held-out partition")
    test = partition_people(read_linked_posts(data), seed=config["seed"], transfer=config["transfer"])["test"]
    for model in models:
        if _identities(test).intersection(model.fit_people):
            raise ValueError("a held-out person or handle contributed to model fitting")
    trials = sample_trials(test, label_basis="linked_handles", seed=config["seed"] + 2)
    by_person = {p.person: p.split_group for p in test}
    groups = [by_person[t.left_person] for t in trials]
    if len(set(groups)) < 2 or any(not g or by_person[t.right_person] != g for g, t in zip(groups, trials)):
        raise ValueError("paired uncertainty requires at least two independent recording windows and within-window trials")
    labels = [t.same for t in trials]
    counts = draw_windows(groups, seed=seed, replicates=replicates)
    result = {
        "source_sha256": source_hash, "checkpoint_sha256": {
            "baseline": _source_digest(baseline_path), "candidate": _source_digest(candidate_path)},
        "split": "test", "sampling_seed": config["seed"] + 2, "trials": len(trials),
        "positive_pairs": sum(labels), "distinct_linked_people": len({t.left_person for t in trials if t.same}),
        "writers": len({p for t in trials for p in (t.left_person, t.right_person)}), "windows": len(set(groups)),
        "bootstrap": {"method": "paired percentile bootstrap resampling entire recording windows",
                      "replicates": replicates, "seed": seed},
        "interpretation": [
            "The original IRC test results were visible during method development. This paired comparison is exploratory.",
            "The sparse scenarios thin the original trials before inference and reuse the same people and windows.",
            "Each model contributes the prior on an inconclusive pair. All losses use class weights for the stated prior.",
            "Lexical cosine excludes the model's list of common terms. It can capture both topic and writer vocabulary and yields ranks only.",
            "Positive gains favor the candidate. Intervals condition on the fixed checkpoints and sampled windows.",
        ], "scenarios": {},
    }
    for scenario in ("original", "one_vs_history", "one_vs_one"):
        predictions = [Predictions([], Counter()), Predictions([], Counter())]
        counts_by_side = []
        lexical = []
        for trial in trials:
            left, right = histories(trial, scenario)
            counts_by_side.append((len(left), len(right)))
            lexical.append(content_cosine(left, right))
            for model, storage in zip(models, predictions):
                storage.record(compare_accounts(model, left, right, prior=0.5))
        result["scenarios"][scenario] = {
            "content_word_cosine_roc_auc": float(roc_auc_score(labels, lexical)),
            "median_supplied_posts_left_right": np.median(counts_by_side, axis=0).tolist(),
            "unequal_length_pairs": sum(a != b for a, b in counts_by_side),
            "coverage": dict(zip(("baseline", "candidate"), (p.coverage() for p in predictions))),
            "all_pairs": [compare_predictions(labels, predictions[0].evidence, predictions[1].evidence,
                                               counts, prior) for prior in priors],
        }
        if scenario == "original":
            result["lexical_overlap_sensitivity"] = topic_sensitivity(
                trials, groups, lexical, predictions, seed=seed, replicates=replicates)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("linked_data", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", type=float, action="append", dest="priors",
                        help="target prevalence (repeatable, default: 0.5 and 0.1)")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.linked_data, args.baseline, args.candidate)}:
        parser.error("choose a separate output path")
    report = run(args.linked_data, args.baseline, args.candidate,
                 priors=args.priors or (0.5, 0.1), replicates=args.bootstrap)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
