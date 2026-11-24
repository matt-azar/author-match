"""Empirical operating thresholds and single-post authorship evaluation."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from math import isfinite

import numpy as np
from sklearn.metrics import roc_auc_score

from .data import length_band
from .text import word_count


_LENGTH_BANDS = ("<10", "10-24", "25-49", "50-99", "100+")


@dataclass(frozen=True)
class Thresholds:
    upper: float | None
    lower: float | None

    def __post_init__(self):
        for value in (self.upper, self.lower):
            if value is not None and not isfinite(value):
                raise ValueError("thresholds must be finite or None")
        if self.upper is not None and self.lower is not None and self.lower >= self.upper:
            raise ValueError("the lower threshold must be strictly below the upper threshold")


@dataclass(frozen=True)
class LengthPolicy:
    """Calibration curves fitted globally and separately by shorter-post length."""

    overall: Thresholds
    by_length: dict[str, Thresholds]
    calibration_support: dict[str, dict[str, int]] = field(default_factory=dict)

    def effective(self, band: str) -> Thresholds:
        group = self.by_length.get(band)
        if group is None:
            return Thresholds(None, None)
        upper = None
        lower = None
        if self.overall.upper is not None and group.upper is not None:
            upper = max(self.overall.upper, group.upper)
        if self.overall.lower is not None and group.lower is not None:
            lower = min(self.overall.lower, group.lower)
        return Thresholds(upper, lower)


def thresholds_for(length: int, policy: LengthPolicy | Thresholds) -> Thresholds:
    """Effective policy for the shorter cleaned input, measured in words."""
    if isinstance(policy, Thresholds):
        return policy
    return policy.effective(length_band(length))


def _pair_length_bins(posts, pairs):
    lengths = {p.id: word_count(p.text) for p in posts}
    if len(lengths) != len(posts):
        raise ValueError("duplicate post id")
    bins = {name: [] for name in _LENGTH_BANDS}
    for i, pair in enumerate(pairs):
        band = length_band(min(lengths[pair.left], lengths[pair.right]))
        bins[band].append(i)
    return bins


def _observations(scores, labels, reasons):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    reasons = list(reasons)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels) or len(scores) != len(reasons):
        raise ValueError("scores, labels and reasons must be one-dimensional and equal in length")
    if not np.isfinite(scores).all():
        raise ValueError("all scores must be finite")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    eligible = np.fromiter((reason is None for reason in reasons), dtype=bool, count=len(reasons))
    return scores, labels.astype(bool), eligible


def _select_upper(correct_scores, wrong_scores, total_wrong, rate):
    """Lowest true-class score whose inclusive upper tail meets the error budget."""
    budget = Decimal(str(rate))
    if not budget.is_finite() or not 0 <= budget < 1:
        raise ValueError("error rates must be finite and in [0, 1)")
    # A zero budget disables that decision. At a positive target require enough
    # eligible observations from the opposing class to resolve one error.
    if budget * int(len(wrong_scores)) < 1 or not len(correct_scores):
        return None
    allowed = int(budget * int(total_wrong))
    wrong = np.sort(wrong_scores)
    candidates = np.unique(correct_scores)
    wrong_at_or_above = len(wrong) - np.searchsorted(wrong, candidates, side="left")
    allowed_candidates = candidates[wrong_at_or_above <= allowed]
    return float(allowed_candidates[0]) if len(allowed_candidates) else None


def calibrate(scores, labels, reasons, false_match_rate=0.01, false_reject_rate=0.05):
    """Fit two empirical operating points using reserved calibration pairs.

    Error budgets use all examples of the opposing class, including abstentions.
    A call is disabled if its positive error target cannot resolve a single
    error among eligible opposing pairs. These are sample operating points,
    not population error guarantees.
    """
    scores, labels, eligible = _observations(scores, labels, reasons)
    return _fit_thresholds(scores, labels, eligible, false_match_rate, false_reject_rate)


def _fit_thresholds(scores, labels, eligible, false_match_rate, false_reject_rate):
    positive = scores[eligible & labels]
    negative = scores[eligible & ~labels]
    upper = _select_upper(positive, negative, (~labels).sum(), false_match_rate)
    # Give the false-match operating point priority if both valid regions overlap.
    candidate_negative = negative if upper is None else negative[negative < upper]
    lower_negative = _select_upper(-candidate_negative, -positive, labels.sum(), false_reject_rate)
    return Thresholds(upper, None if lower_negative is None else -lower_negative)


def fit_policy(posts, pairs, scores, reasons, false_match_rate=0.01, false_reject_rate=0.05):
    """Fit operating points without borrowing evidence from longer posts."""
    posts, pairs = list(posts), list(pairs)
    scores, labels, eligible = _observations(scores, [p.same for p in pairs], reasons)
    overall = _fit_thresholds(scores, labels, eligible, false_match_rate, false_reject_rate)
    by_length, support = {}, {}
    for band, indices in _pair_length_bins(posts, pairs).items():
        group = np.asarray(indices, dtype=np.intp)
        y, ok = labels[group], eligible[group]
        by_length[band] = _fit_thresholds(scores[group], y, ok, false_match_rate, false_reject_rate)
        support[band] = {
            "same_pairs": int(y.sum()), "different_pairs": int((~y).sum()),
            "eligible_same": int((y & ok).sum()), "eligible_different": int((~y & ok).sum()),
        }
    return LengthPolicy(overall, by_length, support)


def decision(score, reason, thresholds):
    if not isfinite(score):
        raise ValueError("score must be finite")
    if reason is not None:
        return "inconclusive"
    if thresholds.upper is not None and score >= thresholds.upper:
        return "same_author"
    if thresholds.lower is not None and score <= thresholds.lower:
        return "different_author"
    return "inconclusive"


def _ratio(num, den):
    return float(num / den) if den else None


def _rates(cells):
    # Rows are true different / same. Columns are called different / inconclusive / same.
    tn, abstained_different, fp, fn, abstained_same, tp = cells
    negatives = tn + abstained_different + fp
    positives = fn + abstained_same + tp
    decided = tn + fp + fn + tp
    return {
        "recall": _ratio(tp, positives),
        "false_match_rate": _ratio(fp, negatives),
        "false_reject_rate": _ratio(fn, positives),
        "coverage": _ratio(decided, positives + negatives),
        "error_among_decided": _ratio(fp + fn, decided),
    }


def _summary(labels, scores, eligible, outcomes, authors, reasons, indices):
    group = np.asarray(indices, dtype=np.intp)
    y = labels[group]
    counts = np.bincount(outcomes[group], minlength=6).tolist()
    tn, abstained_different, fp, fn, abstained_same, tp = counts
    positive_count = int(y.sum())
    auc = None if positive_count in (0, len(group)) else float(roc_auc_score(y, scores[group]))
    scorable = group[eligible[group]]
    eligible_y = labels[scorable]
    eligible_auc = (float(roc_auc_score(eligible_y, scores[scorable]))
                    if 0 < eligible_y.sum() < len(scorable) else None)
    return {
        "pairs": len(group),
        "eligible_pairs": len(scorable),
        "authors": len(np.unique(authors[group])),
        "same_pairs": positive_count,
        "different_pairs": len(group) - positive_count,
        "decisions": {
            "true_different": {"different_author": tn, "inconclusive": abstained_different, "same_author": fp},
            "true_same": {"different_author": fn, "inconclusive": abstained_same, "same_author": tp},
        },
        "evidence_exclusions": dict(Counter(reasons[i] for i in indices if reasons[i] is not None)),
        "roc_auc": auc,
        "roc_auc_eligible": eligible_auc,
        **_rates(counts),
    }


def _author_bootstrap(authors, outcomes, repetitions, seed):
    """Resample authors, giving a pair one multiplier per distinct endpoint."""
    if not isinstance(repetitions, int) or repetitions < 0:
        raise ValueError("bootstrap_samples must be a non-negative integer")
    result = {
        "method": "Percentile author bootstrap, resampling authors across both endpoints of a pair",
        "level": 0.95,
        "repetitions": repetitions,
        "seed": seed,
        "intervals": {},
        "note": (
            "Each interval resamples authors represented in that group and conditions on the fitted model, "
            "thresholds and observed author pairs. "
            "Missing intervals indicate insufficient or constant bootstrap estimates. "
            "A sample rate of zero does not establish a population rate of zero."
        ),
    }
    if repetitions < 2 or len(authors) == 0:
        return result
    _, inverse = np.unique(authors.ravel(), return_inverse=True)
    a, b = inverse.reshape(-1, 2).T
    n_authors = int(inverse.max()) + 1
    if n_authors < 2:
        return result
    estimates = defaultdict(list)
    rng = np.random.default_rng(seed)
    same_endpoint = a == b
    for _ in range(repetitions):
        multiplicity = np.bincount(rng.integers(0, n_authors, n_authors), minlength=n_authors)
        weight = multiplicity[a] * np.where(same_endpoint, 1, multiplicity[b])
        cells = np.bincount(outcomes, weights=weight, minlength=6)
        for name, value in _rates(cells).items():
            if value is not None:
                estimates[name].append(value)
    for name, values in estimates.items():
        if len(values) >= max(2, repetitions * 0.8) and min(values) < max(values):
            result["intervals"][name] = np.quantile(values, (0.025, 0.975)).tolist()
    return result


def report(posts, pairs, scores, reasons, thresholds, *, bootstrap_samples=200, seed=42):
    """Measure a frozen policy on untouched pairs. Missing denominators yield None.

    AUC ranks all raw scores even if the decision policy abstains. Decision
    rates use all examples of the relevant true class. Each length bin uses
    the shorter post's word count.
    """
    posts, pairs, reasons = list(posts), list(pairs), list(reasons)
    scores, labels, eligible = _observations(scores, [pair.same for pair in pairs], reasons)
    length_bins = _pair_length_bins(posts, pairs)
    post_info = {p.id: (p.author, getattr(p, "timestamp", None)) for p in posts}
    authors = np.empty((len(pairs), 2), dtype=object)
    conditions = defaultdict(list)
    time_gaps = defaultdict(list)
    for i, pair in enumerate(pairs):
        left_author, left_time = post_info[pair.left]
        right_author, right_time = post_info[pair.right]
        authors[i] = (left_author, right_author)
        conditions[pair.condition].append(i)
        gap_name = "unknown"
        if left_time is not None and right_time is not None:
            days = abs(left_time - right_time) / 86400
            if days < 30:
                gap_name = "<30d"
            elif days < 180:
                gap_name = "30-179d"
            else:
                gap_name = "180d+"
        time_gaps[gap_name].append(i)
    # A contiguous six-way index makes the same confusion table available to
    # subgroup reports and each draw of the two-endpoint author bootstrap.
    calls = np.ones(len(pairs), dtype=np.int8)
    recorded = asdict(thresholds)
    if isinstance(thresholds, LengthPolicy):
        effective = {band: thresholds.effective(band) for band in length_bins}
        recorded["effective_by_length"] = {band: asdict(value) for band, value in effective.items()}
    else:
        effective = dict.fromkeys(length_bins, thresholds)
    for band, indices in length_bins.items():
        group = np.asarray(indices, dtype=np.intp)
        group = group[eligible[group]]
        rule = effective[band]
        if rule.upper is not None:
            calls[group[scores[group] >= rule.upper]] = 2
        if rule.lower is not None:
            calls[group[scores[group] <= rule.lower]] = 0
    outcomes = 3 * labels.astype(np.int8) + calls

    def summarize(indices):
        return _summary(labels, scores, eligible, outcomes, authors, reasons, indices)

    overall_uncertainty = _author_bootstrap(authors, outcomes, bootstrap_samples, seed)

    def summarize_slice(indices):
        summary = summarize(indices)
        if len(indices) == len(pairs):
            summary["intervals_95"] = overall_uncertainty["intervals"]
        elif summary["authors"] >= 2:
            uncertainty = _author_bootstrap(authors[indices], outcomes[indices], bootstrap_samples, seed)
            summary["intervals_95"] = uncertainty["intervals"]
        else:
            summary["intervals_95"] = {}
        return summary

    return {
        "thresholds": recorded,
        "overall": summarize(range(len(pairs))),
        "by_length": {name: summarize_slice(indices) for name, indices in length_bins.items()},
        "by_condition": {name: summarize_slice(indices) for name, indices in sorted(conditions.items())},
        "by_time_gap": ({name: summarize_slice(indices) for name, indices in sorted(time_gaps.items())}
                        if any(name != "unknown" for name in time_gaps) else {}),
        "uncertainty": overall_uncertainty,
    }
