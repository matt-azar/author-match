"""Train and evaluate calibrated comparisons of two independently supplied histories."""

from __future__ import annotations

import hashlib
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import sklearn

from . import __version__, adaptive_features
from .adaptive_calibration import AdaptiveCalibration, fit_adaptive_calibration
from .adaptive_ranker import AdaptiveRanker, fit_ranker
from .adaptive_trials import sample_adaptive_trials
from .data import load_split
from .profile_features import (FEATURE_VERSION, adaptive_pair_reason, independent_profiles, limit_profile,
                               make_profile, pair_features, pair_reason)
from .profile_io import ProfileArtifact
from .profile_probability import eval_report, fit_account_model, posterior, score
from .profile_trials import partition_people, proxy_posts, read_linked_posts, sample_trials


def _identifier(kind, value):
    return hashlib.sha256(f"account-profile-v1/{kind}/{value}".encode()).hexdigest()[:32]


def _identities(posts):
    return {_identifier("person", p.person) for p in posts} | {_identifier("handle", p.handle) for p in posts}


def _source_digest(path):
    path = Path(path)
    file = path / "manifest.json" if path.is_dir() else path
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sizes(a, b):
    first, second = sorted(((a.posts, a.words), (b.posts, b.words)))
    return [first[0], second[0], first[1], second[1]]


def _trial_stream(posts, basis, seed, sampler=sample_trials):
    # Many trials reuse an identical history. Bound the cache so this works
    # both for a small evaluation and for a longer training job.
    summarize = lru_cache(maxsize=256)(make_profile)
    for trial in sampler(posts, label_basis=basis, seed=seed):
        a, b = independent_profiles(summarize(trial.left), summarize(trial.right))
        yield trial, a, b


def _fit_rows(posts, basis, seed, adaptive=False):
    x, y, support = [], [], []
    index = {}
    excluded = Counter()
    sampler = sample_adaptive_trials if adaptive else sample_trials
    features = adaptive_features.pair_features if adaptive else pair_features
    check = adaptive_pair_reason if adaptive else pair_reason
    for trial, a, b in _trial_stream(posts, basis, seed, sampler):
        reason = check(a, b)
        if reason:
            excluded[reason] += 1
            continue
        x.append(features(a, b))
        y.append(trial.same)
        for owner in (trial.left_person, trial.right_person):
            index.setdefault(owner, len(index))
        support.append([*_sizes(a, b), int(trial.same), index[trial.left_person], index[trial.right_person]])
    if len(set(y)) < 2:
        raise ValueError(f"{basis}: need usable positive and negative account-history trials in each partition")
    return x, y, support, set(index), dict(excluded)


def train_accounts(data, *, seed=42, pretrain=None, exclude_proxy_handles=None,
                   exclude_proxy_windows=None, description=None, link_evidence=None, method="adaptive"):
    if method not in ("baseline", "adaptive"):
        raise ValueError("unknown account method")
    adaptive = method == "adaptive"
    data = Path(data)
    linked = data.is_file()
    basis = "linked_handles" if linked else "account_proxy"
    transfer = pretrain is not None
    if link_evidence is None:
        link_evidence = "provided_owner_labels" if linked else "split_account_history"
    if (linked and link_evidence not in {"provided_owner_labels", "same_session_rename"}
            or not linked and link_evidence != "split_account_history"):
        raise ValueError("link evidence must describe the supplied dataset")
    if transfer and not linked:
        raise ValueError("proxy pretraining requires a separate labeled handle-links JSONL file")
    if (exclude_proxy_handles or exclude_proxy_windows) and not transfer:
        raise ValueError("proxy source exclusions require --pretrain")
    if linked:
        all_posts = read_linked_posts(data)
        splits = partition_people(all_posts, seed=seed, transfer=transfer)
        cal = splits["calibration"]
        if transfer:
            train = proxy_posts(load_split(pretrain, "train")[0])
            # By default linked corpora use the same namespaced account IDs.
            # When the source anonymizes IDs separately, its mapping supplies
            # additional source-side IDs that must be excluded.
            blocked = {p.handle for p in all_posts} | {p.person for p in all_posts}
            if exclude_proxy_handles:
                blocked.update(Path(exclude_proxy_handles).read_text(encoding="utf-8").splitlines())
            windows = ()
            if exclude_proxy_windows:
                windows = tuple(x.strip() for x in Path(exclude_proxy_windows).read_text().splitlines() if x.strip())
            train = [p for p in train if p.handle not in blocked and p.person not in blocked
                     and not p.id.startswith(windows)]
        else:
            train = splits["train"] + splits["dev"]
    else:
        train = proxy_posts(load_split(data, "train")[0])
        cal = proxy_posts(load_split(data, "calibration")[0])
    if _identities(train) & _identities(cal):
        raise ValueError("overlapping writers or handles in training and calibration")
    train_basis = "account_proxy" if transfer else basis
    x, y, _, train_people, train_exclusions = _fit_rows(train, train_basis, seed, adaptive)
    xc, yc, rows, cal_people, cal_exclusions = _fit_rows(cal, basis, seed + 1, adaptive)
    effective_people = min(len({row[5] for row in rows if row[4] == 1}),
                           len({row[5] for row in rows if row[4] == 0}),
                           len({row[6] for row in rows if row[4] == 0}))
    if adaptive:
        model = fit_ranker(x, y, seed)
        support_data = fit_adaptive_calibration(model.margins(xc), rows)
        status = support_data["status"]
    else:
        model = fit_account_model(x, y, xc, yc, train_authors=train_people, calibration_authors=cal_people,
                                  calibration_effective_people=effective_people)
        support_data = {"rows": rows, "min_people": 6}
        status = model.calibration_status
    info = {
        "author_match_version": __version__, "sklearn_version": sklearn.__version__,
        "method": method, "feature_version": adaptive_features.FEATURE_VERSION if adaptive else FEATURE_VERSION, "seed": seed, "label_basis": basis, "link_evidence": link_evidence,
        "training_label_basis": train_basis, "transfer": transfer, "source_sha256": _source_digest(data),
        "training_authors": len(train_people), "calibration_authors": len(cal_people),
        "training_trials": {"same": sum(y), "different": len(y) - sum(y)},
        "calibration_trials": {"same": sum(yc), "different": len(yc) - sum(yc)},
        "training_exclusions": train_exclusions, "calibration_exclusions": cal_exclusions,
        "calibration_status": status,
    }
    info["min_distinct_writers_per_class_role" if adaptive else "effective_calibration_people"] = effective_people
    if description:
        info["label_description"] = description
    if transfer:
        info["pretrain_source_sha256"] = _source_digest(pretrain)
        if exclude_proxy_handles:
            info["exclusions_sha256"] = _source_digest(exclude_proxy_handles)
        if exclude_proxy_windows:
            info["excluded_windows_sha256"] = _source_digest(exclude_proxy_windows)
    return ProfileArtifact(model, basis, support_data, info,
                           tuple(sorted(_identities(train) | _identities(cal))))


class _Support:
    """Count independent calibration writers at comparable history sizes."""

    def __init__(self, support):
        self.rows = np.asarray(support["rows"], dtype=np.int64).reshape(-1, 7)
        self.minimum = support["min_people"]

    def counts(self, a, b):
        sizes = np.array(_sizes(a, b))
        # Broader word windows accommodate writing genres of different lengths
        # within an otherwise comparable amount of evidence.
        tolerance = np.array([2, 2, 3, 3])
        sample = self.rows[:, :4]
        mask = np.all((sample <= sizes * tolerance) & (sizes <= sample * tolerance), axis=1)
        selected = self.rows[mask]
        same, other = selected[selected[:, 4] == 1], selected[selected[:, 4] == 0]
        return {"same_people": len(np.unique(same[:, 5])),
                "different_left_people": len(np.unique(other[:, 5])),
                "different_right_people": len(np.unique(other[:, 6]))}


def _effective(a, b, support):
    # Select any subsample entirely by evidence quantity and stable content
    # hash before computing a match score. Never pick the most similar posts.
    counts = support.counts(a, b)
    if min(counts.values()) >= support.minimum:
        return a, b, counts
    # For very large exports, try deterministic representative subsets that
    # resemble calibration. Counts are tried in order of retained evidence.
    def sizes(n):
        values = [n]
        while values[-1] > 2:
            values.append(max(2, (values[-1] + 1) // 2))
        return values

    candidates = sorted(((i, j) for i in sizes(a.posts) for j in sizes(b.posts)
                         if (i, j) != (a.posts, b.posts)),
                        key=lambda ij: (min(ij), sum(ij)), reverse=True)
    cached_a, cached_b = {}, {}
    for i, j in candidates:
        if i not in cached_a:
            cached_a[i] = limit_profile(a, i)
        if j not in cached_b:
            cached_b[j] = limit_profile(b, j)
        left, right = cached_a[i], cached_b[j]
        # Removing posts can expose an otherwise diluted shared passage.
        if pair_reason(left, right):
            continue
        cons = support.counts(left, right)
        if min(cons.values()) >= support.minimum:
            return left, right, cons
    return a, b, counts


def _estimate(artifact, a, b, prior, support):
    # Validate the prior even when empty inputs require abstention.
    posterior(0.0, prior)
    a, b = independent_profiles(a, b)
    # Stable orientation keeps asymmetric evidence budgets symmetric under a
    # caller swapping accounts. Content digest breaks exact count/word ties.
    swap = (a.posts, a.words) > (b.posts, b.words)
    if not swap and (a.posts, a.words) == (b.posts, b.words):
        swap = a.order_key > b.order_key
    if swap:
        a, b = b, a
    original = [a, b]
    counts = diagnostics = result = None
    if isinstance(support, AdaptiveCalibration):
        reason = adaptive_pair_reason(a, b)
        if not reason:
            features = adaptive_features.pair_features(a, b)
            result = support.estimate(artifact.model.margin(features), _sizes(a, b), prior)
            diagnostics = {k: result[k] for k in ("sampling_precision", "support_weight",
                           "effective_calibration_pairs", "positive_writer_support", "negative_writer_support")}
            diagnostics["calibration_status"] = support.status
    else:
        reason = pair_reason(a, b)
        if not reason:
            a, b, counts = _effective(a, b, support)
            if min(counts.values()) < support.minimum:
                reason = "unsupported_history_sizes"
            elif artifact.model.calibration_status != "fitted":
                reason = "calibration_found_no_directional_evidence"
            else:
                result = score(artifact.model, pair_features(a, b), prior)
    used = [a, b]
    if swap:
        original.reverse()
        used.reverse()
    link_evidence = artifact.provenance.get("link_evidence", "provided_owner_labels"
                                            if artifact.label_basis == "linked_handles" else "split_account_history")
    event = {
        "same_session_rename": "Same operator under the IRC rename benchmark; interpreting this as same person "
                               "assumes both accounts have one operator and this calibration transfers.",
        "provided_owner_labels": "Same person as defined by the supplied ownership labels, assuming "
                                 "comparable writing, account use, and candidate selection.",
        "split_account_history": "Two histories from the same observed handle under the proxy benchmark.",
    }[link_evidence]
    return {
        "status": "inconclusive" if reason else "estimated",
        "reason": reason, "prior": prior, "label_basis": artifact.label_basis,
        "estimated_same_person_probability": (result["probability"] if result and artifact.label_basis == "linked_handles" else None),
        "same_account_proxy_probability": (result["probability"] if result and artifact.label_basis == "account_proxy" else None),
        "likelihood_ratio": None if result is None else float(np.exp(result["log_likelihood_ratio"])),
        "log_likelihood_ratio": None if result is None else result["log_likelihood_ratio"],
        "score_outside_calibration_range": None if result is None else result["extrapolated"],
        "histories": [{"received_posts": initial.received, "independent_posts": initial.posts,
                       "independent_words": initial.words, "used_posts": final.posts if result else 0,
                       "used_words": final.words if result else 0}
                      for initial, final in zip(original, used)],
        "comparable_calibration_writers": counts,
        "calibration_support": diagnostics,
        "link_evidence": link_evidence, "probability_interpretation": event,
        "label_description": artifact.provenance.get("label_description"),
    }


def _support(artifact):
    if isinstance(artifact.model, AdaptiveRanker):
        return AdaptiveCalibration.from_dict(artifact.support)
    return _Support(artifact.support)


def compare_accounts(artifact, left, right, *, prior=None):
    """When no prior is known, report evidence and labeled probability scenarios."""
    result = _estimate(artifact, make_profile(left), make_profile(right),
                       0.5 if prior is None else prior, _support(artifact))
    if prior is None:
        evidence = result["log_likelihood_ratio"]
        result.update(prior=None, estimated_same_person_probability=None, same_account_proxy_probability=None)
        result["probability_scenarios"] = [
            {"prior": p, "probability": posterior(evidence, p) if evidence is not None else None}
            for p in (0.01, 0.1, 0.5)]
    return result


def evaluate_accounts(data, artifact, *, split="test", prior, seed=None, bootstrap_samples=300):
    """Use the fitted method’s trial design and normal inference path."""
    from .profile_probability import quality_intervals, window_quality_intervals

    posterior(0.0, prior)
    if bootstrap_samples < 0:
        raise ValueError("bootstrap samples cannot be negative")
    data = Path(data)
    external = split == "external"
    if not external and split not in ("test", "calibration"):
        raise ValueError("account evaluation split must be test, calibration or external")
    if not external and _source_digest(data) != artifact.provenance["source_sha256"]:
        raise ValueError("dataset differs from model source; use --split external with independent data")
    sampling_seed = artifact.provenance["seed"] if seed is None else seed
    basis = "linked_handles" if data.is_file() else "account_proxy"
    if data.is_file():
        source = read_linked_posts(data)
        posts = source if external else partition_people(
            source, seed=artifact.provenance["seed"], transfer=artifact.provenance["transfer"])[split]
    else:
        posts = proxy_posts(load_split(data, split)[0])
    if split != "calibration" and _identities(posts) & set(artifact.fit_people):
        raise ValueError("evaluation contains writers or handles used in training or calibration")
    support = _support(artifact)
    reasons = Counter()
    labels, evidence, left_people, right_people, supported = [], [], [], [], []
    used_subset = 0
    adaptive = isinstance(artifact.model, AdaptiveRanker)
    sampler = sample_adaptive_trials if adaptive else sample_trials
    offset = 1 if split == "calibration" else 2
    for trial, a, b in _trial_stream(posts, basis, sampling_seed + offset, sampler):
        out = _estimate(artifact, a, b, prior, support)
        valid = out["status"] == "estimated"
        if not valid:
            reasons[out["reason"]] += 1
        elif any(side["used_posts"] < side["independent_posts"] for side in out["histories"]):
            used_subset += 1
        labels.append(trial.same)
        evidence.append(out["log_likelihood_ratio"] if valid else 0.)
        supported.append(valid)
        left_people.append(trial.left_person)
        right_people.append(trial.right_person)
    if len(set(labels)) < 2:
        raise ValueError("evaluation requires both positive and negative account pairs")
    groups = {p.person: p.split_group for p in posts}
    windows = [groups.get(p) for p in left_people]
    window_uncertainty = None
    if all(windows) and all(groups.get(right) == window for right, window in zip(right_people, windows)):
        window_uncertainty = window_quality_intervals(labels, evidence, windows, prior, seed=sampling_seed,
                                                      bootstrap_samples=bootstrap_samples)
    result = eval_report(labels, evidence, prior)
    supported = np.asarray(supported, dtype=bool)
    y = np.array(labels)
    eligible_report = (eval_report(y[supported], np.array(evidence)[supported], prior)
                       if len(set(y[supported])) == 2 else None)
    people = len(set(left_people) | set(right_people))
    return {
        "split": split, "model_label_basis": artifact.label_basis, "evaluation_label_basis": basis,
        "source_sha256": _source_digest(data), "people": people,
        "distinct_positive_people": len({p for p, is_same in zip(left_people, labels) if is_same}),
        "sampled_pairs": len(y), "returned_estimates": int(supported.sum()),
        "coverage": float(supported.mean()), "used_representative_subsets": used_subset,
        "trial_design": "adaptive" if adaptive else "baseline",
        "inconclusive_reasons": dict(reasons), "prior": prior,
        "all_pairs_using_prior_when_inconclusive": result, "estimated_pairs_only": eligible_report,
        "window_uncertainty": window_uncertainty,
        "quality_uncertainty": quality_intervals(labels, evidence, left_people, right_people, prior,
                                                 seed=sampling_seed, bootstrap_samples=bootstrap_samples),
    }
