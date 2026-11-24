"""Model selection, calibration, and inference over the same scoring path."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

from . import __version__
from .data import load_split
from .metrics import LengthPolicy, decision, fit_policy, report, thresholds_for
from .model import Verifier
from .text import clean_text, evidence_reason, word_count


@dataclass
class Artifact:
    verifier: object
    thresholds: LengthPolicy
    provenance: dict
    bundle_policies: dict = field(default_factory=dict)
    fit_authors: tuple = ()


def write_json(path, value):
    """Replace a completed output in one filesystem operation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as output:
        temporary = Path(output.name)
        try:
            json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_model(path, artifact):
    path = Path(path)
    if path.name.endswith((".json", ".json.gz")):
        from .portable import export_cpu
        return export_cpu(artifact, path)
    if path.suffix != ".joblib":
        raise ValueError("model path must end in .json, .json.gz, or .joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(artifact, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(path):
    # joblib files execute Python during loading. The CLI accepts a local, trusted artifact.
    path = Path(path)
    if path.name.endswith((".json", ".json.gz")):
        from .portable import load_cpu
        return load_cpu(path)
    if path.suffix != ".joblib":
        raise ValueError("model path must end in .json, .json.gz, or .joblib")
    artifact = joblib.load(path)
    if not isinstance(artifact, Artifact) or artifact.provenance.get("format_version") != 1:
        raise ValueError("unrecognized author-match artifact")
    return artifact


def manifest_digest(data):
    return hashlib.sha256((Path(data) / "manifest.json").read_bytes()).hexdigest()


def _author_id(author):
    return hashlib.sha256(b"author-match-writer-v1\0" + author.encode()).hexdigest()[:32]


def _evidence(posts, pairs):
    texts = {post.id: post.text for post in posts}
    return [evidence_reason(texts[pair.left], texts[pair.right]) for pair in pairs]


def _labeled(data, split):
    posts, pairs = load_split(data, split)
    labels = np.asarray([pair.same for pair in pairs], dtype=bool)
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{split} needs both same-author and different-author pairs; "
                         "use more independent posts and authors")
    return posts, pairs, labels


def _model(method, neural_options):
    if method == "star":
        from .neural import StarVerifier
        return StarVerifier(**neural_options)
    return Verifier(method=method)


def train(data, methods, *, false_match_rate=0.01, false_reject_rate=0.05,
          neural_options=None, bundle_modes=()):
    """Rank candidates on development writers, then calibrate only the winner."""
    train_posts, train_pairs, _ = _labeled(data, "train")
    dev_posts, dev_pairs, dev_labels = _labeled(data, "dev")
    dev_reasons = _evidence(dev_posts, dev_pairs)
    candidate_results = {}
    selected_key = None
    winner = None
    winner_method = None
    for method in methods:
        candidate = _model(method, neural_options or {}).fit(train_posts, train_pairs)
        scores = candidate.scores(dev_posts, dev_pairs)
        provisional = fit_policy(dev_posts, dev_pairs, scores, dev_reasons,
                                 false_match_rate, false_reject_rate)
        lengths = {p.id: word_count(p.text) for p in dev_posts}
        decisions = [decision(s, r, thresholds_for(min(lengths[p.left], lengths[p.right]), provisional))
                     for s, r, p in zip(scores, dev_reasons, dev_pairs)]
        accepts = np.asarray([d == "same_author" for d in decisions])
        recall = float(accepts[dev_labels].mean())
        auc = float(roc_auc_score(dev_labels, scores))
        eligible = np.asarray([reason is None for reason in dev_reasons])
        eligible_auc = (float(roc_auc_score(dev_labels[eligible], scores[eligible]))
                        if len(np.unique(dev_labels[eligible])) == 2 else None)
        candidate_results[method] = {
            "selection_recall": recall,
            "selection_false_match_rate": float(accepts[~dev_labels].mean()),
            "roc_auc": auc,
            "roc_auc_eligible": eligible_auc,
            "same_pairs": int(dev_labels.sum()),
            "different_pairs": int((~dev_labels).sum()),
            "thresholds": asdict(provisional),
        }
        key = (recall, eligible_auc if eligible_auc is not None else -1.0)
        if selected_key is None or key > selected_key:
            selected_key, winner, winner_method = key, candidate, method
    if winner is None:
        raise ValueError("choose at least one candidate model")

    cal_posts, cal_pairs, cal_labels = _labeled(data, "calibration")
    cal_reasons = _evidence(cal_posts, cal_pairs)
    cal_scores = winner.scores(cal_posts, cal_pairs)
    frozen = fit_policy(cal_posts, cal_pairs, cal_scores, cal_reasons,
                        false_match_rate, false_reject_rate)
    provenance = {
        "format_version": 1,
        "author_match_version": __version__,
        "sklearn_version": sklearn.__version__,
        "method": winner_method,
        "prepared_manifest_sha256": manifest_digest(data),
        "target_false_match_rate": false_match_rate,
        "target_false_reject_rate": false_reject_rate,
        "train_pairs": len(train_pairs),
        "calibration_pairs": len(cal_pairs),
        "calibration_same_pairs": int(cal_labels.sum()),
        "calibration_different_pairs": int((~cal_labels).sum()),
        "thresholds": asdict(frozen),
        "development_candidates": candidate_results,
    }
    if winner_method == "star":
        from .neural import STAR, STAR_REVISION
        provenance["encoder"] = ({"checkpoint_sha256": winner.checkpoint_sha256} if winner.checkpoint else
                                 {"checkpoint": STAR, "revision": STAR_REVISION})
    writers = {post.author for group in (train_posts, dev_posts, cal_posts) for post in group}
    artifact = Artifact(winner, frozen, provenance, fit_authors=tuple(sorted(map(_author_id, writers))))
    if bundle_modes:
        from .bundles import calibrate_bundles
        provenance["bundle_calibration"] = {}
        for mode in dict.fromkeys(bundle_modes):
            policy, summary = calibrate_bundles(winner, cal_posts, mode, false_match_rate=false_match_rate,
                                                false_reject_rate=false_reject_rate)
            artifact.bundle_policies[mode] = policy
            provenance["bundle_calibration"][mode] = {
                "authors": summary["eligible_authors"], "pairs": summary["overall"]["pairs"],
                "thresholds": asdict(policy.thresholds),
            }
    return artifact


def bundle_policy(artifact, mode):
    if mode not in artifact.bundle_policies:
        raise ValueError(f"model has no calibration for {mode}; train with --bundle-mode {mode}")
    return artifact.bundle_policies[mode]


def evaluate(data, split, artifact, *, bootstrap_samples=200, seed=42, mode="1x1"):
    digest = manifest_digest(data)
    if split != "external" and digest != artifact.provenance["prepared_manifest_sha256"]:
        raise ValueError("use the exact prepared dataset used for training, or prepare an independent corpus "
                         "with --external and evaluate --split external")
    posts, pairs = load_split(data, split)
    if split in {"external", "test"}:
        fitted = set(artifact.fit_authors)
        if not fitted:
            raise ValueError("artifact lacks writer fingerprints; retrain before evaluating unseen writers")
        if any(_author_id(author) in fitted for author in {p.author for p in posts}):
            raise ValueError("evaluation contains writer IDs used in training, development, or calibration")
    if mode == "1x1":
        if not pairs:
            raise ValueError(f"no pairs in {split} split")
        scores = artifact.verifier.scores(posts, pairs)
        reasons = _evidence(posts, pairs)
        result = report(posts, pairs, scores, reasons, artifact.thresholds,
                        bootstrap_samples=bootstrap_samples, seed=seed)
    else:
        from .bundles import evaluate_bundles
        result = evaluate_bundles(artifact.verifier, posts, bundle_policy(artifact, mode),
                                  bootstrap_samples=bootstrap_samples, seed=seed)
    result.update(split=split, model=artifact.provenance,
                  evaluated_manifest_sha256=digest)
    return result


def compare(artifact, left, right):
    left, right = clean_text(left), clean_text(right)
    reason = evidence_reason(left, right)
    # Quoted-only, copied, and empty inputs cannot supply an authorship score.
    score = None if reason else float(artifact.verifier.score_pair(left, right))
    lengths = [word_count(left), word_count(right)]
    cutoffs = thresholds_for(min(lengths), artifact.thresholds)
    label = "inconclusive" if reason else decision(score, None, cutoffs)
    if label == "inconclusive" and not reason:
        reason = "no_supported_threshold" if (
            cutoffs.upper is None and cutoffs.lower is None
        ) else "insufficient_score_evidence"
    return {"decision": label, "score": score, "reason": reason,
            "words": lengths,
            "method": artifact.provenance["method"],
            "model_version": artifact.provenance["author_match_version"]}
