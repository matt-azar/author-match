"""Versioned JSON checkpoints for the three CPU models.

Format version 1 fixes preprocessing, pair-feature order, TF–IDF settings and
the positive class (same author). Keep that inference contract when adding
future checkpoint versions. Only numeric parameters and vocabulary are loaded.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .bundles import BundlePolicy, MODES
from .metrics import LengthPolicy, Thresholds
from .model import METHODS, Verifier, _style

if TYPE_CHECKING:
    from .experiment import Artifact

_FORMAT = "author-match/cpu-v1"
_SUPPORT_FIELDS = {"same_pairs", "different_pairs", "eligible_same", "eligible_different"}
_AUTHOR_DIGEST = re.compile(r"[0-9a-f]{32}")


def _object(value, fields):
    if not isinstance(value, dict) or value.keys() != set(fields):
        raise ValueError(f"checkpoint requires fields {sorted(fields)}")
    return value


def _number(value):
    if type(value) not in (int, float):
        raise ValueError("expected a finite checkpoint number")
    try:
        if math.isfinite(value):
            return float(value)
    except OverflowError:
        pass
    raise ValueError("expected a finite checkpoint number")


def _array(values, size, name, *, dtype=float):
    if not isinstance(values, list) or len(values) != size:
        raise ValueError(f"wrong size for checkpoint {name}")
    with np.errstate(over="ignore"):
        result = np.asarray([_number(x) for x in values], dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite checkpoint {name}")
    return result


def _thresholds(value):
    _object(value, ("upper", "lower"))
    params = {name: None if score is None else _number(score) for name, score in value.items()}
    if any(score is not None and not 0 <= score <= 1 for score in params.values()):
        raise ValueError("checkpoint threshold outside score range")
    return Thresholds(**params)


def _policy(value):
    _object(value, ("overall", "by_length", "calibration_support"))
    by_length, support = value["by_length"], value["calibration_support"]
    bands = {"<10", "10-24", "25-49", "50-99", "100+"}
    if not isinstance(by_length, dict) or not by_length.keys() <= bands:
        raise ValueError("invalid checkpoint length bands")
    if not isinstance(support, dict) or not support.keys() <= by_length.keys():
        raise ValueError("invalid checkpoint calibration support")
    for counts in support.values():
        _object(counts, _SUPPORT_FIELDS)
        if any(type(x) is not int or x < 0 for x in counts.values()):
            raise ValueError("invalid checkpoint calibration count")
        if counts["eligible_same"] > counts["same_pairs"] or counts["eligible_different"] > counts["different_pairs"]:
            raise ValueError("eligible calibration counts exceed totals")
    return LengthPolicy(_thresholds(value["overall"]), {k: _thresholds(v) for k, v in by_length.items()}, support)


def _digests(values):
    if not isinstance(values, list) or any(not isinstance(x, str) or not _AUTHOR_DIGEST.fullmatch(x) for x in values):
        raise ValueError("checkpoint fit authors must be 128-bit lowercase hexadecimal digests")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("checkpoint fit authors must be sorted and unique")
    return tuple(values)


def _payload(artifact):
    model = artifact.verifier
    if type(model) is not Verifier or not model.fitted or model.method not in METHODS:
        raise ValueError("only a fitted word, char, or style CPU verifier can be exported")
    if not isinstance(artifact.thresholds, LengthPolicy):
        raise ValueError("a length-calibrated decision policy is required")
    vocab = model.vectorizer.vocabulary_
    linear = None
    if model.method == "style":
        scaler, regression = model.classifier.steps[0][1], model.classifier.steps[1][1]
        linear = {
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coefficients": regression.coef_[0].tolist(), "intercept": float(regression.intercept_[0]),
        }
    bundles = {}
    for mode, policy in getattr(artifact, "bundle_policies", {}).items():
        if mode not in MODES or mode != policy.mode:
            raise ValueError("unsupported checkpoint bundle mode")
        bundles[mode] = asdict(policy.thresholds)
    return {
        "format": _FORMAT, "method": model.method, "vocabulary": sorted(vocab, key=vocab.__getitem__),
        "idf": model.vectorizer.idf_.tolist() if vocab else [],
        "linear": linear, "policy": asdict(artifact.thresholds), "bundle_policies": bundles,
        "fit_authors": _digests(list(artifact.fit_authors)), "provenance": artifact.provenance,
    }


def export_cpu(artifact: Artifact, path):
    """Atomically write compact UTF-8 JSON, deterministically compressed for .gz."""
    value = _payload(artifact)
    # Use ASCII escapes for control/surrogate code points that can occur in a
    # character vocabulary. Each index has exactly one term in this ordered list.
    raw = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                     separators=(",", ":")).encode("ascii") + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            if path.suffix == ".gz":
                with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
                    compressed.write(raw)
            else:
                stream.write(raw)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _restore(value):
    from .experiment import Artifact

    _object(value, ("format", "method", "vocabulary", "idf", "linear", "policy",
                    "bundle_policies", "fit_authors", "provenance"))
    method = value["method"]
    if value["format"] != _FORMAT or method not in METHODS:
        raise ValueError("unknown author-match CPU checkpoint format or method")
    provenance = value["provenance"]
    if (not isinstance(provenance, dict) or provenance.get("method") != method
            or type(provenance.get("format_version")) is not int or provenance["format_version"] != 1
            or not isinstance(provenance.get("author_match_version"), str)):
        raise ValueError("inconsistent checkpoint provenance")
    policy = _policy(value["policy"])
    bundles = value["bundle_policies"]
    if not isinstance(bundles, dict) or not bundles.keys() <= MODES.keys():
        raise ValueError("unsupported checkpoint bundle mode")
    bundle_policies = {mode: BundlePolicy(mode, _policy(config)) for mode, config in bundles.items()}
    if "thresholds" in provenance and provenance["thresholds"] != asdict(policy):
        raise ValueError("checkpoint policy disagrees with provenance")
    vocab = value["vocabulary"]
    if (not isinstance(vocab, list) or any(not isinstance(t, str) or not t for t in vocab)
            or len(set(vocab)) != len(vocab)):
        raise ValueError("checkpoint vocabulary must contain unique nonempty terms")
    if method != "word" and any(len(t) != 3 for t in vocab):
        raise ValueError("expected a character trigram vocabulary")
    idf = _array(value["idf"], len(vocab), "idf", dtype=np.float32)
    if np.any(idf < 1):
        raise ValueError("invalid inverse document frequency")

    model = Verifier(method)
    if vocab:
        # The public IDF setter supports transferring weights onto a fresh
        # vectorizer with a fixed vocabulary. No serialized estimator is read.
        model.vectorizer.set_params(vocabulary={term: i for i, term in enumerate(vocab)})
        model.vectorizer.idf_ = idf
        if model.vectorizer.idf_.dtype != idf.dtype:
            # In sklearn 1.3/1.4 the public setter casts its sparse diagonal
            # to float64. Keep the training dtype at inclusive decision cutoffs.
            weights = model.vectorizer._tfidf
            weights._idf_diag = weights._idf_diag.astype(idf.dtype)
    else:
        model.vectorizer.vocabulary_ = {}

    linear = value["linear"]
    if method == "style":
        _object(linear, ("mean", "scale", "coefficients", "intercept"))
        n = 1 + 2 * len(_style(""))
        scaler = StandardScaler()
        scaler.n_features_in_ = n
        scaler.mean_ = _array(linear["mean"], n, "mean")
        scaler.scale_ = _array(linear["scale"], n, "scale")
        if np.any(scaler.scale_ <= 0):
            raise ValueError("invalid checkpoint style scale")
        reg = LogisticRegression()
        reg.n_features_in_ = n
        reg.coef_ = _array(linear["coefficients"], n, "coefficients")[None, :]
        reg.intercept_ = np.asarray([_number(linear["intercept"])])
        reg.classes_ = np.asarray([False, True])
        model.classifier = make_pipeline(scaler, reg)
    elif linear is not None:
        raise ValueError("unexpected classifier in a similarity-only checkpoint")
    model.fitted = True
    return Artifact(model, policy, provenance, bundle_policies=bundle_policies,
                    fit_authors=_digests(value["fit_authors"]))


def _json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite checkpoint JSON")
    return number


def load_cpu(path) -> Artifact:
    """Read only JSON fields for the supported CPU architectures."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream, parse_constant=_json_float, parse_float=_json_float)
    except (EOFError, gzip.BadGzipFile) as error:
        raise ValueError("invalid compressed CPU checkpoint") from error
    return _restore(value)
