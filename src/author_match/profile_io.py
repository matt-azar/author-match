"""Read one account's posts and persist calibrated profile models as data-only JSON."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import gzip
import json
import os
from pathlib import Path
import tempfile

from . import adaptive_features
from .adaptive_calibration import AdaptiveCalibration
from .adaptive_ranker import AdaptiveRanker
from .portable import _digests, _json_float
from .profile_features import FEATURE_COUNT, FEATURE_VERSION
from .profile_probability import AccountModel

_FORMAT = "author-match/profile-v1"
_ADAPTIVE_FORMAT = "author-match/profile-v2"
_LABEL_BASES = ("account_proxy", "linked_handles")
_HANDLE_KEYS = ("handle", "author", "speaker", "user")


@dataclass
class ProfileArtifact:
    model: AccountModel | AdaptiveRanker
    label_basis: str
    support: dict
    provenance: dict
    fit_people: tuple[str, ...] = ()


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field in JSON object")
        result[key] = value
    return result


def _decode_json(stream):
    return json.load(stream, object_pairs_hook=_unique_fields, parse_float=_json_float, parse_constant=_json_float)


def _calibration_support(value):
    if (not isinstance(value, dict) or not isinstance(value.get("rows"), list)
            or type(value.get("min_people")) is not int or value["min_people"] <= 0):
        raise ValueError("invalid account calibration support")
    for row in value["rows"]:
        if (not isinstance(row, (list, tuple)) or len(row) != 7
                or any(type(i) is not int or i > 2**63 - 1 for i in row)):
            raise ValueError("each calibration support row requires seven int64 values")
        posts_a, posts_b, words_a, words_b, label, person_a, person_b = row
        if (min(posts_a, posts_b, words_a, words_b) <= 0 or (posts_a, words_a) > (posts_b, words_b)
                or label not in (0, 1) or min(person_a, person_b) < 0):
            raise ValueError("invalid calibration support counts, label, or participant indices")
    return value


def _profile_artifact(value):
    keys = {"format", "model", "label_basis", "support", "provenance", "fit_people"}
    if not isinstance(value, dict) or value.keys() != keys or value["format"] not in (_FORMAT, _ADAPTIVE_FORMAT):
        raise ValueError("unrecognized account profile checkpoint schema")
    if value["label_basis"] not in _LABEL_BASES:
        raise ValueError("unknown account profile label basis")
    adaptive = value["format"] == _ADAPTIVE_FORMAT
    model_type = AdaptiveRanker if adaptive else AccountModel
    model = value["model"]
    if not isinstance(model, dict) or model.keys() != {f.name for f in fields(model_type) if f.init}:
        raise ValueError("invalid account model fields")
    provenance = value["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("account profile provenance must be a JSON object")
    version = provenance.get("feature_version")
    if type(version) is not int or version != (adaptive_features.FEATURE_VERSION if adaptive else FEATURE_VERSION):
        raise ValueError("unsupported account profile feature version")
    if provenance.get("label_basis", value["label_basis"]) != value["label_basis"]:
        raise ValueError("account label basis disagrees with provenance")
    try:
        fitted = model_type(**model)
    except OverflowError as error:
        raise ValueError("out-of-range account model parameter") from error
    if fitted.feature_count != (adaptive_features.FEATURE_COUNT if adaptive else FEATURE_COUNT):
        raise ValueError("account model width disagrees with the profile feature definition")
    if adaptive:
        AdaptiveCalibration.from_dict(value["support"])
    else:
        _calibration_support(value["support"])
    return ProfileArtifact(fitted, value["label_basis"], value["support"],
                           provenance, _digests(value["fit_people"]))


def _checkpoint_path(path):
    path = Path(path)
    if not path.name.endswith((".json", ".json.gz")):
        raise ValueError("account profile checkpoints require .json or .json.gz")
    return path


def save_profile_model(path, artifact: ProfileArtifact):
    """Publish one complete data-only model, replacing any prior version atomically."""
    path = _checkpoint_path(path)
    if not isinstance(artifact, ProfileArtifact) or type(artifact.model) not in (AccountModel, AdaptiveRanker):
        raise ValueError("expected a fitted account profile model")
    fmt = _ADAPTIVE_FORMAT if isinstance(artifact.model, AdaptiveRanker) else _FORMAT
    payload = {"format": fmt, **asdict(artifact)}
    payload["fit_people"] = list(artifact.fit_people)
    _profile_artifact(payload)
    raw = json.dumps(payload, allow_nan=False, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")).encode("ascii") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        with temp_path.open("wb") as stream:
            if path.suffix == ".gz":
                with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
                    compressed.write(raw)
            else:
                stream.write(raw)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_profile_model(path) -> ProfileArtifact:
    path = _checkpoint_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            value = _decode_json(stream)
    except (EOFError, gzip.BadGzipFile) as exc:
        raise ValueError("invalid compressed account profile checkpoint") from exc
    return _profile_artifact(value)


def _handle(record):
    values = []
    for key in _HANDLE_KEYS:
        value = record.get(key)
        if value is None:
            continue
        if type(value) is int:
            value = str(value)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("account identifiers must be nonempty strings")
        values.append(value.strip())
    if len(set(values)) > 1:
        raise ValueError("a post has conflicting handle fields")
    return values[0] if values else None


def _post(row, inherited=None):
    if isinstance(row, str):
        return inherited, row
    if not isinstance(row, Mapping) or not isinstance(row.get("text"), str):
        raise ValueError("each post must be a string or an object with a string text field")
    owner = _handle(row)
    if inherited is not None and owner is not None and owner != inherited:
        raise ValueError("post handle conflicts with the enclosing profile handle")
    return owner if owner is not None else inherited, row["text"]


def _select_posts(rows, selector, *, inherited=None):
    posts = []
    owners = set()
    saw_row = False
    for row in rows:
        owner, text = _post(row, inherited)
        saw_row = True
        if selector is not None:
            if owner == selector:
                posts.append(text)
        else:
            owners.add(owner)
            if len(owners) > 1:
                raise ValueError("history contains mixed handles or ambiguous unlabeled posts; select one handle")
            posts.append(text)
    if selector is not None and not posts and (saw_row or inherited is not None and inherited != selector):
        raise ValueError("selected handle is missing from history")
    return posts


def _json_lines(stream):
    for line_number, line in enumerate(stream, start=1):
        if line.strip():
            try:
                yield json.loads(line, object_pairs_hook=_unique_fields,
                                 parse_float=_json_float, parse_constant=_json_float)
            except ValueError as exc:
                raise ValueError(f"invalid history JSON at line {line_number}: {exc}") from exc


def read_history(path, handle=None) -> list[str]:
    """Read original writing from one side. Each row/file represents one post.

    With a selector, include only records explicitly assigned to that exact
    handle or its enclosing profile. Without one, all records must agree on
    their account, including whether they have an account label at all.
    """
    path = Path(path)
    if handle is not None:
        if not isinstance(handle, str) or not handle.strip():
            raise ValueError("handle selector must be a nonempty string")
        handle = handle.strip()
    if path.is_dir():
        files = (p for p in sorted(path.glob("*.txt")) if p.is_file())
        return _select_posts((p.read_text(encoding="utf-8-sig") for p in files), handle)
    if path.suffix == ".txt":
        text = path.read_text(encoding="utf-8-sig")
        return _select_posts([text] if text.strip() else [], handle)

    name = path.name.removesuffix(".gz")
    if not name.endswith((".json", ".jsonl")):
        raise ValueError("history must be JSON, JSONL, gzipped JSON/JSONL, or UTF-8 text")
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8-sig") as stream:
            if name.endswith(".jsonl"):
                return _select_posts(_json_lines(stream), handle)
            data = stream.read()
    except (EOFError, gzip.BadGzipFile) as error:
        raise ValueError("invalid compressed account history") from error
    if not data.strip():
        return []
    value = json.loads(data, object_pairs_hook=_unique_fields,
                       parse_float=_json_float, parse_constant=_json_float)
    if isinstance(value, dict) and "posts" in value:
        if not isinstance(value["posts"], list):
            raise ValueError("profile posts must be a JSON array")
        return _select_posts(value["posts"], handle, inherited=_handle(value))
    if isinstance(value, list):
        return _select_posts(value, handle)
    if isinstance(value, dict) and "text" in value:
        return _select_posts([value], handle)
    raise ValueError("expected a JSON array of posts or a profile object with a posts field")
