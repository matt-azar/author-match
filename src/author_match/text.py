"""Prepare informal text without correcting the writer's spelling or typography."""

from __future__ import annotations

import html
import re
import unicodedata

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_QUOTE_OR_NOTE = re.compile(r"^[ \t]*(?:>|(?:&(?:amp;)*gt;))", re.I)
_SIGNATURE = re.compile(r"^(?:-- |sent from my (?:iphone|ipad|android)[.!]?)$", re.I)
_REPOST = re.compile(r"^\s*(?:RT|MT)\s+@[\w.]+\s*:", re.I)
_LINK = re.compile(r"!?\[([^\]\n]*)\]\(\s*(?:https?://|//)[^\s()]*(?:\([^()\s]*\)[^\s()]*)*\s*\)", re.I)
_URL = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.I)
_HANDLE = re.compile(r"(?<![\w@])(?:@[\w.-]*[\w-]|/?u/[\w-]+)", re.I)
_COMMUNITY = re.compile(r"(?<!\w)/?r/[a-z\d_]+", re.I)
_HASHTAG = re.compile(r"(?<!\w)#[\w]+")
_AUTOLINK = re.compile(r"<(?:https?://|www\.)[^\s<>]+>", re.I)
_HTML_QUOTE = re.compile(r"<blockquote\b[^>]*>.*?(?:</blockquote\s*>|$)", re.I | re.S)
MIN_WORDS = 4


def _strip_url(match: re.Match) -> str:
    # Punctuation written immediately after the URL still belongs to the post.
    value = match.group()
    end = len(value.rstrip(".,!?;:"))
    excess = value.count(")") - value.count("(")
    while excess > 0 and end and value[end - 1] == ")":
        end -= 1
        excess -= 1
    return value[end:]


def clean_text(text: str) -> str:
    """Remove attributed/linked material. Reapplying this operation is safe."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    while True:
        cleaned = _clean_once(text)
        if cleaned == text:
            return cleaned
        text = cleaned


def _clean_once(text: str) -> str:
    text = _HTML_QUOTE.sub("", html.unescape(text))
    if _REPOST.match(text):
        return ""

    lines = []
    fence = None
    for line in text.split("\n"):
        stripped = line.lstrip()
        if fence:
            if stripped.startswith(fence) and not stripped.strip(fence[0]).strip():
                fence = None
            continue
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:len(stripped) - len(stripped.lstrip(stripped[0]))]
            continue
        if _QUOTE_OR_NOTE.match(line):
            continue
        if _SIGNATURE.fullmatch(line.lstrip()):
            break
        lines.append(line)
    result = "\n".join(lines)
    result = _LINK.sub(r"\1", result)
    result = _AUTOLINK.sub("", result)
    result = _URL.sub(_strip_url, result)
    result = _HANDLE.sub("@", result)
    result = _COMMUNITY.sub("#", result)
    return result.strip()


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def comparison_words(text: str) -> list[str]:
    """Canonical words used to identify copied passages and post variants."""
    return words(text.casefold().replace("’", "'"))


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def mask_hashtags(text: str) -> str:
    return _HASHTAG.sub("#", text)


def evidence_reason(clean_left: str, clean_right: str) -> str | None:
    """Return a reason to abstain, independent of which text is on the left."""
    if not clean_left or not clean_right:
        return "empty"
    left = comparison_words(clean_left)
    right = comparison_words(clean_right)
    if min(len(left), len(right)) < MIN_WORDS:
        return "too_short"
    if left == right:
        return "copied_overlap"
    if min(len(set(left)), len(set(right))) < 2:
        return "repeated_words"
    if not any(t.isalpha() for t in clean_left) or not any(t.isalpha() for t in clean_right):
        return "no_prose"

    # Detect whole or mostly copied passages, including a short quote pasted
    # inside a much longer post. Sets keep this linear for repetitive input.
    def shingles(tokens: list[str]) -> set[tuple[str, ...]]:
        return set(zip(tokens, tokens[1:], tokens[2:], tokens[3:]))

    a, b = shingles(left), shingles(right)
    if len(a.intersection(b)) >= 0.70 * min(len(a), len(b)):
        return "copied_overlap"
    return None
