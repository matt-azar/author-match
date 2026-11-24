"""Extract local two-handle histories around explicit IRC system rename notices.

Source: Kummerfeld et al., ACL 2019 (CC BY 4.0), revision and retrieval
instructions in prepare_irc_pilot.py. Window-only identifiers are opaque. The
optional exclusion file contains opaque author digests used by that converter.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from prepare_irc_pilot import BOT, MESSAGE, account_id, canonical_text, conversations, irc_name, remove_address


RENAME = re.compile(r"^===\s+(\S+) is now known as (\S+)\s*$")
SYSTEM = re.compile(r"^===\s+(\S+)")
WINDOW_LINES = 300
MIN_WORDS = 3


@dataclass
class Candidate:
    window: str
    handles: tuple[str, str]
    histories: tuple[list, list]
    event: int

    @property
    def quality(self):
        return (min(map(len, self.histories)), sum(map(len, self.histories)))


def opaque(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:32]


def window_candidates(window, lines, threads, partners, counts):
    changes = []
    boundaries = defaultdict(list)
    messages = defaultdict(list)
    spellings = defaultdict(set)
    for number, line in enumerate(lines):
        rename = RENAME.fullmatch(line)
        if rename:
            a, b = map(irc_name, rename.groups())
            counts['rename_notices'] += 1
            boundaries[a].append(number)
            if a != b:
                boundaries[b].append(number)
                partners[a].add(b)
                partners[b].add(a)
                changes.append((number, a, b))
        elif marker := SYSTEM.match(line):
            # Join/leave/quit and other server notices interrupt continuity;
            # some raw release windows only record a subset of these events.
            boundaries[irc_name(marker[1])].append(number)
            if ' kick' in line.lower():
                # Different loggers report a kick with different argument order.
                boundaries[None].append(number)
        elif speech := MESSAGE.match(line):
            name = irc_name(speech[1])
            messages[name].append((number, speech[2]))
            spellings[name].add(speech[1])

    names = set(messages)
    for event, a, b in changes:
        if (a not in names or b not in names or any(
                BOT.fullmatch(n) or n.endswith('bot') for n in (a, b))):
            continue
        start, stop = max(0, event - WINDOW_LINES), min(len(lines), event + WINDOW_LINES + 1)
        for stream in (boundaries[a], boundaries[b], boundaries[None]):
            prev = bisect_left(stream, event) - 1
            following = bisect_right(stream, event)
            if prev >= 0:
                start = max(start, stream[prev] + 1)
            if following < len(stream):
                stop = min(stop, stream[following])
        if (any(start <= i < event for i, _ in messages[b])
                or any(event < i < stop for i, _ in messages[a])):
            continue
        aliases = sorted(spellings[a] | spellings[b], key=lambda x: (-len(x), x))
        alias = re.compile(r'(?<!\w)(?:' + '|'.join(map(re.escape, aliases)) + r')(?!\w)', re.I)
        groups = []
        for handle, lo, hi in ((a, start, event), (b, event + 1, stop)):
            posts = []
            for number, raw in messages[handle]:
                if lo <= number < hi:
                    body = alias.sub('', remove_address(raw, names)).strip()
                    if '\ufffd' not in body and len(re.findall(r"\b\w+\b", body)) >= MIN_WORDS:
                        posts.append((number, threads.get(number), body))
            groups.append(posts)
        if all(len(group) >= 2 for group in groups):
            yield Candidate(window, (a, b), tuple(groups), event)


def block_components(partners):
    """Use transitive connections only to *exclude* related observations."""
    components = {}
    for name in sorted(partners):
        if name in components:
            continue
        members = {name}
        pending = [name]
        while pending:
            unseen = partners[pending.pop()] - members
            members.update(unseen)
            pending.extend(unseen)
        group = frozenset(members)
        components.update(dict.fromkeys(group, group))
    return components


def drop_copies(candidate, occurrences):
    groups = []
    for group in candidate.histories:
        groups.append([p for p in group if len(occurrences[canonical_text(p[2])]) == 1])
    # A near copy within either history or spanning both would inflate the
    # amount of independent writing. Remove both sides of a suspected copy.
    rows = [post for group in groups for post in group]
    tokens = [re.findall(r'\w+', p[2].casefold()) for p in rows]
    drop = set()
    for i, left in enumerate(tokens):
        for j in range(i):
            right = tokens[j]
            if min(len(left), len(right)) < 3:
                continue
            if SequenceMatcher(None, left, right, autojunk=False).ratio() >= 0.85:
                drop.update((rows[i][0], rows[j][0]))
    return tuple([p for p in group if p[0] not in drop] for group in groups)


def collect(source):
    partners = defaultdict(set)
    counts = Counter()
    candidates = []
    index = source / 'data/list.ubuntu.train.txt'
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        ascii_path, anno, _ = line.split()
        path = source / ascii_path.replace('.ascii.txt', '.raw.txt')
        with path.open(encoding='utf-8', errors='replace') as stream:
            lines = [s.rstrip('\r\n') for s in stream]
        name = path.name.removesuffix('.raw.txt')
        candidates.extend(window_candidates(name, lines, conversations(source / anno), partners, counts))

    # A nickname can be reused by someone else in a different window. Never
    # use the graph as positive labels or concatenate across windows. Retain
    # at most one directly witnessed event per globally connected component.
    components = block_components(partners)
    occurrences = defaultdict(set)
    for c in candidates:
        for group in c.histories:
            for number, _, body in group:
                occurrences[canonical_text(body)].add((c.window, number))
    best = {}
    for c in candidates:
        c.histories = drop_copies(c, occurrences)
        if min(map(len, c.histories)) < 2:
            continue
        key = components[c.handles[0]]
        if key not in best or c.quality > best[key].quality:
            best[key] = c
    counts['usable_pairs'] = len(best)
    exclusions = sorted({account_id(h) for component in best for h in component})
    return sorted(best.values(), key=lambda c: (c.window, c.event)), counts, exclusions


def write_private(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            tmp = Path(stream.name)
            for line in lines:
                stream.write(line + '\n')
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def output_rows(candidates):
    for c in candidates:
        person_id = opaque('ubuntu', c.window, c.event)
        for handle, group in zip(c.handles, c.histories):
            for number, thread, body in group:
                post = {'id': opaque('post', c.window, number), 'person_id': person_id,
                        'handle': opaque('ubuntu', c.window, handle), 'text': body}
                if thread is not None:
                    post['thread'] = opaque('thread', c.window, thread)
                yield post


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--exclude-proxy-authors', type=Path, required=True,
                        help='write one Ubuntu pilot author digest per line for leakage exclusion')
    parser.add_argument('--exclude-proxy-windows', type=Path,
                        help='write original pilot post ID prefixes (window/) for conversation exclusion')
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.output, args.exclude_proxy_authors, args.exclude_proxy_windows) if p]
    if len(set(paths)) < len(paths):
        parser.error('use different paths for histories and exclusions')
    candidates, counts, exclusions = collect(args.source)
    posts = list(output_rows(candidates))
    write_private(args.exclude_proxy_authors, exclusions)
    if args.exclude_proxy_windows:
        write_private(args.exclude_proxy_windows, sorted({c.window + '/' for c in candidates}))
    write_private(args.output, (json.dumps(p, ensure_ascii=False) for p in posts))
    counts['posts'] = len(posts)
    counts['unique_handles'] = 2 * len(candidates)
    counts['excluded_proxy_author_digests'] = len(exclusions)
    counts['posts_with_annotated_thread'] = sum('thread' in p for p in posts)
    print(json.dumps(counts, sort_keys=True))


if __name__ == '__main__':
    main()
