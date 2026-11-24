"""Build same-window control histories for the Ubuntu IRC rename experiment.

Source: Kummerfeld et al., ACL 2019 (CC BY 4.0). Use the published revision in
prepare_irc_pilot.py. Keep all generated data and sidecars local to the study.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from prepare_irc_handle_changes import (
    MIN_WORDS, RENAME, SYSTEM, WINDOW_LINES, collect, drop_copies, opaque, write_private,
)
from prepare_irc_pilot import BOT, MESSAGE, account_id, canonical_text, conversations, irc_name, remove_address


@dataclass
class History:
    window: str
    handles: tuple
    histories: tuple
    event: int
    role: str


def index_handles(source):
    windows = defaultdict(set)
    renamed = set()
    paths = {}
    for line in (source / 'data/list.ubuntu.train.txt').read_text().splitlines():
        if not line.strip():
            continue
        ascii_path, annotation, _ = line.split()
        raw = source / ascii_path.replace('.ascii.txt', '.raw.txt')
        window = raw.name.removesuffix('.raw.txt')
        paths[window] = raw, source / annotation
        with raw.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                if rename := RENAME.fullmatch(line.rstrip('\r\n')):
                    for nick in rename.groups():
                        renamed.add(irc_name(nick))
                        windows[irc_name(nick)].add(window)
                elif match := MESSAGE.match(line) or SYSTEM.match(line):
                    windows[irc_name(match[1])].add(window)
    return paths, windows, renamed


def mask_names(names):
    # Preserve an IRC name's equivalent RFC1459 spellings, including brackets.
    # The same window-wide mask is applied to both classes. Common English
    # words that are also handles can be removed as a consequence.
    equivalents = {'{': r'[\[{]', '}': r'[\]}]', '|': r'[\\|]', '~': r'[\^~]'}
    alternatives = []
    for nick in sorted(names, key=lambda n: (-len(n), n)):
        alternatives.append(''.join(equivalents.get(c, re.escape(c)) for c in nick))
    return re.compile(r'(?<!\w)(?:' + '|'.join(alternatives) + r')(?!\w)', re.IGNORECASE)


def window_histories(window, paths, links, name_windows, renamed):
    raw, annotation = paths[window]
    membership = conversations(annotation)
    with raw.open(encoding='utf-8', errors='replace') as stream:
        lines = [line.rstrip('\r\n') for line in stream]
    speeches = defaultdict(list)
    boundaries = defaultdict(list)
    display_names = set()
    for num, line in enumerate(lines):
        if match := MESSAGE.match(line):
            nick = irc_name(match[1])
            display_names.add(nick)
            speeches[nick].append((num, match[2]))
        elif rename := RENAME.fullmatch(line):
            display_names.update(map(irc_name, rename.groups()))
        elif marker := SYSTEM.match(line):
            boundaries[irc_name(marker[1])].append(num)
            if ' kick' in line.lower():
                boundaries[None].append(num)
    mask = mask_names(display_names)

    def cleaned(posts):
        kept = []
        for num, thread, original in posts:
            body = mask.sub('', remove_address(original, display_names)).strip()
            if '\ufffd' not in body and len(re.findall(r'\b\w+\b', body)) >= MIN_WORDS:
                kept.append((num, thread, body))
        return kept

    for link in links:
        yield History(window, link.handles, tuple(cleaned(group) for group in link.histories),
                      link.event, 'observed_rename')
    for name, posts in speeches.items():
        if (name in renamed or len(name_windows[name]) != 1 or name.endswith('bot') or BOT.fullmatch(name)):
            continue
        for event in sorted(link.event for link in links):
            start, stop = event - WINDOW_LINES, event + WINDOW_LINES + 1
            for boundary in boundaries[name] + boundaries[None]:
                if boundary < event:
                    start = max(start, boundary + 1)
                elif boundary > event:
                    stop = min(stop, boundary)
            group = cleaned((i, membership.get(i), s) for i, s in posts if start <= i < stop)
            if group and group[0][0] < event < group[-1][0]:
                yield History(window, (name,), (group,), event, 'same_window_control')


def build(source):
    links, _, excluded = collect(source)
    by_window = defaultdict(list)
    for link in links:
        by_window[link.window].append(link)
    paths, name_windows, renamed = index_handles(source)
    all_histories = [history for window, links_in_window in sorted(by_window.items())
                     for history in window_histories(window, paths, links_in_window, name_windows, renamed)]
    occurrences = defaultdict(set)
    for h in all_histories:
        for group in h.histories:
            for i, _, text in group:
                occurrences[canonical_text(text)].add((h.window, i))
    viable = defaultdict(list)
    for h in all_histories:
        h.histories = drop_copies(h, occurrences)
        if min(map(len, h.histories)) < 2:
            continue
        if h.role == 'same_window_control' and not h.histories[0][0][0] < h.event < h.histories[0][-1][0]:
            continue
        viable[h.window].append(h)
    selected = []
    for window, histories in sorted(viable.items()):
        witnessed = [h for h in histories if h.role == 'observed_rename']
        events = {h.event for h in witnessed}
        if not events:
            continue
        # Controls must write on both sides of a retained rename. Prefer those
        # whose nearest messages bracket that change most closely. A single
        # control may be eligible at more than one event in the same window.
        controls = [h for h in histories if h.role == 'same_window_control' and h.event in events]
        def gap(h):
            indices = [p[0] for p in h.histories[0]]
            return min(i for i in indices if i > h.event) - max(i for i in indices if i < h.event)
        unique = {}
        for h in sorted(controls, key=lambda x: (gap(x), opaque(window, x.handles), x.event)):
            unique.setdefault(h.handles[0], h)
        if len(unique) < 3:
            continue
        selected.extend(witnessed)
        selected.extend(list(unique.values())[:5])
    exclude_ids = set(excluded)
    for h in selected:
        exclude_ids.update(account_id(n) for n in h.handles)
    return selected, sorted(exclude_ids)


def records(selected):
    for h in selected:
        split_group = opaque('ubuntu-window', h.window)
        if h.role == 'observed_rename':
            person = opaque('ubuntu', h.window, h.event)
        else:
            person = opaque('control', h.window, h.handles[0])
        for nick, group in zip(h.handles, h.histories):
            for i, thread, body in group:
                row = {'id': opaque('post', h.window, i), 'person_id': person,
                       'handle': opaque('ubuntu', h.window, nick), 'text': body,
                       'split_group': split_group, 'community': 'ubuntu-irc', 'label_source': h.role}
                if thread is not None:
                    row['thread'] = opaque('thread', h.window, thread)
                yield row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--exclude-proxy-authors', type=Path, required=True)
    p.add_argument('--exclude-proxy-windows', type=Path, required=True)
    args = p.parse_args()
    out = (args.output, args.exclude_proxy_authors, args.exclude_proxy_windows)
    if len({f.resolve() for f in out}) != len(out):
        p.error('use different paths for the three output files')
    histories, exclusions = build(args.source)
    windows = sorted({h.window + '/' for h in histories})
    data = list(records(histories))
    write_private(args.exclude_proxy_authors, exclusions)
    write_private(args.exclude_proxy_windows, windows)
    write_private(args.output, (json.dumps(row, ensure_ascii=False) for row in data))
    print(json.dumps({'owners': dict(Counter(h.role for h in histories)), 'windows': len(windows),
                      'posts': len(data), 'proxy_authors_excluded': len(exclusions)}, sort_keys=True))


if __name__ == '__main__':
    main()
