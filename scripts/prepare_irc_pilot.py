"""Convert annotated IRC conversations to single-message JSONL.

Source: Kummerfeld et al., ACL 2019, https://aclanthology.org/P19-1374/
Dataset: https://github.com/jkkummerfeld/irc-disentanglement (data: CC BY 4.0).
Pilot revision: 82ed04f9627a45d0d6f2ec3c8683c88eb408a0d5
External corpus: Gouravajhala et al., ALTA 2023, https://alta2023.alta.asn.au/files/13.pdf

Usage: python scripts/prepare_irc_pilot.py /path/to/irc-disentanglement posts.jsonl
Add --external to prepare the four other channels for evaluation.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path


MESSAGE = re.compile(r"^\[\d\d:\d\d\] <([^>]+)> (.*)$")
DIRECT_ADDRESS = re.compile(r"^(\S{1,48}?)[,:]\s+")
BOT = re.compile(r"(?:ubotu|ubottu|ubot\d*|ubot[ou]-\w+)", re.IGNORECASE)
IRC_CASEMAP = str.maketrans({"[": "{", "]": "}", "\\": "|", "^": "~"})
DOMAIN_MESSAGE = re.compile(r"^\S+ \d{4}-\d{2}-\d{2} \[\d\d:\d\d(?::\d\d)?\]\s+<([^>]+)>\s?(.*)$")
# Network labels are from Table 1 of Gouravajhala et al. Networks are
# separated even when they may share upstream infrastructure.
NETWORKS = {"mediawiki": "wikimedia", "rust": "mozilla", "stripe": "freenode", "ubuntu-meeting": "ubuntu"}


def irc_name(nick):
    return nick.lower().translate(IRC_CASEMAP)


def account_id(name):
    return hashlib.sha256(name.encode()).hexdigest()[:32]


def canonical_text(text):
    return " ".join(text.casefold().split())


def remove_address(text, names):
    address = DIRECT_ADDRESS.match(text)
    return text[address.end():] if address and irc_name(address[1]) in names else text


def conversations(annotation):
    """The paper defines a conversation by connected components of reply links."""
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        while node != parents[node]:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for line in annotation.read_text().splitlines():
        # Each line specifies the endpoints of an undirected conversation edge.
        left, right, _ = line.split()
        a, b = find(int(left)), find(int(right))
        parents[max(a, b)] = min(a, b)
    return {node: find(node) for node in parents}


def normalized_messages(source):
    index = source / "data/list.ubuntu.train.txt"
    for row in index.read_text().splitlines():
        if not row.strip():
            continue
        ascii_file, annotation, _ = row.split()
        prefix = ascii_file.removesuffix(".ascii.txt")
        window = Path(prefix).name
        membership = conversations(source / annotation)
        with (source / (prefix + ".raw.txt")).open(encoding="utf-8", errors="replace") as raw:
            messages = [MESSAGE.match(line.rstrip("\r\n")) for line in raw]
        names = {irc_name(match[1]) for match in messages if match}
        for number, component in sorted(membership.items()):
            match = messages[number]
            if not match:
                continue
            nick, text = match.groups()
            name = irc_name(nick)
            if BOT.fullmatch(name) or name.endswith("bot") or "\ufffd" in nick + text:
                continue
            text = remove_address(text, names)
            if not text.strip():
                continue
            yield {
                "id": f"{window}/{number}",
                "thread": f"{window}/{component}",
                "author": account_id(name),
                "community": "ubuntu-irc",
                "text": text,
            }


def external_messages(source):
    # Exclude matching handles across networks, including accounts present
    # only in the Ubuntu context lines. A nickname cannot verify a person.
    excluded_names = set()
    for raw in (source / "data/train").glob("*.raw.txt"):
        with raw.open(encoding="utf-8", errors="replace") as stream:
            excluded_names.update(
                irc_name(match[1]) for line in stream if (match := MESSAGE.match(line))
            )
    seen_ubuntu = {canonical_text(row["text"]) for row in normalized_messages(source)}
    records = []
    networks_by_handle = {}
    for channel, network in NETWORKS.items():
        for sample in range(3):
            window = f"{channel}.{sample}"
            directory = source / "data/new-domains"
            membership = conversations(directory / (window + ".annotation.txt"))
            with (directory / (window + ".ascii.txt")).open(encoding="utf-8") as stream:
                messages = [DOMAIN_MESSAGE.match(line.rstrip("\r\n")) for line in stream]
            names = {irc_name(match[1]) for match in messages if match}
            for name in names:
                networks_by_handle.setdefault(name, set()).add(network)
            for number, component in sorted(membership.items()):
                match = messages[number]
                if not match:
                    continue
                name = irc_name(match[1])
                if name in excluded_names or name.endswith("bot") or name in {"wikibugs", "meetingology"}:
                    continue
                body = remove_address(match[2], names)
                if not body.strip() or canonical_text(body) in seen_ubuntu:
                    continue
                records.append((name, {
                    "id": f"{window}/{number}",
                    "thread": f"{window}/{component}",
                    "author": account_id(network + "/" + name),
                    "community": channel,
                    "text": body,
                }))
    for name, post in records:
        if len(networks_by_handle[name]) == 1:
            yield post


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--external", action="store_true",
                        help="prepare four independent channels, excluding Ubuntu speaker overlap")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    authors = Counter()
    threads = set()
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent, delete=False
        ) as out:
            tmp = Path(out.name)
            converter = external_messages if args.external else normalized_messages
            for post in converter(args.source):
                authors[post["author"]] += 1
                threads.add(post["thread"])
                out.write(json.dumps(post, ensure_ascii=False) + "\n")
        os.replace(tmp, args.output)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    print(json.dumps({
        "posts": authors.total(),
        "nicknames": len(authors),
        "conversations": len(threads),
    }))


if __name__ == "__main__":
    main()
