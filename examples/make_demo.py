"""Write an invented social-post corpus for exercising the CLI on a laptop.

The repeated habits here come from a random text generator. Its scores
measure pipeline behavior and have no bearing on real authorship accuracy.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys


# Every synthetic writer can discuss every topic. The community and thread
# identifiers are metadata and never appear in the generated text.
TOPICS = {
    "cooking": ("soup", "dough", "sauce", "broth", "pasta", "crust"),
    "garden": ("compost", "mulch", "soil", "mint", "thyme", "basil"),
    "bikes": ("chain", "brake", "saddle", "pedal", "rack", "tire"),
    "music": ("chorus", "intro", "bridge", "beat", "track", "mix"),
    "photos": ("lens", "flash", "tripod", "filter", "camera", "framing"),
    "camping": ("stove", "tent", "lantern", "tarp", "pack", "blanket"),
}

TIMES = ("this morning", "on Monday", "after work", "over the weekend", "last week")
OPENINGS = ("honestly", "ok so", "weirdly enough", "I guess", "tbh", "yeah")
REACTIONS = ("fair enough", "good call", "I see it now", "that makes sense")
EMOJI = ("🙂", "🙃", "🤷", "😅")
OBSERVATIONS = (
    "I tried the {thing} {when} and it clicked",
    "the {thing} was the part that surprised me {when}",
    "did anyone else try the {thing} {when}",
    "my first thought {when} was to try the {thing} again",
    "I gave the {thing} another shot {when}",
    "I paid more attention to the {thing} {when}",
    "the {other} bothered me more than the {thing} {when}",
)
FOLLOWUPS = (
    "maybe I'm missing something about the {other} though",
    "I probably spent way too long thinking about the {other}",
    "a friend pointed out that I'd changed the {other} too",
    "next time I'll pay closer attention to the {other} first",
    "has anyone tried leaving the {other} alone for a while",
    "the {other} is still sitting exactly where I left it",
    "I wasn't expecting to notice much difference the second time around",
    "my notes from the first attempt make almost no sense to me now",
    "I had to walk away for a minute before I could figure it out",
    "someone else here mentioned the same thing earlier and that helped",
    "it sounds obvious now but I hadn't thought of it that way",
    "I might try again tomorrow if I can find a quiet minute",
)


@dataclass(frozen=True)
class Habits:
    opening: str
    reaction: str
    emoji: str
    separator: str
    lowercase: bool
    punctuation: str

    @classmethod
    def sample(cls, rng):
        return cls(
            rng.choice(OPENINGS), rng.choice(REACTIONS), rng.choice(EMOJI),
            rng.choice((". ", "... ", " — ", "\n", "! ")),
            rng.random() < 0.5, rng.choice((".", "!", "!!", "...", "")),
        )


def write_post(rng, habits, topic):
    thing, other = rng.sample(TOPICS[topic], 2)
    when = rng.choice(TIMES)
    if rng.random() < 0.16:
        clauses = [rng.choice((f"wait which {thing}", f"same with my {other}",
                               f"the {thing} again", f"{habits.reaction} about the {thing}"))]
    else:
        first = rng.choice(OBSERVATIONS).format(thing=thing, other=other, when=when)
        if rng.random() < 0.5:
            first = f"{habits.opening}, {first}"
        clauses = [first]
        extra = rng.choices((0, 1, 2, 4, 6, 10), weights=(3, 5, 5, 4, 2, 1))[0]
        for sentence in rng.sample(FOLLOWUPS, extra):
            clauses.append(sentence.format(other=other))
    if habits.lowercase:
        clauses = [clause.lower() for clause in clauses]
    else:
        clauses = [clause[0].upper() + clause[1:] for clause in clauses]
    text = habits.separator.join(clauses) + habits.punctuation
    if rng.random() < 0.25:
        text += " " + habits.emoji
    return text


def records(authors=320, rounds=12, seed=42):
    rng = random.Random(seed)
    styles = [Habits.sample(rng) for _ in range(authors)]
    topics = list(TOPICS)
    for round_id in range(rounds):
        participants = list(range(authors))
        rng.shuffle(participants)
        for start in range(0, authors, 4):
            shared_topic = rng.choice(topics)
            for author in participants[start:start + 4]:
                # Each round contributes a shared discussion and a second,
                # independent thread. Thus rare author splits still have data.
                for thread, topic in (
                    (f"group-{round_id}-{start}", shared_topic),
                    (f"single-{round_id}-{author}", rng.choice(topics)),
                ):
                    yield {
                        "id": f"demo-{author}-{thread}",
                        "author": f"demo-writer-{author}",
                        "thread": thread,
                        "community": topic,
                        "text": write_post(rng, styles[author], topic),
                        "timestamp": 1704067200 + 604800 * round_id + rng.randrange(86400),
                    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for row in records(seed=args.seed):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote synthetic demo comments to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
