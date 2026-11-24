import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

# The corpus converter is a standalone reproducibility script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_irc_handle_changes as irc


def segment(a, b, marker=None):
    return [
        f'[14:01] <{a}> my favourite meal would include carrots',
        f'[14:02] <{a}> probably two cups of water',
        marker or f'=== {a} is now known as {b}',
        f'[14:03] <{b}> take the quiet road home',
        f'[14:04] <{b}> stop when the leaves turn yellow',
    ]


class IrcRenameTests(unittest.TestCase):
    def extract(self, lines):
        return list(irc.window_candidates('example', lines, {}, defaultdict(set), Counter()))

    def test_requires_a_server_notice_and_two_distinct_spells(self):
        self.assertEqual(len(self.extract(segment('first', 'next'))), 1)
        forged = '[14:02] <outsider> === first is now known as next'
        self.assertFalse(self.extract(segment('first', 'next', forged)))
        self.assertFalse(self.extract(segment('First', 'first')))

    def test_conflicting_speaker_or_session_break_excludes_trial(self):
        swap = segment('first', 'next')
        self.assertFalse(self.extract(['[14:00] <next> already holds this name'] + swap))
        self.assertFalse(self.extract(swap + ['[14:07] <first> the name is in use again']))
        self.assertFalse(self.extract(swap[:1] + ['=== first [test]  has joined #fixture'] + swap[1:]))
        self.assertFalse(self.extract(swap[:3] + ['=== next [test]  has left #fixture'] + swap[3:]))

    def test_copies_do_not_satisfy_minimum_history(self):
        swap = segment('first', 'next')
        swap[-1] = '[14:04] <next> my favourite meal would include carrots!'
        candidate, = self.extract(swap)
        occurrences = defaultdict(set)
        for group in candidate.histories:
            for i, _, body in group:
                occurrences[irc.canonical_text(body)].add((candidate.window, i))
        groups = irc.drop_copies(candidate, occurrences)
        self.assertEqual([len(g) for g in groups], [1, 1])

    def test_extra_links_only_block_people_and_exclude_proxy_training(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            files = source / 'data/train'
            files.mkdir(parents=True)
            index = []
            for window, a, b in [('first-day', 'alpha', 'bravo'), ('second-day', 'bravo', 'charlie')]:
                lines = segment(a, b)
                if window == 'second-day':
                    # Unrelated subject matter in the independently observed change.
                    lines[0] = '[14:00] <bravo> snow stays on the mountain peak'
                    lines[1] = '[14:01] <bravo> the trail runs beside the lake'
                    lines[3] = '[14:03] <charlie> a train departs at midnight'
                    lines[4] = '[14:04] <charlie> tickets expire by next spring'
                (files / f'{window}.raw.txt').write_text('\n'.join(lines))
                (files / f'{window}.annotation.txt').write_text('0 1 -\n3 4 -\n')
                index.append(f'data/train/{window}.ascii.txt data/train/{window}.annotation.txt unused')
            (source / 'data/list.ubuntu.train.txt').write_text('\n'.join(index))
            selected, counts, excluded = irc.collect(source)
            self.assertEqual(counts['rename_notices'], 2)
            self.assertEqual(len(selected), 1)
            self.assertEqual(set(excluded), {irc.account_id(n) for n in ('alpha', 'bravo', 'charlie')})
            rows = list(irc.output_rows(selected))
            self.assertEqual(len({r['handle'] for r in rows}), 2)
            self.assertEqual(len({r['person_id'] for r in rows}), 1)
            self.assertEqual(len({r['thread'] for r in rows}), 2)
            self.assertEqual(len({r['id'] for r in rows}), 4)
