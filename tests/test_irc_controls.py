import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_irc_controlled_handles as controls
from prepare_irc_pilot import account_id


class IrcControlTests(unittest.TestCase):
    def test_mask_preserves_substrings_and_matches_irc_case_variants(self):
        mask = controls.mask_names({'bird', 'x{fish}'})
        self.assertEqual(mask.sub('', 'Bird on birdseed near X[fish]'), ' on birdseed near ')

    def test_controls_bracket_the_event_and_share_one_group(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / 'data/train'
            directory.mkdir(parents=True)
            lines = [
                '[12:01] <A1> the silver bell rang loudly',
                '[12:02] <A1> put the sweater in a bag',
                '[12:03] <Bird> taking a steady walk home',
                '[12:04] <Fish> someone took the new car',
                '[12:05] <Cat> place a nice cup here',
                '[12:06] <Away> dropped the keys down stairs',
                '[12:07] <Else> sun is setting early',
                '[12:08] <Else> rain was forecast yesterday morning',
                '=== A1 is now known as A2',
                '=== Away [private]  has left #example',
                '[12:09] <A2> tomorrow the sea will rise',
                '[12:10] <A2> can Bird please explain these numbers',
                '[12:11] <Bird> those houses need fresh paint',
                '[12:12] <Fish> my brother drinks lots of tea',
                '[12:13] <Cat> we listened for a while',
                '[12:14] <Away> print the next three pages',
            ]
            (directory / 'window.raw.txt').write_text('\n'.join(lines))
            (directory / 'window.annotation.txt').write_text('0 1 -\n10 11 -\n')
            (root / 'data/list.ubuntu.train.txt').write_text('data/train/window.ascii.txt data/train/window.annotation.txt unused\n')
            selected, excluded = controls.build(root)
            self.assertEqual(Counter(h.role for h in selected), {'observed_rename': 1, 'same_window_control': 3})
            self.assertEqual(set(excluded), {account_id(n) for n in ('a1', 'a2', 'bird', 'fish', 'cat')})
            rows = list(controls.records(selected))
            self.assertEqual(len({r['split_group'] for r in rows}), 1)
            self.assertEqual(len({r['handle'] for r in rows}), 5)
            self.assertEqual(len({r['person_id'] for r in rows}), 4)
            self.assertTrue(all('bird' not in r['text'].lower() for r in rows))
            # A nickname in a second window is ineligible as a single-person
            # control. Only two controls then remain, so this window is skipped.
            (directory / 'other.raw.txt').write_text('[12:11] <Fish> a different conversation again')
            (directory / 'other.annotation.txt').write_text('0 0 -\n')
            with (root / 'data/list.ubuntu.train.txt').open('a') as stream:
                stream.write('data/train/other.ascii.txt data/train/other.annotation.txt unused\n')
            self.assertEqual(controls.build(root)[0], [])
