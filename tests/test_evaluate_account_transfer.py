from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_account_transfer import fingerprint, person_weights, transfer_trials
from author_match.profile_trials import AccountPost, ProfileTrial


class TransferTest(unittest.TestCase):
    def test_controls_stay_in_channel_and_overlap_is_removed(self):
        posts, topics = [], {}
        for group in ('north', 'south'):
            for writer in range(5):
                handle = f'{group}/{writer}'
                for j in range(8):
                    pid = f'{handle}/{j}'
                    text = f'{group} writer {writer} posted this new note on day {j} around lunch'
                    posts.append(AccountPost(pid, handle, handle, pid, text, group))
                    topics[text] = group
        repeated = posts[0].text
        trials, counts = transfer_trials(posts, {fingerprint(repeated.upper() + '!')}, 44)
        self.assertEqual(counts['seen_source_writing'], 1)
        self.assertEqual({t.same for _, t in trials}, {True, False})
        for channel, t in trials:
            self.assertEqual({topics[s] for s in t.left + t.right}, {channel})
            self.assertTrue(set(t.left).isdisjoint(t.right))
            self.assertNotIn(repeated, t.left + t.right)

    def test_bootstrap_reuses_author_multiplicity_for_both_endpoints(self):
        def trial(a, b):
            return ('community', ProfileTrial(('left',), ('right',), a == b, a, b))
        trials = [trial('a', 'a'), trial('b', 'b'), trial('a', 'b'), trial('b', 'a')]
        weights = person_weights(trials, 17, 100)
        np.testing.assert_array_equal(weights[:, 0] + weights[:, 1], 2)
        np.testing.assert_array_equal(weights[:, 2], weights[:, 0] * weights[:, 1])
        np.testing.assert_array_equal(weights[:, 3], weights[:, 2])
