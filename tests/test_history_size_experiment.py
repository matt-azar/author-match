from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_history_size import choose, summarize
from history_size_cross_case import description, partner, pick, repeated_metrics


def entries(day, page, prefix, n):
    return [(day, (page, str(day)), f"{prefix} sample {i}", f"{prefix} {i}") for i in range(n)]


class HistorySizeExperimentTests(unittest.TestCase):
    def test_matched_budget_and_source_exclusions(self):
        h = {(1, "a"): entries(1, "old", "early", 50),
             (1, "b"): entries(61, "new", "later", 50),
             (0, "c"): entries(61, "new", "control", 50)}
        draw = lambda: choose(h, set(), random.Random(7))
        owners, left, pairs = draw()
        self.assertEqual(owners, ("a", "b", "c"))
        self.assertEqual((len(left), len(pairs)), (50, 50))
        for bad_day, bad_page in [(60, "new"), (61, "old"), (61, "different")]:
            h[0, "c"] = entries(bad_day, bad_page, "control", 50)
            self.assertIsNone(draw())

    def test_same_cases_across_budgets(self):
        cases = [{5:[.9, .1]}, {5:[.2, .8], 20:[.9, .1]}]
        out = summarize(cases, 20)
        self.assertEqual(out["investigations"], 1)
        self.assertEqual(out["budgets"][5]["auc"], 0)
        self.assertEqual(out["budgets"][20]["auc"], 1)

    def test_cross_case_negative_uses_other_investigation_and_matched_time(self):
        def case(start, offset):
            a = entries(start, "early", "original words", 50)
            b = entries(start + offset, "later", "continuation words", 50)
            return description(a, b)
        records = [case(1, 60), case(1, 61), case(1, 161)]
        self.assertEqual(partner(0, records, {0, 1, 2}), 1)
        self.assertIsNone(partner(0, records, {0, 2}))

    def test_fifty_distinct_later_summaries_required(self):
        h = {(1, "a"): entries(1, "first", "early", 50),
             (1, "b"): entries(61, "second", "later", 49)}
        self.assertIsNone(pick(h, set(), random.Random(7)))

    def test_multiple_draws_keep_cases_and_full_budget_fixed(self):
        best = [.9, .1]
        worst = [.1, .9]
        a = {5: best, 20: worst, 50: best}
        b = {5: worst, 20: best, 50: best}
        result = repeated_metrics([[a, a], [b, b]])
        self.assertAlmostEqual(result[5]["mean_auc"], .5)
        self.assertEqual(result[5]["draw_auc_range"], [0., 1.])
        self.assertEqual(result[50]["mean_within_case_concordance"], 1.)
        self.assertEqual(result[50]["paired_auc_change_from_5_95pct"], [.5, .5])
