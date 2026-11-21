from dataclasses import replace
import unittest

import numpy as np

from fcg_ia5.src.algorithm import CATEGORICAL, CELLS, Encoder, GROUPS, NUMERIC, candidate_pools, proposed_final, proposed_metrics
from fcg_paper_faithful.src.core import ScoreLedger
from tests.fixtures import development, support


class IA5Tests(unittest.TestCase):
    def test_support_fit_unknown_clip_constant_and_sensitive_rejection(self):
        rows = support()
        encoder = Encoder.fit(rows)
        features = rows[0].features | {NUMERIC[0]: 100, CATEGORICAL[0]: "unseen"}
        self.assertAlmostEqual(np.linalg.norm(encoder.transform(features) - encoder.transform(rows[0].features)) ** 2, .2)
        self.assertEqual(Encoder.fit(rows[:1]).transform(features)[0], 0)
        np.testing.assert_array_equal(encoder.transform(rows[0].features),
                                      Encoder.fit([replace(r, race="Non-white") for r in rows]).transform(rows[0].features))
        with self.assertRaises(ValueError):
            encoder.transform(features | {"sex": "Female"})
        with self.assertRaises(ValueError):
            Encoder.fit(development())

    def test_cell_partition_duplicates_small_and_centers(self):
        rows = support()
        pools, diagnostics = candidate_pools(rows)
        self.assertEqual((pools, diagnostics), candidate_pools(rows[::-1]))
        lookup = {r.identifier: (GROUPS.index((r.sex, r.race)), r.label) for r in rows}
        self.assertEqual(tuple(pools), CELLS)
        for cell, pool in pools.items():
            self.assertTrue(all(lookup[i] == cell for i in pool))
            self.assertEqual(diagnostics[cell]["distinct_vectors"], 5)
            self.assertEqual(diagnostics[cell]["clusters"], 5)
            self.assertGreater(len(pool), 5)
            self.assertEqual(len(set(pool)), len(pool))

    def test_empty_small_and_duplicate_id_fail(self):
        rows = support()
        for invalid in (rows[10:], rows[6:], rows + rows[:1]):
            with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                candidate_pools(invalid)

    def test_metrics_hand_calculation_and_zero_rate(self):
        dev = development()
        self.assertEqual(proposed_metrics(dev, (0, 1) * 4), (1, 1))
        self.assertEqual(proposed_metrics(dev, (0,) * 8), (0, 1))
        self.assertEqual(proposed_metrics(dev, (0, 0) + (0, 1) * 3), (6 / 7, 0))

    def test_missing_denominator_fails(self):
        for dev in (development()[:-1], development()[:6]):
            with self.subTest(length=len(dev)), self.assertRaises(ValueError):
                proposed_metrics(dev, (1,) * len(dev))

    def test_final_ties_use_lexicographic_ids(self):
        ledgers = {(g, y): ScoreLedger((f"{g}{y}b", f"{g}{y}a")) for g, y in CELLS}
        self.assertEqual(proposed_final(ledgers), ("00a", "01a", "10a", "20a", "31a"))

    def test_all_negative_f1_denominator_fails(self):
        dev = tuple(replace(row, label=0) for row in development())
        with self.assertRaisesRegex(ValueError, "Undefined F1"):
            proposed_metrics(dev, (0,) * len(dev))
