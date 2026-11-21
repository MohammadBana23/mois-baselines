"""Synthetic decision checks; no dataset or model access."""
import unittest

import numpy as np
from sklearn.cluster import KMeans

from fcg_ia5.src.algorithm import CELLS, candidate_pools, proposed_final, proposed_metrics
from fcg_paper_faithful.src.core import ScoreLedger, cluster, evolution_score, nearest, rates
from tests.fixtures import development, support


class ContractAudit(unittest.TestCase):
    def test_c06_numeric_direction_and_floor(self):
        rows = development()
        baseline = proposed_metrics(rows, (0, 1, 1, 1, 0, 0, 0, 1))
        improved = proposed_metrics(rows, tuple(row.label for row in rows))
        self.assertEqual(baseline, (0.75, 0.0))
        self.assertEqual(improved, (1.0, 1.0))
        self.assertEqual(evolution_score(improved, baseline), 0.625)
        self.assertEqual(evolution_score(baseline, improved), 0.05)
        self.assertEqual(proposed_metrics(rows, (0,) * 8), (0.0, 1.0))

    def test_c06_undefined_denominators_and_invalid_prediction(self):
        rows = development()
        with self.assertRaisesRegex(ValueError, "Undefined group"):
            proposed_metrics(rows[1:], (1,) * 7)
        with self.assertRaisesRegex(ValueError, "Undefined F1"):
            rates((0,), (0,), (0,), (0,))
        with self.assertRaisesRegex(ValueError, "Binary"):
            proposed_metrics(rows, (True,) * 8)

    def test_c09_tie_coverage_and_maximum(self):
        ledgers = {
            cell: ScoreLedger((f"{group}-{label}-b", f"{group}-{label}-a"))
            for cell in CELLS for group, label in (cell,)
        }
        self.assertEqual(proposed_final(ledgers), ("0-0-a", "0-1-a", "1-0-a", "2-0-a", "3-1-a"))
        ledgers[(3, 1)].update(("3-1-b",), 0.4)
        self.assertEqual(proposed_final(ledgers), ("0-0-a", "0-1-a", "1-0-a", "2-0-a", "3-1-b"))

    def test_c09_insufficient_pool_stops_before_search(self):
        rows = support()
        for remaining in (0, 4):
            with self.subTest(remaining=remaining):
                with self.assertRaisesRegex(ValueError, "BASELINE_NOT_READY"):
                    candidate_pools(rows[:remaining] + rows[10:])

    def test_backend_diagnostics(self):
        # Deliberately use a small-scale dataset to expose absolute vs relative tolerance.
        generator = np.random.default_rng(507)
        differing_pools = 0
        for index in range(12):
            x = generator.normal(size=(24, 2)) * 0.002
            weights = generator.integers(1, 5, size=len(x))
            actual = cluster(x, weights, 4, seed=47)
            reference = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300,
                               tol=1e-4, algorithm="lloyd", random_state=47).fit(x, sample_weight=weights)
            points = {f"{i:02}": value for i, value in enumerate(x)}
            current_pool = {key for ids in nearest(points, actual, 5) for key in ids}
            reference_pool = {key for ids in nearest(points, reference.cluster_centers_, 5) for key in ids}
            loss = float(weights @ ((x[:, None] - actual) ** 2).sum(axis=2).min(axis=1))
            differing_pools += current_pool != reference_pool
            if index == 0:
                print(f"audit fixture 0: NumPy centers={actual.tolist()} inertia={loss:.12g} pool={sorted(current_pool)}")
                print(f"audit fixture 0: sklearn centers={reference.cluster_centers_.tolist()} inertia={reference.inertia_:.12g} pool={sorted(reference_pool)}")
            self.assertTrue(np.isfinite(loss) and np.isfinite(reference.inertia_))
        print(f"audit: differing candidate unions in {differing_pools}/12 fixtures")
