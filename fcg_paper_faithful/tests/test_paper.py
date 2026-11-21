from pathlib import Path
from random import Random
import tomllib
import unittest

import numpy as np

from fcg_paper_faithful.src.algorithm import CELLS, metrics, run
from fcg_paper_faithful.src.core import ScoreLedger, cluster, evolution_score, nearest
from tests.fixtures import development


class PaperTests(unittest.TestCase):
    def test_equation_five_orientation(self):
        utility, fair = metrics(development(2), (1, 1, 1, 0), epsilon=.01)
        self.assertAlmostEqual(utility, .4)
        self.assertAlmostEqual(fair, 1 / 1.01)
        _, reverse = metrics(development(2), (1, 0, 1, 1), epsilon=.01)
        self.assertEqual(reverse, 0)

    def test_fitness_floor_and_mean(self):
        self.assertAlmostEqual(evolution_score((.9, .4), (.6, .5)), .175)
        ledger = ScoreLedger(("b", "a", "c"))
        self.assertEqual(ledger.rank(), ("a", "b", "c"))
        ledger.update(("b",), .2)
        ledger.update(("b",), .4)
        self.assertAlmostEqual(ledger.score("b"), .3)
        self.assertEqual(ledger.history["a"], [])

    def test_roulette_replay_and_uniqueness(self):
        ledger = ScoreLedger(tuple("abcde"))
        ledger.update(("e",), 1)
        a = ledger.draw(3, Random(9))
        self.assertEqual(a, ledger.draw(3, Random(9)))
        self.assertEqual(len(set(a)), 3)

    def test_centers_and_neighbor_tie(self):
        centers = cluster([[0], [4]], [3, 1], 1, 12)
        np.testing.assert_allclose(centers, [[1]])
        self.assertEqual(nearest({"b": np.array([2]), "a": np.array([0])}, centers, 1), (("a",),))

    def test_full_four_cell_synthetic_search(self):
        params = tomllib.loads(Path("fcg_paper_faithful/configs/synthetic.toml").read_text())
        points = {f"synthetic:support:{c}:{i}": [i] for c in range(4) for i in range(12)}
        membership = {i: CELLS[int(i.split(":")[2])] for i in points}
        selected, ledgers = run(points, membership, development(2), **params)
        self.assertEqual(tuple(membership[i] for i in selected), CELLS)
        for ledger in ledgers.values():
            self.assertEqual(sum(map(len, ledger.history.values())), 40)
        with self.assertRaises(ValueError):
            run(points, membership, development(2), **(params | {"budget": 0}))

    def test_real_mode_blocked_before_inputs(self):
        with self.assertRaises(PermissionError):
            run({}, {}, (), k=4, quotas=(1, 1, 1, 1), epsilon=1e-6, seed=47, budget=0, mode="calibration")
