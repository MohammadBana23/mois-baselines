"""Synthetic characterization of the current KMeans backend and candidate stage."""
import unittest

import numpy as np

from fcg_paper_faithful.src.core import cluster, nearest

try:
    from sklearn.cluster import KMeans
except ModuleNotFoundError as error:
    if error.name != "sklearn":
        raise
    KMeans = None


class KMeansAuditTests(unittest.TestCase):
    def test_single_center_respects_duplicate_multiplicity(self):
        points = np.array([[0.0], [10.0]])
        weights = np.array([9, 1])
        centers = cluster(points, weights, 1, seed=47)
        expanded = cluster(np.repeat(points, weights, axis=0), np.ones(10), 1, seed=47)
        np.testing.assert_allclose(centers, [[1.0]])
        np.testing.assert_allclose(centers, expanded)

    def test_seeded_centers_and_stable_neighbor_ties(self):
        points = {"b": np.array([1.0]), "a": np.array([-1.0]), "c": np.array([4.0])}
        centers = cluster(list(points.values()), [1, 1, 1], 2, seed=47)
        replay = cluster(list(points.values()), [1, 1, 1], 2, seed=47)
        np.testing.assert_array_equal(centers, replay)
        self.assertEqual(nearest(points, [[0.0], [1.0]], 2), (("a", "b"), ("b", "a")))

    def test_rejects_insufficient_unique_vectors_and_invalid_weights(self):
        with self.assertRaises(ValueError):
            cluster([[0.0], [0.0]], [1, 1], 2, seed=47)
        with self.assertRaises(ValueError):
            cluster([[0.0], [1.0]], [1, 0], 1, seed=47)

    @unittest.skipIf(KMeans is None, "Optional scikit-learn comparator unavailable")
    def test_reference_agrees_on_well_separated_weighted_fixture(self):
        points = np.array([[0.0], [1.0], [10.0], [11.0]])
        weights = np.array([3, 1, 1, 3])
        actual = cluster(points, weights, 2, seed=47)
        reference = KMeans(
            n_clusters=2, init="k-means++", n_init=10, max_iter=300,
            tol=1e-4, algorithm="lloyd", random_state=47,
        ).fit(points, sample_weight=weights)
        np.testing.assert_allclose(np.sort(actual[:, 0]), [0.25, 10.75])
        np.testing.assert_allclose(np.sort(actual[:, 0]), np.sort(reference.cluster_centers_[:, 0]))
        distances = ((points[:, None, :] - actual) ** 2).sum(axis=2)
        self.assertAlmostEqual(float(weights @ distances.min(axis=1)), reference.inertia_)
