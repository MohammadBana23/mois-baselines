"""WORK-03 proposed IA5 mechanics for synthetic Support and Development."""
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import fsum, isfinite, sqrt

import numpy as np

from fcg_paper_faithful.src.core import cluster, nearest, rates

NUMERIC = ("age", "capital-gain", "capital-loss", "hours-per-week")
CATEGORICAL = ("workclass", "education", "marital-status", "occupation", "relationship", "native-country")
GROUPS = (("Male", "White"), ("Female", "White"), ("Male", "Non-white"), ("Female", "Non-white"))
CELLS = tuple((group, label) for group in range(4) for label in (0, 1))


@dataclass(frozen=True)
class Support:
    identifier: str
    sex: str
    race: str
    label: int
    features: dict

    def __post_init__(self):
        if (not self.identifier.startswith("synthetic:support:") or (self.sex, self.race) not in GROUPS
                or type(self.label) is not int or self.label not in (0, 1)):
            raise ValueError("Invalid synthetic Support metadata")


@dataclass(frozen=True)
class Encoder:
    ranges: tuple
    categories: tuple

    @classmethod
    def fit(cls, support):
        if not support or any(type(row) is not Support for row in support):
            raise ValueError("Nonempty Support required to fit")
        for row in support:
            cls.validate(row.features)
        values = [row.features for row in support]
        ranges = tuple((min(row[key] for row in values), max(row[key] for row in values)) for key in NUMERIC)
        categories = tuple(tuple(sorted({row[key] for row in values})) for key in CATEGORICAL)
        return cls(ranges, categories)

    @staticmethod
    def validate(features):
        if set(features) != set(NUMERIC + CATEGORICAL):
            raise ValueError("Expected exactly ten permitted predictors")
        if any(type(features[k]) not in (int, float) or not isfinite(features[k]) for k in NUMERIC):
            raise ValueError("Invalid numeric predictor")
        if any(not isinstance(features[k], str) or not features[k].strip() or features[k] == "?"
               for k in CATEGORICAL):
            raise ValueError("Invalid categorical predictor")

    def transform(self, features):
        self.validate(features)
        values = [0 if low == high else min(1, max(0, (features[k] - low) / (high - low)))
                  for k, (low, high) in zip(NUMERIC, self.ranges)]
        for key, vocabulary in zip(CATEGORICAL, self.categories):
            category = features[key]
            values.extend(float(category == item) / sqrt(2) for item in vocabulary)
            values.append(float(category not in vocabulary) / sqrt(2))
        return np.array(values, dtype=np.float64) / sqrt(10)


def candidate_pools(support):
    encoder = Encoder.fit(support)
    if len({row.identifier for row in support}) != len(support):
        raise ValueError("Duplicate Support ID")
    by_cell = {cell: {} for cell in CELLS}
    for row in sorted(support, key=lambda row: row.identifier):
        cell = (GROUPS.index((row.sex, row.race)), row.label)
        by_cell[cell][row.identifier] = encoder.transform(row.features)
    pools, diagnostics = {}, {}
    for index, (cell, points) in enumerate(by_cell.items()):
        if len(points) < 5:
            raise ValueError(f"BASELINE_NOT_READY: fewer than five IDs in {cell}")
        unique, weights = np.unique(np.stack(list(points.values())), axis=0, return_counts=True)
        count = min(8, len(unique))
        centers = cluster(unique, weights, count, 47 + index)
        neighbors = nearest(points, centers, min(5, len(points)))
        pool = tuple(sorted({identifier for ids in neighbors for identifier in ids}))
        if len(pool) < 5:
            raise ValueError(f"BASELINE_NOT_READY: pool too small in {cell}")
        pools[cell] = pool
        diagnostics[cell] = {
            "rows": len(points), "distinct_vectors": len(unique), "clusters": count,
            "pool_size": len(pool), "centers_sha256": sha256(centers.astype("<f8").tobytes()).hexdigest(),
        }
    return pools, diagnostics


def proposed_metrics(development, predictions):
    """C06 PROPOSED: positive F1 and one minus intersectional EOdds gap."""
    f1, by_group = rates(tuple(row.group for row in development),
                         tuple(row.label for row in development), predictions, range(4))
    return f1, 1 - max(max(values) - min(values) for values in zip(*by_group))


def proposed_final(ledgers):
    """C09 PROPOSED: five cells, four groups, two or three positives."""
    options = []
    for cells in combinations(CELLS, 5):
        if len({g for g, _ in cells}) != 4 or sum(y for _, y in cells) not in (2, 3):
            continue
        winners = [(cell, ledgers[cell].rank()[0]) for cell in cells]
        score = fsum(ledgers[cell].score(identifier) for cell, identifier in winners)
        options.append((-score, tuple(sorted(identifier for _, identifier in winners))))
    return min(options)[1]
