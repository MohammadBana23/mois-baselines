"""Independent implementations of the mechanics in Hu et al., Algorithm 1."""
from dataclasses import dataclass, field
from math import fsum, isfinite
from random import Random

import numpy as np


def rates(groups, truth, predictions, expected):
    if len(groups) != len(truth) or len(truth) != len(predictions):
        raise ValueError("Evaluation lengths differ")
    if any(type(v) is not int or v not in (0, 1) for v in (*truth, *predictions)):
        raise ValueError("Binary labels and predictions required")
    if set(groups) != set(expected):
        raise ValueError("Missing or unexpected development groups")
    tp = sum(y == h == 1 for y, h in zip(truth, predictions))
    denominator = sum(truth) + sum(predictions)
    if not denominator:
        raise ValueError("Undefined F1")
    result = []
    for group in expected:
        numerators = [0, 0]
        denominators = [0, 0]
        for g, y, h in zip(groups, truth, predictions):
            if g == group:
                numerators[y] += h
                denominators[y] += 1
        if not all(denominators):
            raise ValueError("Undefined group TPR/FPR")
        result.append(tuple(n / d for n, d in zip(numerators, denominators)))
    return 2 * tp / denominator, result


def evolution_score(icl, baseline):
    if any(not isfinite(v) or v < 0 for v in (*icl, *baseline)):
        raise ValueError("Invalid metric")
    return fsum(0.5 * max(a - b, 0.05) for a, b in zip(icl, baseline, strict=True))


@dataclass
class ScoreLedger:
    ids: tuple[str, ...]
    history: dict = field(init=False)

    def __post_init__(self):
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate stable ID")
        self.history = {identifier: [] for identifier in self.ids}

    def score(self, identifier):
        values = self.history[identifier]
        return fsum(values) / len(values) if values else 0.05

    def rank(self):
        return tuple(sorted(self.ids, key=lambda identifier: (-self.score(identifier), identifier)))

    def draw(self, k, rng: Random):
        if type(k) is not int or not 1 <= k <= len(self.ids):
            raise ValueError("Insufficient unique candidates")
        pool = sorted(self.ids)
        chosen = []
        for _ in range(k):
            weights = [self.score(identifier) for identifier in pool]
            index = rng.choices(range(len(pool)), weights=weights, k=1)[0]
            chosen.append(pool.pop(index))
        return tuple(sorted(chosen))

    def update(self, selected, score):
        if (not selected or len(set(selected)) != len(selected)
                or not set(selected) <= self.history.keys() or not isfinite(score) or score < 0.05):
            raise ValueError("Invalid score observation")
        for identifier in selected:
            self.history[identifier].append(score)


def cluster(vectors, weights, k, seed):
    """Weighted Lloyd KMeans with ten seeded k-means++ starts."""
    x = np.asarray(vectors, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all()
            or w.shape != (len(x),) or not np.isfinite(w).all() or not (w > 0).all()
            or not 1 <= k <= len(np.unique(x, axis=0))):
        raise ValueError("Invalid KMeans input or insufficient distinct vectors")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(10):
        centers = [x[rng.choice(len(x), p=w / w.sum())]]
        for _ in range(1, k):
            mass = w * ((x[:, None] - centers) ** 2).sum(axis=2).min(axis=1)
            centers.append(x[rng.choice(len(x), p=mass / mass.sum())])
        centers = np.array(centers)
        for _ in range(300):
            distances = ((x[:, None] - centers) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            updated = centers.copy()
            for j in range(k):
                mask = labels == j
                if mask.any():
                    updated[j] = np.average(x[mask], weights=w[mask], axis=0)
            shift = ((centers - updated) ** 2).sum()
            centers = updated
            if shift <= 1e-4:
                break
        loss = w @ ((x[:, None] - centers) ** 2).sum(axis=2).min(axis=1)
        if best is None or loss < best[0]:
            best = (loss, centers)
    return best[1]


def nearest(points, centers, m):
    """Return ordered per-center neighbors, preserving cross-center overlap."""
    if not 1 <= m <= len(points):
        raise ValueError("Insufficient neighbors")
    return tuple(
        tuple(sorted(points, key=lambda i: (float(np.sum((points[i] - center) ** 2)), i))[:m])
        for center in centers
    )
