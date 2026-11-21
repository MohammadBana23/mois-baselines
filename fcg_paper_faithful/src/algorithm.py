"""Four-cell paper reconstruction for synthetic development predictions."""
from random import Random

import numpy as np

from .core import ScoreLedger, cluster, evolution_score, nearest, rates
from .synthetic import evaluate, validate_development

CELLS = ((1, 0), (1, 1), (0, 0), (0, 1))


def metrics(development, predictions, *, epsilon):
    if not 0 < epsilon < 1:
        raise ValueError("Explicit positive Eq. 5 epsilon required")
    truth = tuple(row.label for row in development)
    groups = tuple(row.group for row in development)
    f1, (z0, z1) = rates(groups, truth, predictions, (0, 1))
    # Preserve the printed group orientation, including ratios above one.
    return f1, min(a / (b + epsilon) for a, b in zip(z0, z1))


def run(points, membership, development, *, k, quotas, epsilon, seed, budget, mode="synthetic"):
    """Run 10 rounds per cell and apply explicit Figure 4 quotas."""
    if mode != "synthetic":
        raise PermissionError("Real calibration is disabled")
    validate_development(development)
    if any(not i.startswith("synthetic:support:") for i in points):
        raise ValueError("Only synthetic Support IDs are supported")
    if k not in (4, 8) or len(quotas) != 4 or any(q not in (0, k // 4, k // 2, k) for q in quotas) or sum(quotas) != k:
        raise ValueError("Invalid paper strategy quotas")
    covered = {cell for cell, count in zip(CELLS, quotas) if count}
    if len(covered) != len({z for z, _ in covered}) * len({y for _, y in covered}):
        raise ValueError("Strategy must be a rectangular group/label allocation")
    nonzero = [q for q in quotas if q]
    if len(set(nonzero)) != 1 or type(budget) is not int or budget < 41 * len(development):
        raise ValueError("Invalid quota distribution or synthetic budget")
    if points.keys() != membership.keys() or any(
        cell not in CELLS for cell in membership.values()
    ) or set(points) & {row.identifier for row in development}:
        raise ValueError("Invalid membership or overlapping roles")
    ledgers = {}
    for index, cell in enumerate(CELLS):
        subset = {i: np.array(points[i], dtype=float) for i in sorted(points) if membership[i] == cell}
        centers = cluster(list(subset.values()), [1] * len(subset), 8, seed + index)
        candidates = tuple(sorted({i for group in nearest(subset, centers, 5) for i in group}))
        if len(candidates) < k:
            raise ValueError("Insufficient unique candidates for paper search")
        ledgers[cell] = ScoreLedger(candidates)
    baseline = metrics(development, evaluate((), development), epsilon=epsilon)
    for index, cell in enumerate(CELLS):
        rng = Random(seed + index)
        ledger = ledgers[cell]
        for _ in range(10):
            selected = ledger.draw(k, rng)
            score = evolution_score(
                metrics(development, evaluate(selected, development), epsilon=epsilon), baseline
            )
            ledger.update(selected, score)
    return tuple(i for cell, quota in zip(CELLS, quotas) for i in ledgers[cell].rank()[:quota]), ledgers
