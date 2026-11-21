"""Synthetic-only runner with bounded attempts and an atomic journal."""
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile

import numpy as np

from fcg_paper_faithful.src.core import ScoreLedger, evolution_score
from fcg_paper_faithful.src.synthetic import evaluate, validate_development
from .algorithm import CELLS, candidate_pools, proposed_final, proposed_metrics


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, state):
    path = Path(path)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump({"state": state, "digest": digest(state)}, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(support, development, *, checkpoint, budget, max_batches=81, mode="synthetic"):
    """Execute complete synthetic batches. Use one process per checkpoint."""
    if mode != "synthetic":
        raise PermissionError("Real calibration is disabled")
    validate_development(development)
    size = len(development)
    if type(budget) is not int or budget < 81 * size or type(max_batches) is not int or max_batches < 0:
        raise ValueError("Insufficient prepaid synthetic budget or invalid batch limit")
    pools, diagnostics = candidate_pools(support)
    signature = digest({
        "format": 1, "numpy": np.__version__, "budget": budget,
        "support": [asdict(row) for row in sorted(support, key=lambda row: row.identifier)],
        "development": [asdict(row) for row in development],
        "pools": list(pools.values()), "centers": list(diagnostics.values()),
    })
    path = Path(checkpoint)
    state = {"signature": signature, "attempts": 0, "events": []}
    if path.exists():
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        state = wrapper["state"]
        if wrapper["digest"] != digest(state) or state["signature"] != signature:
            raise ValueError("Checkpoint input or integrity mismatch")
    events, attempts = state["events"], state["attempts"]
    if (type(attempts) is not int or not size * len(events) <= attempts <= budget
            or attempts % size or len(events) > 81):
        raise ValueError("Invalid checkpoint prefix or accounting")
    ledgers = {cell: ScoreLedger(pools[cell]) for cell in CELLS}
    rng = [Random(10047 + i) for i in range(8)]
    baseline = None
    end = min(81, len(events) + max_batches)
    for index in range(end):
        cell_index = (index - 1) % 8
        ledger = ledgers[CELLS[cell_index]]
        selected = ledger.draw(5, rng[cell_index]) if index else ()
        if index < len(events):
            event = events[index]
            if event["selected"] != list(selected) or event["index"] != index:
                raise ValueError("Checkpoint draw differs from deterministic prefix")
            predictions = tuple(event["predictions"])
        else:
            if state["attempts"] + size > budget:
                raise RuntimeError("Synthetic attempt budget exhausted")
            # Reserve the complete batch before prediction. A crash consumes the reservation.
            state["attempts"] += size
            save(path, state)
            predictions = evaluate(selected, development)
        metrics = proposed_metrics(development, predictions)
        if index:
            ledger.update(selected, evolution_score(metrics, baseline))
        else:
            baseline = metrics
        if index == len(events):
            events.append({"index": index, "selected": list(selected), "predictions": predictions})
            save(path, state)
    return {
        "status": "COMPLETE" if end == 81 else "PAUSED",
        "predictions_attempted_or_reserved": state["attempts"],
        "predictions_completed": size * len(events),
        "fitness_evaluations": max(0, len(events) - 1),
        "diagnostics": diagnostics,
        "selected": proposed_final(ledgers) if end == 81 else None,
    }
