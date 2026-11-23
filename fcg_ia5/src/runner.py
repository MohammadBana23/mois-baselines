"""Synthetic-only runner with bounded attempts and an atomic journal."""
from dataclasses import asdict
from hashlib import sha256
import json
import os
import sys
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile

import numpy as np
import sklearn

from fcg_paper_faithful.src.core import ScoreLedger, evolution_score
from fcg_paper_faithful.src.synthetic import validate_development
from .prompt_contract import FEATURES, NUMERIC, parse_prediction, render
from .offline_safety import checkpoint_lock, raw_query
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


def fake_response(messages):
    return json.dumps({"prediction": int(digest(messages), 16) % 2})


def synthetic_features(signal):
    return {key: signal if key in NUMERIC else "synthetic" for key in FEATURES}


def run(support, development, *, checkpoint, budget, max_batches=81, mode="synthetic", sample_resources=None,
        execution=None):
    """Run synthetic-only calibration mechanics under an OS writer lock."""
    if mode != "synthetic":
        raise PermissionError("Real calibration is disabled")
    if execution is not None and sample_resources is not None:
        raise ValueError("Use the execution adapter resource guard")
    with checkpoint_lock(checkpoint):
        return _run(support, development, checkpoint, budget, max_batches, sample_resources, execution)


def _run(support, development, checkpoint, budget, max_batches, sample_resources, execution):
    validate_development(development)
    size = len(development)
    if type(budget) is not int or budget < 81 * size or type(max_batches) is not int or max_batches < 0:
        raise ValueError("Insufficient prepaid synthetic budget or invalid batch limit")
    if sample_resources is not None:
        available, peak = sample_resources()
        if available < 6 * 2**30 or peak > 12 * 2**30:
            raise RuntimeError("Resource preflight failed")
    if execution is not None:
        execution.check()
    pools, diagnostics = candidate_pools(support)
    manifest = {
        "format": 3, "mode": "synthetic", "model": None,
        "code": {p.name: sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).with_name("algorithm.py"),
            Path(__file__).with_name("offline_safety.py"),
            Path(__file__).with_name("prompt_contract.py"),
        )}, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "python": list(sys.version_info[:3]), "budget": budget,
        "support_sha256": digest([asdict(row) for row in sorted(support, key=lambda row: row.identifier)]),
        "development_sha256": digest([asdict(row) for row in development]),
        "pools_sha256": digest(list(pools.values())), "centers": list(diagnostics.values()),
        "settings": {"iterations": 10, "cells": 8, "shots": 5,
                     "preflight_gib": 6, "available_floor_mib": 512, "peak_gib": 12},
    }
    if execution is not None:
        manifest["execution"] = execution.profile
    signature = digest(manifest)
    path = Path(checkpoint)
    state = {"signature": signature, "manifest": manifest, "attempts": 0, "events": [], "pending": []}
    if path.exists():
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        state = wrapper["state"]
        if wrapper["digest"] != digest(state) or state["signature"] != signature or state["manifest"] != manifest:
            raise ValueError("Checkpoint input or integrity mismatch")
    events, attempts = state["events"], state["attempts"]
    if (type(attempts) is not int or not size * len(events) <= attempts <= budget
            or len(events) > 81):
        raise ValueError("Invalid checkpoint prefix or accounting")
    pending = state["pending"]
    if (len(pending) > size or (pending and len(events) == 81)
            or attempts < size * len(events) + len(pending)):
        raise ValueError("Invalid pending query accounting")
    support_by_id = {row.identifier: row for row in support}
    low_samples = 0
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
            demos = []
            for identifier in selected:
                demo = support_by_id[identifier]
                demos.append((demo.features | {"sex": demo.sex, "race": demo.race}, demo.label))
            for position, row in enumerate(development):
                if position < len(pending):
                    entry = pending[position]
                    if entry["query"] != row.identifier or entry["selected"] != list(selected):
                        raise ValueError("Checkpoint query does not match deterministic prefix")
                    if entry["status"] == "ok":
                        continue
                    if entry["status"] == "invalid" and execution is not None:
                        raise RuntimeError("Invalid prediction requires explicit disposition")
                    if entry["status"] != "invalid":
                        raise RuntimeError("Unresolved prediction reservation")
                else:
                    entry = {"query": row.identifier, "selected": list(selected)}
                if sample_resources is not None:
                    available, peak = sample_resources()
                    low_samples = low_samples + 1 if available < 512 * 2**20 else 0
                    if low_samples >= 2 or peak > 12 * 2**30:
                        raise RuntimeError("Resource limit exceeded")
                if state["attempts"] >= budget:
                    raise RuntimeError("Synthetic attempt budget exhausted")
                query = raw_query(synthetic_features(row.signal))
                messages = render(query, demos) if execution is None else execution.render(query, demos)
                if execution is not None:
                    native, token_count = execution.prepare(messages)
                if position == len(pending):
                    pending.append(entry)
                state["attempts"] += 1
                entry.update(status="reserved", prompt_sha256=digest(messages))
                if execution is not None:
                    entry.update(native_prompt_sha256=sha256(native.encode("utf-8")).hexdigest(),
                                 prompt_tokens=token_count)
                save(path, state)
                try:
                    if execution is None:
                        predicted = parse_prediction(fake_response(messages))
                    else:
                        predicted = execution.parse(execution.complete(native))
                except ValueError:
                    entry["status"] = "invalid"
                    entry["invalid_attempts"] = entry.get("invalid_attempts", 0) + 1
                    save(path, state)
                    raise
                entry.update(status="ok", prediction=predicted)
                save(path, state)
            predictions = tuple(entry["prediction"] for entry in pending)
        metrics = proposed_metrics(development, predictions)
        if index:
            ledger.update(selected, evolution_score(metrics, baseline))
        else:
            baseline = metrics
        if index == len(events):
            events.append({"index": index, "selected": list(selected), "predictions": predictions,
                           "queries": list(pending)})
            pending.clear()
            save(path, state)
    return {
        "status": "COMPLETE" if end == 81 else "PAUSED",
        "predictions_attempted_or_reserved": state["attempts"],
        "predictions_completed": size * len(events) + sum(q["status"] == "ok" for q in pending),
        "fitness_evaluations": max(0, len(events) - 1),
        "diagnostics": diagnostics,
        "selected": proposed_final(ledgers) if end == 81 else None,
    }
