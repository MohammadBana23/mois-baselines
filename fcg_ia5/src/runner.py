"""Synthetic-only runner with bounded attempts and an atomic journal."""
from dataclasses import asdict
from hashlib import sha256
import json
import os
import time
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
        for attempt, delay in enumerate((.05, .1, .2, .4, .8, 0)):
            try:
                temporary.replace(path)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or delay == 0:
                    raise
                time.sleep(delay)
    finally:
        temporary.unlink(missing_ok=True)


def fake_response(messages):
    return json.dumps({"prediction": int(digest(messages), 16) % 2})


def synthetic_features(signal):
    return {key: signal if key in NUMERIC else "synthetic" for key in FEATURES}


def run(support, development, *, checkpoint, budget, max_batches=81, mode="synthetic", sample_resources=None,
        execution=None, verified_features=None):
    """Run synthetic-only calibration mechanics under an OS writer lock."""
    if mode != "synthetic":
        raise PermissionError("Real calibration is disabled")
    if execution is not None and sample_resources is not None:
        raise ValueError("Use the execution adapter resource guard")
    if verified_features is not None and execution is None:
        raise ValueError("Verified inputs require an execution adapter")
    with checkpoint_lock(checkpoint):
        return _run(support, development, checkpoint, budget, max_batches, sample_resources, execution, verified_features)


def _run(support, development, checkpoint, budget, max_batches, sample_resources, execution, verified_features):
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
    if verified_features is not None:
        support_features, query_features, expected_pool_hash = verified_features
        if set(support_features) != {row.identifier for row in support} or set(query_features) != {row.identifier for row in development}:
            raise ValueError("Verified feature IDs differ")
        canonical = [[list(cell), [item.removeprefix("synthetic:support:") for item in pools[cell]]]
                     for cell in pools]
        observed = sha256(json.dumps(canonical, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
        if observed != expected_pool_hash:
            raise ValueError("Candidate pool hash mismatch")
        manifest["verified_features"] = digest([support_features, query_features, observed])
    signature = digest(manifest)
    path = Path(checkpoint)
    state = {"signature": signature, "manifest": manifest, "attempts": 0, "events": [], "pending": []}
    if path.exists():
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        state = wrapper["state"]
        if wrapper["digest"] != digest(state) or state["signature"] != digest(state["manifest"]):
            raise ValueError("Checkpoint input or integrity mismatch")
        replacement = False
        if (state["manifest"] != manifest and budget == 13609 and state["attempts"] == 877
                and sha256(path.read_bytes()).hexdigest() == "62690184285eba3e2b412b8a14e5b41d724175691885f9f79d161a3e19ea6994"
                and state["manifest"] == (manifest | {"code": manifest["code"] | {"runner.py": "89263c454c093d64d2c222ad11dc3e31fa5261257fd5300bfce014a135be95ed"}})):
            state["manifest"], state["signature"] = manifest, signature
        if state["manifest"] != manifest:
            legacy = manifest | {"budget": 13608, "code": manifest["code"] | {
                "runner.py": "2343d385bfa1468cb156113985806caba7c1016ca10929ad46f8c1f36ca4c4e2"}}
            replacement = (
                execution is not None and budget == 13609 and state["manifest"] == legacy
                and sha256(path.read_bytes()).hexdigest() == "b73d3abe6f7a2e82fad3388d5902d3677ebc8a08cd472509c2a543b95fe01948"
                and state["attempts"] == 142 and not state["events"]
                and len(state["pending"]) == 142
                and state["pending"][-1]["status"] == "unknown"
                and all(item["status"] == "ok" for item in state["pending"][:-1])
            )
            if not replacement:
                raise ValueError("Checkpoint input or integrity mismatch")
            state["manifest"], state["signature"] = manifest, signature
    else:
        replacement = False
        if budget == 13609:
            raise ValueError("Replacement budget requires the existing checkpoint")
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
                features = (demo.features | {"sex": demo.sex, "race": demo.race}
                            if verified_features is None else support_features[identifier])
                demos.append((features, demo.label))
            for position, row in enumerate(development):
                if position < len(pending):
                    entry = pending[position]
                    if entry["query"] != row.identifier or entry["selected"] != list(selected):
                        raise ValueError("Checkpoint query does not match deterministic prefix")
                    if entry["status"] == "ok":
                        continue
                    if entry["status"] == "invalid" and execution is not None:
                        raise RuntimeError("Invalid prediction requires explicit disposition")
                    replacing = replacement and index == 0 and position == 141
                    if entry["status"] != "invalid" and not replacing:
                        raise RuntimeError("Unresolved prediction reservation")
                else:
                    replacing = False
                    entry = {"query": row.identifier, "selected": list(selected)}
                if sample_resources is not None:
                    available, peak = sample_resources()
                    low_samples = low_samples + 1 if available < 512 * 2**20 else 0
                    if low_samples >= 2 or peak > 12 * 2**30:
                        raise RuntimeError("Resource limit exceeded")
                if state["attempts"] >= budget:
                    raise RuntimeError("Synthetic attempt budget exhausted")
                query = (raw_query(synthetic_features(row.signal)) if verified_features is None
                         else query_features[row.identifier])
                messages = render(query, demos) if execution is None else execution.render(query, demos)
                if execution is not None:
                    native, token_count = execution.prepare(messages)
                if position == len(pending):
                    pending.append(entry)
                if replacing:
                    if (entry["prompt_sha256"] != digest(messages)
                            or entry["native_prompt_sha256"] != sha256(native.encode("utf-8")).hexdigest()):
                        raise ValueError("Replacement prompt differs from attempt 142")
                    state["replacement_audit"] = {
                        "decision": "PM-00: one replacement only; physical ceiling 13609",
                        "original_attempt": 142, "original": dict(entry), "replacement_attempt": 143,
                    }
                    entry = {"query": row.identifier, "selected": list(selected), "attempt": 143}
                    pending[position] = entry
                    replacement = False
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
