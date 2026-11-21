"""An in-process fake predictor and explicit labeled evaluator view."""
from dataclasses import dataclass
from hashlib import sha256
import json


@dataclass(frozen=True)
class Development:
    identifier: str
    group: int
    label: int
    signal: int

    def __post_init__(self):
        if not self.identifier.startswith("synthetic:dev:") or type(self.signal) is not int:
            raise ValueError("Only synthetic development records are supported")


def predict(demos, query):
    """Parse an ephemeral, label-free request and emit a binary fake response."""
    payload = json.loads(query)
    if set(payload) != {"id", "signal"}:
        raise ValueError("Invalid synthetic query schema")
    digest = sha256(json.dumps([demos, payload], sort_keys=True).encode()).digest()
    return digest[0] % 2


def validate_development(development):
    if (not development or any(type(row) is not Development for row in development)
            or len({row.identifier for row in development}) != len(development)):
        raise ValueError("Synthetic Development keys must be nonempty and unique")


def evaluate(demos, development):
    validate_development(development)
    return tuple(
        predict(demos, json.dumps({"id": row.identifier, "signal": row.signal}, sort_keys=True))
        for row in development
    )
