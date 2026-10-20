"""Record decisions in production, attach what actually happened, fit a threshold on it.

`decide.calibration` fits a temperature or a conformal cut from rows of
`(probabilities, correct index)`. Nothing produced those rows, so the threshold stayed a
guess. A `Recorder` on the client writes one row per decision; `record_outcome` adds the
label once it is known, by id; `rows_from` joins the two and hands the result to the
fitters.

Recording never fails a decision: a recorder that raises is ignored.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from decide.types import ChoiceAnswer, NoulAnswer, Request, Response, ScoreAnswer

__all__ = [
    "JsonlRecorder",
    "Recorder",
    "new_decision_id",
    "record_outcome",
    "recorded_confidences",
    "rows_from",
]

log = logging.getLogger(__name__)


def new_decision_id() -> str:
    """An id for one decision, for attaching its outcome later."""
    return uuid.uuid4().hex


@runtime_checkable
class Recorder(Protocol):
    """Somewhere to put one row. Write your own for a database or a log pipeline."""

    def record(self, row: dict[str, Any]) -> None: ...


def _answer_row(answer: Any) -> dict[str, Any]:
    """One answer, flattened to what a fitter needs: the options in order, and the pick."""
    if isinstance(answer, ChoiceAnswer):
        return {
            "type": "choice",
            "choice": answer.choice,
            "options": list(answer.probabilities),
            "probabilities": list(answer.probabilities.values()),
            "confidence": answer.confidence,
        }
    if isinstance(answer, NoulAnswer):
        # Index order is [false, true], so an outcome of true is index 1.
        return {
            "type": "noul",
            "noul": answer.noul,
            "options": ["false", "true"],
            "probabilities": [1.0 - answer.noul, answer.noul],
            "confidence": answer.confidence,
        }
    if isinstance(answer, ScoreAnswer):
        return {
            "type": "score",
            "score": answer.score,
            "options": list(answer.levels),
            "probabilities": list(answer.probabilities),
            "confidence": answer.confidence,
        }
    raise TypeError(f"unknown answer type: {type(answer)!r}")


def decision_row(decision_id: str, request: Request, response: Response) -> dict[str, Any]:
    """The row a client writes for one decision."""
    return {
        "kind": "decision",
        "id": decision_id,
        "at": time.time(),
        "backend": response.meta.backend,
        "model": response.meta.model,
        "latency_ms": round(response.meta.latency_ms, 3),
        "route": list(response.meta.route),
        "answers": {name: _answer_row(a) for name, a in response.answers.items()},
    }


class JsonlRecorder:
    """Appends one JSON object per line. Append-only, so it is safe across processes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def safe_record(recorder: Recorder | None, row: dict[str, Any]) -> None:
    """Record, and swallow anything it raises: a full disk must not fail a decision."""
    if recorder is None:
        return
    try:
        recorder.record(row)
    except Exception:  # pragma: no cover (defensive; test_recording exercises it)
        log.warning("decide: recording failed, decision unaffected", exc_info=True)


def record_outcome(
    recorder: Recorder | str | Path,
    decision_id: str,
    outcomes: Mapping[str, Any],
) -> None:
    """Attach what actually happened to a decision, by id.

    `outcomes` maps a question name to the right answer: an option for a `choice`, a bool
    for a `noul`, a level index or name for a `score`.
    """
    row = {
        "kind": "outcome",
        "id": decision_id,
        "at": time.time(),
        "outcomes": dict(outcomes),
    }
    target = recorder if isinstance(recorder, Recorder) else JsonlRecorder(recorder)
    safe_record(target, row)


def _correct_index(answer: Mapping[str, Any], outcome: Any) -> int | None:
    """Where the outcome sits in this answer's option order, or None when it does not."""
    options = answer["options"]
    if answer["type"] == "noul" and isinstance(outcome, bool):
        return 1 if outcome else 0
    if isinstance(outcome, int) and not isinstance(outcome, bool):
        return outcome if 0 <= outcome < len(options) else None
    if outcome in options:
        return options.index(outcome)
    return None


def read_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Every row in a JSONL file, skipping blank and unreadable lines."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("decide: skipping an unreadable row")


def rows_from(
    path: str | Path,
    question: str,
    *,
    outcomes_path: str | Path | None = None,
) -> list[tuple[list[float], int]]:
    """Join decisions to their outcomes and return what `decide.calibration` fits on.

    A decision with no outcome, or an outcome naming something the answer never offered,
    is left out: an unlabelled row cannot inform a threshold. The last outcome recorded
    for an id wins, so a correction can be appended.
    """
    decisions: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, Any] = {}
    sources = [path] if outcomes_path is None else [path, outcomes_path]
    for source in sources:
        for row in read_rows(source):
            if row.get("kind") == "decision":
                decisions[row["id"]] = row
            elif row.get("kind") == "outcome" and question in row.get("outcomes", {}):
                outcomes[row["id"]] = row["outcomes"][question]

    fitted: list[tuple[list[float], int]] = []
    for decision_id, outcome in outcomes.items():
        decision = decisions.get(decision_id)
        if decision is None:
            continue
        answer = decision["answers"].get(question)
        if answer is None:
            continue
        index = _correct_index(answer, outcome)
        if index is None:
            continue
        fitted.append(([float(p) for p in answer["probabilities"]], index))
    return fitted


def recorded_confidences(
    path: str | Path,
    question: str,
    *,
    outcomes_path: str | Path | None = None,
) -> list[float]:
    """The confidence recorded for every labelled answer to `question`.

    A threshold table of all zeros usually means the thresholds sit above everything the
    backend reported, which this makes visible.
    """
    return [confidence for confidence, _ in _labelled_pairs(path, question, outcomes_path)]


def _labelled_pairs(
    path: str | Path,
    question: str,
    outcomes_path: str | Path | None,
) -> list[tuple[float, bool]]:
    """(confidence, was the top option right) for every decision with an outcome."""
    decisions: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, Any] = {}
    sources = [path] if outcomes_path is None else [path, outcomes_path]
    for source in sources:
        for row in read_rows(source):
            if row.get("kind") == "decision":
                decisions[row["id"]] = row
            elif row.get("kind") == "outcome" and question in row.get("outcomes", {}):
                outcomes[row["id"]] = row["outcomes"][question]

    pairs: list[tuple[float, bool]] = []
    for decision_id, outcome in outcomes.items():
        decision = decisions.get(decision_id)
        if decision is None or question not in decision["answers"]:
            continue
        answer = decision["answers"][question]
        index = _correct_index(answer, outcome)
        if index is None:
            continue
        probabilities = [float(p) for p in answer["probabilities"]]
        confidence = answer["confidence"]
        if confidence is None:
            confidence = max(probabilities)
        picked = probabilities.index(max(probabilities))
        pairs.append((float(confidence), picked == index))
    return pairs


def accuracy_by_threshold(
    path: str | Path,
    question: str,
    thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
    *,
    outcomes_path: str | Path | None = None,
) -> list[dict[str, float]]:
    """What each threshold would have done, over the recorded rows.

    For every threshold: how many answers it accepts, and how many of those were right.
    This is the table that says whether a gate is safe to turn on.
    """
    pairs = _labelled_pairs(path, question, outcomes_path)
    out = []
    for threshold in thresholds:
        accepted = [right for confidence, right in pairs if confidence >= threshold]
        out.append(
            {
                "threshold": float(threshold),
                "rows": float(len(pairs)),
                "accepted": float(len(accepted)),
                "accepted_share": round(len(accepted) / len(pairs), 4) if pairs else 0.0,
                "accuracy_when_accepted": (
                    round(sum(accepted) / len(accepted), 4) if accepted else 0.0
                ),
            }
        )
    return out
