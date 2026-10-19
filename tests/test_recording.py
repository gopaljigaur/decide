"""Recording decisions, attaching outcomes, and fitting a threshold from the pair."""

import json

import pytest

from decide import Choice, Client, Gate, Noul, Score
from decide.calibration import fit_temperature
from decide.recording import (
    JsonlRecorder,
    accuracy_by_threshold,
    new_decision_id,
    record_outcome,
    recorded_confidences,
    rows_from,
)
from decide.types import ChoiceAnswer, NoulAnswer, ScoreAnswer
from tests.conftest import FakeBackend

Q = {"tier": Choice("How much effort?", ["simple", "moderate", "complex"])}


def _client(path, **kwargs):
    return Client([FakeBackend("fake", **kwargs)], Gate(), recorder=JsonlRecorder(path))


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_decision_is_recorded_with_the_id_the_caller_gave(tmp_path):
    path = tmp_path / "d.jsonl"
    _client(path).decide("s", Q, decision_id="abc")

    rows = _rows(path)
    assert len(rows) == 1
    assert rows[0]["kind"] == "decision"
    assert rows[0]["id"] == "abc"
    assert rows[0]["backend"] == "fake"
    assert rows[0]["answers"]["tier"]["options"] == ["simple", "moderate", "complex"]
    assert rows[0]["answers"]["tier"]["choice"] == "simple"


def test_an_id_is_generated_when_the_caller_gives_none(tmp_path):
    path = tmp_path / "d.jsonl"
    _client(path).decide("s", Q)
    assert len(_rows(path)[0]["id"]) == 32


def test_nothing_is_written_without_a_recorder(tmp_path):
    path = tmp_path / "d.jsonl"
    Client([FakeBackend("fake")], Gate()).decide("s", Q)
    assert not path.exists()


def test_outcomes_join_to_decisions_and_feed_the_fitter(tmp_path):
    path = tmp_path / "d.jsonl"
    client = _client(path, confidence=0.8)
    for i in range(4):
        client.decide(f"task {i}", Q, decision_id=f"t{i}")
    for i in range(4):
        record_outcome(path, f"t{i}", {"tier": "simple" if i < 3 else "complex"})

    rows = rows_from(path, "tier")
    assert len(rows) == 4
    probabilities, correct = rows[0]
    assert len(probabilities) == 3
    assert correct == 0
    assert rows[3][1] == 2
    assert fit_temperature(rows) > 0


def test_a_decision_without_an_outcome_is_left_out(tmp_path):
    path = tmp_path / "d.jsonl"
    client = _client(path)
    client.decide("labelled", Q, decision_id="a")
    client.decide("never labelled", Q, decision_id="b")
    record_outcome(path, "a", {"tier": "simple"})

    assert len(rows_from(path, "tier")) == 1


def test_an_outcome_naming_something_never_offered_is_left_out(tmp_path):
    path = tmp_path / "d.jsonl"
    _client(path).decide("s", Q, decision_id="a")
    record_outcome(path, "a", {"tier": "nonsense"})

    assert rows_from(path, "tier") == []


def test_the_last_outcome_for_an_id_wins(tmp_path):
    path = tmp_path / "d.jsonl"
    _client(path).decide("s", Q, decision_id="a")
    record_outcome(path, "a", {"tier": "simple"})
    record_outcome(path, "a", {"tier": "complex"})

    rows = rows_from(path, "tier")
    assert len(rows) == 1
    assert rows[0][1] == 2


def test_outcomes_can_live_in_their_own_file(tmp_path):
    decisions, outcomes = tmp_path / "d.jsonl", tmp_path / "o.jsonl"
    _client(decisions).decide("s", Q, decision_id="a")
    record_outcome(outcomes, "a", {"tier": "moderate"})

    assert rows_from(decisions, "tier") == []
    joined = rows_from(decisions, "tier", outcomes_path=outcomes)
    assert len(joined) == 1
    assert joined[0][1] == 1


def test_a_noul_outcome_takes_a_bool(tmp_path):
    path = tmp_path / "d.jsonl"
    client = _client(path, confidence=0.7)
    client.decide("s", {"angry": Noul("Is the customer angry?")}, decision_id="a")
    record_outcome(path, "a", {"angry": True})

    rows = rows_from(path, "angry")
    assert rows[0][1] == 1
    assert rows[0][0] == [pytest.approx(0.3), pytest.approx(0.7)]


def test_a_score_outcome_takes_a_level_index_or_name(tmp_path):
    path = tmp_path / "d.jsonl"
    client = _client(path)
    client.decide("s", {"sev": Score("How severe?", ["low", "high"])}, decision_id="a")
    client.decide("s", {"sev": Score("How severe?", ["low", "high"])}, decision_id="b")
    record_outcome(path, "a", {"sev": 1})
    record_outcome(path, "b", {"sev": "low"})

    assert sorted(index for _, index in rows_from(path, "sev")) == [0, 1]


def test_the_threshold_table_counts_what_each_one_would_accept(tmp_path):
    path = tmp_path / "d.jsonl"
    client = _client(path, confidence=0.9)
    for i in range(5):
        client.decide(f"t{i}", Q, decision_id=f"t{i}")
    for i in range(5):
        record_outcome(path, f"t{i}", {"tier": "simple" if i < 4 else "complex"})

    table = {r["threshold"]: r for r in accuracy_by_threshold(path, "tier", (0.5, 0.95))}
    assert table[0.5]["accepted"] == 5
    assert table[0.5]["accuracy_when_accepted"] == pytest.approx(0.8)
    assert table[0.95]["accepted"] == 0
    assert recorded_confidences(path, "tier") == [pytest.approx(0.9)] * 5


def test_recording_failure_never_fails_the_decision(tmp_path, caplog):
    class Broken:
        def record(self, row):
            raise OSError("disk full")

    client = Client([FakeBackend("fake")], Gate(), recorder=Broken())
    r = client.decide("s", Q)

    assert r.choices["tier"].choice == "simple"


async def test_the_async_client_records_too(tmp_path):
    from decide import AsyncClient

    path = tmp_path / "d.jsonl"
    client = AsyncClient([FakeBackend("fake")], Gate(), recorder=JsonlRecorder(path))
    await client.decide("s", Q, decision_id="a")

    assert _rows(path)[0]["id"] == "a"


def test_the_reported_confidence_is_what_lands_in_the_row(tmp_path):
    """A backend that reports its own confidence is recorded with that, not the peak."""
    path = tmp_path / "d.jsonl"
    answers = {"tier": ChoiceAnswer("simple", {"simple": 0.91, "moderate": 0.09}, 0.55)}
    client = Client([FakeBackend("fake", answers=answers)], Gate(), recorder=JsonlRecorder(path))
    client.decide("s", Q, decision_id="a")
    record_outcome(path, "a", {"tier": "simple"})

    assert _rows(path)[0]["answers"]["tier"]["confidence"] == 0.55
    assert recorded_confidences(path, "tier") == [0.55]


def test_new_decision_ids_are_unique():
    assert len({new_decision_id() for _ in range(100)}) == 100


def test_an_unreadable_line_is_skipped(tmp_path):
    path = tmp_path / "d.jsonl"
    _client(path).decide("s", Q, decision_id="a")
    with path.open("a") as fh:
        fh.write("{not json\n\n")
    record_outcome(path, "a", {"tier": "simple"})

    assert len(rows_from(path, "tier")) == 1


def test_a_recorded_score_answer_keeps_its_levels(tmp_path):
    path = tmp_path / "d.jsonl"
    answers = {"sev": ScoreAnswer(1.0, [0.1, 0.9], ["low", "high"], 0.9)}
    client = Client([FakeBackend("fake", answers=answers)], Gate(), recorder=JsonlRecorder(path))
    client.decide("s", {"sev": Score("How severe?", ["low", "high"])}, decision_id="a")

    row = _rows(path)[0]["answers"]["sev"]
    assert row["options"] == ["low", "high"]
    assert row["probabilities"] == [0.1, 0.9]


def test_a_recorded_noul_answer_orders_false_then_true(tmp_path):
    path = tmp_path / "d.jsonl"
    answers = {"angry": NoulAnswer(0.8)}
    client = Client([FakeBackend("fake", answers=answers)], Gate(), recorder=JsonlRecorder(path))
    client.decide("s", {"angry": Noul("angry?")}, decision_id="a")

    row = _rows(path)[0]["answers"]["angry"]
    assert row["options"] == ["false", "true"]
    assert row["probabilities"] == [pytest.approx(0.2), pytest.approx(0.8)]
