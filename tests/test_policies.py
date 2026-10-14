"""The alternative gating policies, and the fitters that set their parameters."""

from __future__ import annotations

import math
import random

import pytest

from decide.calibration import (
    Calibration,
    fit_conformal_quantile,
    fit_temperature,
    temper,
)
from decide.gate import Gate
from decide.policies import (
    CalibratedGate,
    ConformalGate,
    EntropyGate,
    MarginGate,
    Policy,
)
from decide.types import ChoiceAnswer, Meta, NoulAnswer, Response, ScoreAnswer


def make_response(**answers: object) -> Response:
    return Response(
        answers=dict(answers),  # type: ignore[arg-type]
        meta=Meta(backend="test", model=None, latency_ms=0.0),
    )


def choice(**probabilities: float) -> ChoiceAnswer:
    winner = max(probabilities, key=lambda k: probabilities[k])
    return ChoiceAnswer(choice=winner, probabilities=dict(probabilities))


# --- temper -----------------------------------------------------------------


def test_temperature_of_one_changes_nothing() -> None:
    probabilities = [0.7, 0.2, 0.1]
    assert temper(probabilities, 1.0) == pytest.approx(probabilities, abs=1e-9)


def test_high_temperature_flattens_and_low_sharpens() -> None:
    probabilities = [0.7, 0.2, 0.1]
    assert max(temper(probabilities, 3.0)) < max(probabilities)
    assert max(temper(probabilities, 0.3)) > max(probabilities)


def test_temper_still_sums_to_one() -> None:
    for temperature in (0.2, 1.0, 5.0):
        assert sum(temper([0.5, 0.3, 0.2], temperature)) == pytest.approx(1.0)


def test_temper_rejects_a_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        temper([0.5, 0.5], 0.0)


# --- fit_temperature --------------------------------------------------------


def test_overconfident_rows_fit_a_temperature_above_one() -> None:
    """Reports 0.95 but is right 70% of the time, so its probabilities need pulling down."""
    rows = [([0.95, 0.05], 0)] * 70 + [([0.95, 0.05], 1)] * 30
    assert fit_temperature(rows) > 1.0


def test_underconfident_rows_fit_a_temperature_below_one() -> None:
    rows = [([0.55, 0.45], 0)] * 95 + [([0.55, 0.45], 1)] * 5
    assert fit_temperature(rows) < 1.0


def test_calibrated_rows_fit_a_temperature_near_one() -> None:
    rows = [([0.8, 0.2], 0)] * 80 + [([0.8, 0.2], 1)] * 20
    assert fit_temperature(rows) == pytest.approx(1.0, abs=0.15)


def test_fit_temperature_needs_rows() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        fit_temperature([])


def test_fit_temperature_rejects_an_out_of_range_label() -> None:
    with pytest.raises(ValueError, match="outside"):
        fit_temperature([([0.5, 0.5], 7)])


# --- fit_conformal_quantile -------------------------------------------------


def test_conformal_quantile_covers_the_true_label_at_the_stated_rate() -> None:
    """Fit on one half, measure coverage on the other. This is the whole promise."""
    rng = random.Random(11)
    rows = []
    for _ in range(4000):
        peak = rng.uniform(0.34, 0.99)
        rest = (1.0 - peak) / 2
        probabilities = [peak, rest, rest]
        # The winner is right with probability equal to its own confidence.
        correct = 0 if rng.random() < peak else rng.choice([1, 2])
        rows.append((probabilities, correct))

    fit, held_out = rows[:2000], rows[2000:]
    for alpha in (0.05, 0.1, 0.2):
        quantile = fit_conformal_quantile(fit, alpha=alpha)
        floor = 1.0 - quantile
        covered = sum(1 for p, correct in held_out if p[correct] >= floor)
        assert covered / len(held_out) == pytest.approx(1 - alpha, abs=0.03)


def test_conformal_quantile_admits_everything_when_rows_are_too_few() -> None:
    assert fit_conformal_quantile([([0.9, 0.1], 0)], alpha=0.01) == 1.0


def test_fit_conformal_quantile_rejects_a_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        fit_conformal_quantile([([0.9, 0.1], 0)], alpha=1.5)


# --- the policies read what Gate cannot -------------------------------------


def test_entropy_separates_two_answers_gate_cannot_tell_apart() -> None:
    """Same winning probability, different shape underneath."""
    clear = make_response(label=choice(a=0.51, b=0.02, c=0.02, d=0.45))
    tied = make_response(label=choice(a=0.51, b=0.17, c=0.16, d=0.16))

    gate = Gate(min_confidence=0.5)
    assert gate.passes(clear)[0] is True
    assert gate.passes(tied)[0] is True  # Gate sees 0.51 both times

    entropy = EntropyGate(max_entropy=0.75)
    assert entropy.passes(clear)[0] is True
    assert entropy.passes(tied)[0] is False


def test_margin_separates_a_runaway_winner_from_a_photo_finish() -> None:
    runaway = make_response(label=choice(a=0.51, b=0.02, c=0.02, d=0.45))
    photo_finish = make_response(label=choice(a=0.51, b=0.49, c=0.0, d=0.0))

    assert Gate(min_confidence=0.5).passes(photo_finish)[0] is True

    margin = MarginGate(min_margin=0.1)
    assert margin.passes(runaway)[0] is False  # 0.51 against 0.45 is still close
    assert margin.passes(make_response(label=choice(a=0.9, b=0.1)))[0] is True
    assert margin.passes(photo_finish)[0] is False


def test_entropy_reason_names_the_question_and_the_numbers() -> None:
    ok, reason = EntropyGate(max_entropy=0.1).passes(
        make_response(team=choice(a=0.34, b=0.33, c=0.33))
    )
    assert ok is False
    assert reason.startswith("high_entropy:team=")
    assert ">0.10" in reason


def test_margin_reason_names_the_question_and_the_numbers() -> None:
    ok, reason = MarginGate(min_margin=0.5).passes(make_response(team=choice(a=0.6, b=0.4)))
    assert ok is False
    assert reason == "low_margin:team=0.20<0.50"


# --- conformal --------------------------------------------------------------


def test_conformal_accepts_one_survivor_and_escalates_several() -> None:
    gate = ConformalGate(quantile=0.5)  # survivors need at least 0.5
    assert gate.passes(make_response(label=choice(a=0.9, b=0.1)))[0] is True
    assert gate.passes(make_response(label=choice(a=0.5, b=0.5)))[0] is False


def test_conformal_reports_the_set_size_in_the_reason() -> None:
    gate = ConformalGate(quantile=0.8)  # survivors need at least 0.2
    ok, reason = gate.passes(make_response(team=choice(a=0.4, b=0.35, c=0.25)))
    assert ok is False
    assert reason == "ambiguous:team=set_size=3"


def test_conformal_confidence_is_one_over_the_set_size() -> None:
    gate = ConformalGate(quantile=0.8)
    scores = gate.confidence(make_response(team=choice(a=0.4, b=0.35, c=0.25)))
    assert scores["team"] == pytest.approx(1 / 3)


def test_conformal_set_is_never_empty() -> None:
    """A cut above every probability still leaves the winner, not nothing."""
    gate = ConformalGate(quantile=0.0)  # survivors would need a probability of 1.0
    assert gate.prediction_set_size("team", choice(a=0.4, b=0.35, c=0.25)) == 1


def test_per_question_quantiles_override_the_default() -> None:
    gate = ConformalGate(quantile=0.5, quantiles={"loose": 0.9})
    answer = choice(a=0.45, b=0.3, c=0.25)
    assert gate.prediction_set_size("strict", answer) == 1
    assert gate.prediction_set_size("loose", answer) == 3


# --- calibrated gate --------------------------------------------------------


def test_calibrated_gate_at_temperature_one_matches_gate() -> None:
    response = make_response(label=choice(a=0.7, b=0.3))
    assert CalibratedGate(min_confidence=0.6).passes(response) == Gate(min_confidence=0.6).passes(
        response
    )


def test_calibration_pulls_an_overconfident_answer_below_the_threshold() -> None:
    response = make_response(label=choice(a=0.88, b=0.12))
    assert Gate(min_confidence=0.85).passes(response)[0] is True
    assert CalibratedGate(min_confidence=0.85, temperature=2.5).passes(response)[0] is False


def test_per_question_temperatures_override_the_default() -> None:
    response = make_response(hot=choice(a=0.88, b=0.12), cool=choice(a=0.88, b=0.12))
    gate = CalibratedGate(min_confidence=0.85, temperatures={"hot": 3.0})
    scores = gate.confidence(response)
    assert scores["hot"] < 0.85
    assert scores["cool"] == pytest.approx(0.88)


def test_calibrated_gate_rejects_a_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="positive"):
        CalibratedGate(temperature=0.0)


# --- shared behaviour -------------------------------------------------------


ALL_POLICIES = [
    CalibratedGate(min_confidence=0.9),
    EntropyGate(max_entropy=0.0),
    MarginGate(min_margin=0.9),
    ConformalGate(quantile=0.5),
]


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_score_answers_are_never_gated(policy: Policy) -> None:
    """Same rule as Gate: an expected value over levels has no single confidence."""
    response = make_response(
        urgency=ScoreAnswer(score=1.5, probabilities=[0.33, 0.34, 0.33], levels=["a", "b", "c"])
    )
    assert policy.passes(response) == (True, "ok")
    assert policy.confidence(response) == {}


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_every_policy_satisfies_the_protocol(policy: Policy) -> None:
    assert isinstance(policy, Policy)
    assert policy.on_error == "next"


@pytest.mark.parametrize("policy", ALL_POLICIES)
def test_noul_answers_are_gated_as_a_two_way_distribution(policy: Policy) -> None:
    undecided = make_response(angry=NoulAnswer(noul=0.5))
    assert policy.passes(undecided)[0] is False


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CalibratedGate(per_type={"nope": 0.5}),
        lambda: EntropyGate(per_type={"nope": 0.5}),
        lambda: MarginGate(per_type={"nope": 0.5}),
    ],
)
def test_per_type_rejects_unknown_keys(factory: object) -> None:
    with pytest.raises(ValueError, match="per_type has unknown keys"):
        factory()  # type: ignore[operator]


def test_out_of_range_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="max_entropy"):
        EntropyGate(max_entropy=1.5)
    with pytest.raises(ValueError, match="min_margin"):
        MarginGate(min_margin=-0.1)
    with pytest.raises(ValueError, match="quantile"):
        ConformalGate(quantile=2.0)


# --- Calibration round trip -------------------------------------------------


def test_calibration_round_trips_through_json(tmp_path: object) -> None:
    original = Calibration(
        temperatures={"team": 1.8}, quantiles={"team": 0.42}, note="ag_news, 150 rows"
    )
    restored = Calibration.from_json(original.to_json())
    assert restored == original
    assert restored.temperature_for("team") == 1.8
    assert restored.quantile_for("team") == 0.42


def test_calibration_falls_back_to_neutral_defaults() -> None:
    empty = Calibration()
    assert empty.temperature_for("anything") == 1.0
    assert empty.quantile_for("anything") == 1.0


def test_calibration_saves_and_loads(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "calibration.json"
    Calibration(temperatures={"a": 2.0}).save(path)
    assert Calibration.load(path).temperature_for("a") == 2.0


# --- the fitters feed the policies ------------------------------------------


def test_a_fitted_temperature_feeds_the_gate_that_uses_it() -> None:
    rows = [([0.95, 0.05], 0)] * 70 + [([0.95, 0.05], 1)] * 30
    temperature = fit_temperature(rows)
    gate = CalibratedGate(min_confidence=0.9, temperature=temperature)
    assert gate.passes(make_response(label=choice(a=0.95, b=0.05)))[0] is False


def test_a_fitted_quantile_feeds_the_gate_that_uses_it() -> None:
    rows = [([0.6, 0.4], 0)] * 80 + [([0.6, 0.4], 1)] * 20
    quantile = fit_conformal_quantile(rows, alpha=0.1)
    gate = ConformalGate(quantile=quantile)
    assert gate.prediction_set_size("label", choice(a=0.6, b=0.4)) == 2
    assert math.isclose(gate.quantile, quantile)


# --- the llm backend caps its own output ------------------------------------


def test_llm_backend_caps_output_tokens_by_default() -> None:
    """A model that loses the format generates until the context ends, and bills for it."""
    from decide.backends.llm import LLMBackend
    from decide.types import Choice, Request

    backend = LLMBackend(base_url="http://example.invalid/v1", model="m")
    body = backend._body(
        Request(state="s", questions={"q": Choice("pick", ["a", "b"])}), response_format=True
    )
    assert body["max_tokens"] == 512


def test_llm_backend_can_send_no_cap() -> None:
    from decide.backends.llm import LLMBackend
    from decide.types import Choice, Request

    backend = LLMBackend(base_url="http://example.invalid/v1", model="m", max_tokens=None)
    body = backend._body(
        Request(state="s", questions={"q": Choice("pick", ["a", "b"])}), response_format=True
    )
    assert "max_tokens" not in body


def test_llm_backend_refuses_a_nonsense_cap() -> None:
    from decide.backends.llm import LLMBackend

    with pytest.raises(ValueError, match="max_tokens"):
        LLMBackend(base_url="http://example.invalid/v1", max_tokens=0)
