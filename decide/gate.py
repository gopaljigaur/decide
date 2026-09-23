from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from decide.types import ChoiceAnswer, NoulAnswer, Response, ScoreAnswer

_TYPE_NAMES: Mapping[type, str] = {
    ChoiceAnswer: "choice",
    ScoreAnswer: "score",
    NoulAnswer: "noul",
}


@dataclass(frozen=True)
class Gate:
    """Confidence gate for filtering responses based on answer confidence."""

    min_confidence: float = 0.0
    per_question: Mapping[str, float] | None = None
    per_type: Mapping[str, float] | None = None
    on_error: Literal["next", "raise"] = "next"

    def __post_init__(self) -> None:
        if self.per_type is not None:
            unknown = set(self.per_type) - {"choice", "score", "noul"}
            if unknown:
                raise ValueError(f"Gate.per_type has unknown keys: {sorted(unknown)}")

    def confidence(self, response: Response) -> dict[str, float]:
        """Calculate confidence for each non-Score answer.

        - ChoiceAnswer: max(probabilities.values())
        - NoulAnswer: max(noul, 1 - noul)
        - ScoreAnswer: not included (not gated)
        """
        result = {}
        for name, answer in response.answers.items():
            if isinstance(answer, ChoiceAnswer):
                result[name] = max(answer.probabilities.values(), default=0.0)
            elif isinstance(answer, NoulAnswer):
                result[name] = max(answer.noul, 1 - answer.noul)
            # ScoreAnswer is skipped (not gated)
        return result

    def passes(self, response: Response) -> tuple[bool, str]:
        """Check if response passes confidence gate.

        Returns (True, "ok") if all pass, or
        (False, "low_confidence:<name>=<val:.2f><threshold:.2f>") for the first failing question.
        """
        confidences = self.confidence(response)

        for name in response.answers.keys():
            # Skip Score answers
            if isinstance(response.answers[name], ScoreAnswer):
                continue

            # Get the threshold for this question: per_question, else per_type
            # (keyed by the answer's type), else min_confidence.
            if self.per_question and name in self.per_question:
                threshold = self.per_question[name]
            elif self.per_type and _TYPE_NAMES[type(response.answers[name])] in self.per_type:
                threshold = self.per_type[_TYPE_NAMES[type(response.answers[name])]]
            else:
                threshold = self.min_confidence

            # Check confidence
            confidence_value = confidences[name]
            if confidence_value < threshold:
                return (False, f"low_confidence:{name}={confidence_value:.2f}<{threshold:.2f}")

        return (True, "ok")
