"""Decision models against an LLM-as-classifier, on the same labelled sets.

Everything here runs locally: laya on MLX, a sentence-transformers cross-encoder,
and chat models through ollama's OpenAI-compatible endpoint. No API keys, so the
numbers cost nothing to reproduce.

    python benchmarks/run.py --n 150
    python benchmarks/run.py --tasks ag_news --systems laya_mlx,crossencoder --n 50

Results land in benchmarks/results/ as JSON, one file per task and system.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from decide import Client
from decide.types import Choice, ChoiceAnswer, Noul, NoulAnswer

RESULTS = Path(__file__).parent / "results"
SEED = 20261015
WARMUP = 3
OLLAMA = "http://localhost:11434/v1"


@dataclass(frozen=True)
class Task:
    """A labelled set plus the typed question that asks for its label."""

    name: str
    dataset: str
    split: str
    text_column: str
    labels: Sequence[str]
    question: str
    kind: str  # "choice" or "noul"

    def load(self, n: int) -> list[tuple[str, int]]:
        from datasets import load_dataset

        rows = load_dataset(self.dataset, split=self.split)
        idx = list(range(len(rows)))
        random.Random(SEED).shuffle(idx)
        out = []
        for i in idx[: n + WARMUP]:
            row = rows[i]
            out.append((str(row[self.text_column]).strip(), int(row["label"])))
        return out

    def questions(self) -> dict[str, Choice | Noul]:
        if self.kind == "choice":
            return {"label": Choice(self.question, list(self.labels))}
        return {"label": Noul(self.question)}


TASKS = {
    "ag_news": Task(
        name="ag_news",
        dataset="fancyzhx/ag_news",
        split="test",
        text_column="text",
        labels=["World", "Sports", "Business", "Sci/Tech"],
        question="Which section of the paper does this story belong to?",
        kind="choice",
    ),
    "sst2": Task(
        name="sst2",
        dataset="stanfordnlp/sst2",
        split="validation",
        text_column="sentence",
        labels=["negative", "positive"],
        question="Is the sentiment of this review positive?",
        kind="noul",
    ),
}


def _laya_mlx() -> Client:
    from decide.backends.laya_mlx import LayaMLXBackend

    return Client([LayaMLXBackend()])


def _crossencoder() -> Client:
    from decide.backends.crossencoder import CrossEncoderBackend

    return Client([CrossEncoderBackend("cross-encoder/ms-marco-MiniLM-L-6-v2")])


def _ollama(model: str) -> Callable[[], Client]:
    def build() -> Client:
        from decide.backends.llm import LLMBackend

        # 256 is well above a correct typed answer, so the cap only truncates a model
        # that has lost the format. Uncapped, llama3.2:1b burns the timeout per row.
        return Client(
            [
                LLMBackend(
                    base_url=OLLAMA,
                    model=model,
                    api_key="ollama",
                    timeout=120,
                    max_tokens=256,
                )
            ]
        )

    return build


SYSTEMS: dict[str, Callable[[], Client]] = {
    "laya_mlx": _laya_mlx,
    "crossencoder": _crossencoder,
    "llama3.2:1b": _ollama("llama3.2:1b"),
    "qwen3.8": _ollama("qwen3.8:latest"),
}


@dataclass
class Run:
    task: str
    system: str
    n: int = 0
    correct: int = 0
    failures: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    hits: list[int] = field(default_factory=list)
    confusion: dict[str, int] = field(default_factory=dict)
    failure_reasons: dict[str, int] = field(default_factory=dict)
    # One record per row, in task order, so a chain can be replayed across systems
    # offline. Failures keep their slot with predicted=None.
    items: list[dict[str, object]] = field(default_factory=list)


def _predict(task: Task, answer: ChoiceAnswer | NoulAnswer) -> tuple[int, float]:
    """Return (predicted label index, the model's confidence in it)."""
    if isinstance(answer, ChoiceAnswer):
        label = list(task.labels).index(answer.choice)
        return label, max(answer.probabilities.values())
    positive = answer.noul >= 0.5
    return (1 if positive else 0), max(answer.noul, 1 - answer.noul)


def _ece(confidences: list[float], hits: list[int], bins: int = 10) -> float:
    """Expected calibration error: average gap between confidence and accuracy."""
    if not confidences:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        chosen = [i for i, c in enumerate(confidences) if (c > lo or b == 0) and c <= hi]
        if not chosen:
            continue
        acc = sum(hits[i] for i in chosen) / len(chosen)
        conf = sum(confidences[i] for i in chosen) / len(chosen)
        total += (len(chosen) / len(confidences)) * abs(acc - conf)
    return total


def _macro_f1(task: Task, run: Run) -> float:
    scores = []
    for i, _name in enumerate(task.labels):
        tp = run.confusion.get(f"{i}|{i}", 0)
        fp = sum(
            v for k, v in run.confusion.items() if k.endswith(f"|{i}") and not k.startswith(f"{i}|")
        )
        fn = sum(
            v for k, v in run.confusion.items() if k.startswith(f"{i}|") and not k.endswith(f"|{i}")
        )
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return sum(scores) / len(scores)


def run_one(task: Task, system: str, n: int) -> Run:
    rows = task.load(n)
    client = SYSTEMS[system]()
    questions = task.questions()
    run = Run(task=task.name, system=system)

    for text, _ in rows[:WARMUP]:
        try:
            client.decide(text, questions)
        except Exception:  # a warm-up failure is not a measurement
            pass

    for text, gold in rows[WARMUP:]:
        run.n += 1
        started = time.perf_counter()
        try:
            response = client.decide(text, questions)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            run.failures += 1
            reason = type(exc).__name__
            run.failure_reasons[reason] = run.failure_reasons.get(reason, 0) + 1
            run.items.append(
                {
                    "gold": gold,
                    "predicted": None,
                    "confidence": None,
                    "latency_ms": elapsed,
                    "error": reason,
                }
            )
            continue
        elapsed = (time.perf_counter() - started) * 1000
        predicted, confidence = _predict(task, response.answers["label"])
        hit = int(predicted == gold)
        run.correct += hit
        run.latencies_ms.append(elapsed)
        run.confidences.append(confidence)
        run.hits.append(hit)
        key = f"{gold}|{predicted}"
        run.confusion[key] = run.confusion.get(key, 0) + 1
        run.items.append(
            {
                "gold": gold,
                "predicted": predicted,
                "confidence": confidence,
                "latency_ms": elapsed,
                "error": None,
            }
        )

    close = getattr(client, "close", None)
    if callable(close):
        close()
    return run


def summarize(task: Task, run: Run) -> dict[str, object]:
    scored = len(run.hits)
    lat = sorted(run.latencies_ms)

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, int(len(lat) * p))] if lat else float("nan")

    return {
        "task": run.task,
        "system": run.system,
        "n": run.n,
        "scored": scored,
        "failures": run.failures,
        "failure_reasons": run.failure_reasons,
        "accuracy": (run.correct / scored) if scored else float("nan"),
        "macro_f1": _macro_f1(task, run) if scored else float("nan"),
        "ece": _ece(run.confidences, run.hits),
        "mean_confidence": statistics.fmean(run.confidences) if run.confidences else float("nan"),
        "latency_ms_p50": pct(0.50),
        "latency_ms_p95": pct(0.95),
        "latency_ms_mean": statistics.fmean(lat) if lat else float("nan"),
        "total_seconds": sum(run.latencies_ms) / 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--n", type=int, default=150)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_name in args.tasks.split(","):
        task = TASKS[task_name]
        for system in args.systems.split(","):
            print(f"==> {task_name} / {system}", flush=True)
            started = time.perf_counter()
            run = run_one(task, system, args.n)
            summary = summarize(task, run)
            summary["wall_seconds"] = round(time.perf_counter() - started, 1)
            rows.append(summary)
            path = RESULTS / f"{task_name}.{system.replace(':', '-')}.json"
            path.write_text(json.dumps({"summary": summary, "run": asdict(run)}, indent=2))
            print(
                f"    acc {summary['accuracy']:.3f}  f1 {summary['macro_f1']:.3f}  "
                f"ece {summary['ece']:.3f}  p50 {summary['latency_ms_p50']:.1f}ms  "
                f"fail {summary['failures']}",
                flush=True,
            )

    print("\n| task | system | n | accuracy | macro F1 | ECE | p50 ms | p95 ms | failures |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['task']} | `{r['system']}` | {r['scored']} | {r['accuracy']:.3f} | "
            f"{r['macro_f1']:.3f} | {r['ece']:.3f} | {r['latency_ms_p50']:.1f} | "
            f"{r['latency_ms_p95']:.1f} | {r['failures']} |"
        )


if __name__ == "__main__":
    main()
