"""Replay the confidence gate over recorded runs, without calling a model again.

`run.py` records one item per row per system, in task order, so a two-tier chain
can be reconstructed offline: at threshold t, every row where tier one's
confidence fell below t is answered by tier two instead. The reconstruction
follows `Gate.passes` and the chain in `decide.client`, including the case where
tier two fails and the chain keeps tier one's low-confidence answer.

    python benchmarks/gate_analysis.py --task ag_news --tier1 laya_mlx --tier2 qwen3.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def load(task: str, system: str) -> list[dict]:
    path = RESULTS / f"{task}.{system.replace(':', '-')}.json"
    if not path.exists():
        raise SystemExit(f"no results at {path}. Run run.py for that task and system first.")
    return json.loads(path.read_text())["run"]["items"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--tier1", required=True)
    parser.add_argument("--tier2", required=True)
    args = parser.parse_args()

    one, two = load(args.task, args.tier1), load(args.task, args.tier2)
    if len(one) != len(two):
        raise SystemExit(f"runs differ in length: {len(one)} vs {len(two)}")

    print(f"task {args.task}, tier 1 {args.tier1}, tier 2 {args.tier2}, {len(one)} rows\n")
    print("| threshold | escalated | accuracy | tier 1 alone | mean ms | tier 2 failures kept |")
    print("|---|---|---|---|---|---|")

    base = sum(1 for a in one if a["predicted"] == a["gold"]) / len(one)
    for step in range(0, 21):
        threshold = step / 20
        hits = escalated = kept_failures = 0
        total_ms = 0.0
        for a, b in zip(one, two, strict=True):
            confident = a["confidence"] is not None and a["confidence"] >= threshold
            if confident:
                total_ms += a["latency_ms"]
                hits += int(a["predicted"] == a["gold"])
                continue
            escalated += 1
            total_ms += a["latency_ms"] + b["latency_ms"]
            if b["predicted"] is None:  # tier two failed: the chain keeps tier one's answer
                kept_failures += 1
                hits += int(a["predicted"] == a["gold"])
            else:
                hits += int(b["predicted"] == b["gold"])
        print(
            f"| {threshold:.2f} | {escalated / len(one):.0%} | {hits / len(one):.3f} | "
            f"{base:.3f} | {total_ms / len(one):.1f} | {kept_failures} |"
        )


if __name__ == "__main__":
    main()
