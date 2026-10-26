"""Run the prompt-injection evaluation with a local Transformers model."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_layering.evaluator import build_dry_run_report, run_evaluation
from context_layering.local_model import LocalModelRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--output", default="reports/local-evaluation.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate XML/cases without loading a model")
    args = parser.parse_args()

    if args.dry_run:
        report = build_dry_run_report(args.output)
        print(f"Dry run passed: {report['xml_payloads_validated']} XML payloads validated.")
        print(f"JSON report: {args.output}")
        return

    runner = LocalModelRunner(model_id=args.model, max_new_tokens=args.max_new_tokens)
    asyncio.run(run_evaluation(runner, args.output, model_id=args.model))


if __name__ == "__main__":
    main()
