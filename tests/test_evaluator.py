import json

import pytest

from context_layering.cases import EVALUATION_CASES
from context_layering.evaluator import run_evaluation


class FakeLocalRunner:
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        assert system_prompt.startswith("You are an enterprise AI")
        assert "<system_directives>" not in user_prompt
        return "Answer: The verified account balance is $1450.50, tier is Enterprise, and active tickets are TICK-101 and TICK-105."


@pytest.mark.asyncio
async def test_evaluator_writes_machine_readable_report(tmp_path):
    output = tmp_path / "nested" / "report.json"
    report = await run_evaluation(FakeLocalRunner(), output, model_id="fake-local-test")
    assert report["case_count"] == len(EVALUATION_CASES)
    assert report["passed"] == len(EVALUATION_CASES)
    assert report["adherence_percentage"] == 100.0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["api_key_required"] is False
    assert saved["cases"][0]["passed"] is True
