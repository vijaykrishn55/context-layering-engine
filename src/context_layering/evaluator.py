"""Local model evaluation and machine-readable adherence reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cases import CUSTOMER_DATA, EVALUATION_CASES, EvaluationCase
from .engine import ContextLayerEngine
from .models import ContextPackage, CustomerRecord
from .validation import validate_layered_xml


async def run_evaluation(
    runner: Any,
    output_path: str | Path,
    model_id: str,
    cases: tuple[EvaluationCase, ...] = EVALUATION_CASES,
) -> dict[str, Any]:
    """Run cases sequentially and write all raw responses plus aggregate metrics."""
    engine = ContextLayerEngine()
    record = CustomerRecord.model_validate(CUSTOMER_DATA)
    results: list[dict[str, Any]] = []

    for case in cases:
        package = ContextPackage(knowledge_base=record, user_query=case.query)
        full_xml = engine.compile_prompt(package)
        validate_layered_xml(full_xml)
        xml_prompt = engine.compile_model_user_prompt(package)
        response = await runner.generate(engine.system_prompt, xml_prompt)
        passed, reason = case.evaluate(response)
        results.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "passed": passed,
                "reason": reason,
                "response": response,
            }
        )
        print(f"{case.case_id} | {'PASS' if passed else 'FAIL'} | {case.category}")

    total = len(results)
    passed_count = sum(item["passed"] for item in results)
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "evaluation_type": "local_transformers_inference",
        "api_key_required": False,
        "case_count": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "adherence_percentage": round((passed_count / total * 100.0) if total else 0.0, 2),
        "cases": results,
        "important_note": (
            "This heuristic evaluation measures only the listed prompts, model, and data. "
            "Passing does not prove that prompt injection is eliminated. XML delimiting is "
            "one defense layer, not a complete security boundary."
        ),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nAdherence: {passed_count}/{total} ({report['adherence_percentage']}%)")
    print(f"JSON report: {target}")
    return report


def build_dry_run_report(output_path: str | Path) -> dict[str, Any]:
    """Validate all case payloads/XML without loading or downloading a model."""
    engine = ContextLayerEngine()
    record = CustomerRecord.model_validate(CUSTOMER_DATA)
    checked = 0
    for case in EVALUATION_CASES:
        xml_prompt = engine.compile_prompt(ContextPackage(knowledge_base=record, user_query=case.query))
        validate_layered_xml(xml_prompt)
        checked += 1
    report = {
        "evaluation_type": "dry_run_no_model_loaded",
        "model_loaded": False,
        "case_count": len(EVALUATION_CASES),
        "xml_payloads_validated": checked,
        "prompt_injection_cases": sum(case.category != "baseline" for case in EVALUATION_CASES),
        "passed": checked,
        "failed": 0,
        "note": "This validates the engine and XML structure only; it does not test model behavior.",
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
