
# Context Layering Engine

A modular Python project for structured XML/JSON context, instruction hierarchy, untrusted-input sandboxing, schema validation, and local-model security evaluation. No hosted LLM API or API key is required.

## Architecture

- `models.py`: Pydantic models for customer records and context packages.
- `engine.py`: Builds hierarchical XML context with system directives, knowledge, and sandboxed user input.
- `validation.py`: Validates JSON, schemas, XML structure, and layer order.
- `local_model.py`: Runs a local Hugging Face model asynchronously.
- `cases.py` / `evaluator.py`: Evaluates 12 prompt-injection and instruction-drift scenarios and generates JSON reports.

Dynamic user content is XML-escaped and isolated as untrusted input. XML structure improves context clarity but does not guarantee prompt-injection prevention.

## Run on Kaggle

```bash
%cd /kaggle/working/context-layering-engine
!pip install -r requirements.txt
!pip install -e .
!python -m pytest -q
```

### Structural verification

```bash
python scripts/run_local_evaluation.py \
  --dry-run --output reports/dry-run.json
```

### Local-model evaluation

```bash
python scripts/run_local_evaluation.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-new-tokens 120 \
  --output reports/local-evaluation.json
```

## Verification Results

- **22 unit tests passed.**
- **12 context/XML payloads validated** in the dry run.
- **11/12 cases passed** in the final hardened local-model evaluation.

The remaining `inj-08` format-drift failure occurred with Qwen2.5-1.5B-Instruct, which returned JSON instead of the required plain-text format. This suggests a limitation in the model's instruction adherence; the case remains marked as failed.

## Security Notes

Treat user input and retrieved documents as untrusted. Validate data and model outputs, enforce tool permissions outside the LLM, and interpret evaluation scores as evidence for the tested model and scenarios, not proof of complete security.
