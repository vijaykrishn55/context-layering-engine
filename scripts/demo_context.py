"""Print the compiled XML context without loading a model."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_layering import ContextLayerEngine, ContextPackage, validate_context_json


def main() -> None:
    payload = json.loads(Path("examples/customer_request.json").read_text(encoding="utf-8"))
    package = validate_context_json(payload)
    xml_payload = ContextLayerEngine().compile_prompt(package)
    print(xml_payload)


if __name__ == "__main__":
    main()
