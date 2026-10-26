"""JSON, Pydantic, and XML hierarchy validation."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any

from pydantic import ValidationError

from .exceptions import ContextValidationError
from .models import ContextPackage, CustomerRecord

EXPECTED_LAYER_ORDER = (
    "system_directives",
    "operational_schema",
    "knowledge_base",
    "user_input_sandbox",
)


def validate_customer_json(payload: str | dict[str, Any]) -> CustomerRecord:
    """Parse and validate a customer record from JSON text or a Python mapping."""
    try:
        if isinstance(payload, str):
            return CustomerRecord.model_validate_json(payload)
        return CustomerRecord.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as exc:
        raise ContextValidationError(f"Invalid customer JSON payload: {exc}") from exc


def validate_context_json(payload: str | dict[str, Any]) -> ContextPackage:
    """Parse and validate the complete context package before inference."""
    try:
        if isinstance(payload, str):
            return ContextPackage.model_validate_json(payload)
        return ContextPackage.model_validate(payload)
    except (ValidationError, ValueError, TypeError) as exc:
        raise ContextValidationError(f"Invalid context JSON payload: {exc}") from exc


def customer_json_schema() -> dict[str, Any]:
    """Return the JSON Schema generated from the Pydantic customer contract."""
    return CustomerRecord.model_json_schema()


def validate_layered_xml(xml_payload: str) -> ET.Element:
    """Parse XML and enforce required layers, order, and sandbox structure."""
    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError as exc:
        raise ContextValidationError(f"Malformed XML payload: {exc}") from exc

    if root.tag != "instruction_hierarchy":
        raise ContextValidationError("XML root must be <instruction_hierarchy>")

    actual_order = tuple(child.tag for child in list(root))
    if actual_order != EXPECTED_LAYER_ORDER:
        raise ContextValidationError(
            "Missing or incorrectly ordered context layers; expected "
            + " -> ".join(EXPECTED_LAYER_ORDER)
        )

    sandbox = root.find("./user_input_sandbox/untrusted_user_payload")
    if sandbox is None:
        raise ContextValidationError("XML is missing the untrusted user payload sandbox")
    if len(sandbox):
        raise ContextValidationError("Untrusted user payload must be plain text, not nested XML")

    if root.findtext("./system_directives") is None:
        raise ContextValidationError("System directives cannot be missing")
    if root.find("./knowledge_base/customer_id") is None:
        raise ContextValidationError("Knowledge base is missing customer_id")
    return root


def parse_json_strict(payload: str) -> dict[str, Any]:
    """Parse a JSON object and reject malformed JSON and non-object top-level values."""
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ContextValidationError(f"Malformed JSON payload: {exc}") from exc
    if not isinstance(value, dict):
        raise ContextValidationError("JSON payload must be an object")
    return value
