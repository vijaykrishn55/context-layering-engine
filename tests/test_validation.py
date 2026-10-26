import json

import pytest
from pydantic import ValidationError

from context_layering import (
    ContextValidationError,
    customer_json_schema,
    parse_json_strict,
    validate_context_json,
    validate_customer_json,
)

CUSTOMER = {
    "customer_id": "CUST-1",
    "name": "Test User",
    "tier": "Free",
    "account_balance": 10.0,
    "active_tickets": [],
}


def test_valid_customer_json():
    record = validate_customer_json(json.dumps(CUSTOMER))
    assert record.customer_id == "CUST-1"


def test_missing_customer_id_rejected():
    invalid = dict(CUSTOMER)
    invalid.pop("customer_id")
    with pytest.raises(ContextValidationError, match="Invalid customer JSON"):
        validate_customer_json(invalid)


def test_unknown_customer_field_rejected():
    invalid = {**CUSTOMER, "admin_override": True}
    with pytest.raises(ContextValidationError, match="Invalid customer JSON"):
        validate_customer_json(invalid)


def test_malformed_json_rejected():
    with pytest.raises(ContextValidationError, match="Invalid customer JSON"):
        validate_customer_json('{"customer_id":')


def test_non_object_json_rejected():
    with pytest.raises(ContextValidationError, match="must be an object"):
        parse_json_strict("[1, 2, 3]")


def test_invalid_json_syntax_rejected():
    with pytest.raises(ContextValidationError, match="Malformed JSON"):
        parse_json_strict('{"a": trailing}')


def test_context_json_validation():
    package = validate_context_json({"knowledge_base": CUSTOMER, "user_query": "Hello"})
    assert package.knowledge_base.tier == "Free"


def test_context_missing_query_rejected():
    with pytest.raises(ContextValidationError, match="Invalid context JSON"):
        validate_context_json({"knowledge_base": CUSTOMER})


def test_schema_is_machine_readable():
    schema = customer_json_schema()
    assert "properties" in schema
    assert "customer_id" in schema["properties"]
