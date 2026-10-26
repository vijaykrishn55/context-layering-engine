"""Validated data contracts for the context layering pipeline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerRecord(BaseModel):
    """Trusted customer data loaded from an application's verified data source."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    customer_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    tier: str = Field(min_length=1, max_length=80)
    account_balance: float = Field(allow_inf_nan=False)
    active_tickets: list[str] = Field(default_factory=list, max_length=100)

    @field_validator("active_tickets")
    @classmethod
    def validate_ticket_ids(cls, values: list[str]) -> list[str]:
        if any(not item.strip() for item in values):
            raise ValueError("ticket values cannot be empty")
        return [item.strip() for item in values]


class ContextPackage(BaseModel):
    """Untrusted query plus validated, trusted knowledge for one request."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    knowledge_base: CustomerRecord
    user_query: str = Field(min_length=1, max_length=20_000)
    request_id: str = Field(default="local-demo", min_length=1, max_length=100)


class EvaluationResult(BaseModel):
    case_id: str
    category: str
    passed: bool
    reason: str
    response: str
