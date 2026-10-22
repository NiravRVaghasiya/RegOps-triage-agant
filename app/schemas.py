"""Pydantic data contracts shared by every stage, the API and the eval harness."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Status(str, Enum):
    RESOLVED = "RESOLVED"
    ACTION_RECOMMENDED = "ACTION_RECOMMENDED"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"


class ExceptionIn(BaseModel):
    """Request body for POST /triage: a free-text exception."""

    text: str = Field(..., min_length=1, max_length=4000, pattern=r"\S", description="Free-text exception to triage.")


# Spec name for the request model; `Exception` shadows the builtin, so it is exported as an alias only.
Exception = ExceptionIn  # noqa: A001


class RuleHit(BaseModel):
    rule_id: str
    text: str
    score: float


class TriageResult(BaseModel):
    input_text: str
    category: str  # from classify
    severity: Severity
    retrieved_rules: list[RuleHit]
    cited_rule_id: Optional[str] = None  # from decide
    recommended_action: Optional[str] = None
    status: Status
    faithful: bool  # from verify
    rationale: str
