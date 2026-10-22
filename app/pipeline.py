"""Orchestration: classify -> retrieve -> decide -> verify -> TriageResult."""

from __future__ import annotations

import logging
import time
from typing import Callable

from app import rag
from app.schemas import TriageResult
from app.stages.classify import classify
from app.stages.decide import Decision, decide
from app.stages.verify import verify

log = logging.getLogger(__name__)

DecideFn = Callable[..., Decision]


def run_pipeline(text: str, *, llm=None, k: int = 3, decide_fn: DecideFn = decide) -> TriageResult:
    """Triage one exception. `llm` / `decide_fn` are injection points for tests and the eval harness."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    classification = classify(text, llm=llm)
    timings["classify"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rules = rag.retrieve(text, k=k)
    timings["retrieve"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    decision = decide_fn(text, classification, rules, llm=llm)
    timings["decide"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    verdict = verify(decision, rules)
    timings["verify"] = time.perf_counter() - t0

    log.info("triage category=%s cited=%s faithful=%s status=%s timings=%s", classification.category,
             decision.cited_rule_id, verdict.faithful, verdict.status.value,
             {k_: round(v, 3) for k_, v in timings.items()})

    # Guard: an unfaithful decision never surfaces its citation or action as if it were actionable.
    return TriageResult(
        input_text=text,
        category=classification.category,
        severity=classification.severity,
        retrieved_rules=rules,
        cited_rule_id=decision.cited_rule_id if verdict.faithful else None,
        recommended_action=decision.recommended_action if verdict.faithful else None,
        status=verdict.status,
        faithful=verdict.faithful,
        rationale=f"{decision.rationale} | verify: {verdict.reason}",
    )
