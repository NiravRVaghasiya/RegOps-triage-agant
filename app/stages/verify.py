"""Stage 4 — verify: the faithfulness guard.

    faithful = cited_rule_id in {r.rule_id for r in retrieved}

and, additionally, the recommended action must be exactly the knowledge-base action of the cited
rule (decide copies it verbatim, so any difference means the decision was tampered with or
free-generated). Not faithful, or no rule cited -> NEEDS_HUMAN_REVIEW; otherwise ACTION_RECOMMENDED.

This check is deliberately independent of decide's own sanitising: it re-derives everything from
the retrieved rules and the rulebook, so it still holds if decide is replaced or bypassed.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.rag import get_rule
from app.schemas import RuleHit, Status
from app.stages.decide import Decision


class Verdict(BaseModel):
    faithful: bool
    status: Status
    reason: str


def verify(decision: Decision, retrieved: list[RuleHit]) -> Verdict:
    retrieved_ids = {r.rule_id for r in retrieved}
    cited = decision.cited_rule_id

    if cited is None:
        note = f" Model attempted to cite {decision.rejected_citation!r}, which was not retrieved." if (
            decision.rejected_citation) else ""
        return Verdict(faithful=False, status=Status.NEEDS_HUMAN_REVIEW,
                       reason="No retrieved rule was cited; refusing to act without a citable rule." + note)
    if cited not in retrieved_ids:
        return Verdict(faithful=False, status=Status.NEEDS_HUMAN_REVIEW,
                       reason=f"Cited rule {cited!r} is not among the retrieved rules {sorted(retrieved_ids)}; "
                       "citation is unfaithful.")
    rule = get_rule(cited)
    if rule is None or decision.recommended_action != rule.action:
        return Verdict(faithful=False, status=Status.NEEDS_HUMAN_REVIEW,
                       reason=f"Recommended action does not match the rulebook action for {cited}; "
                       "action is not grounded in the cited rule.")
    return Verdict(faithful=True, status=Status.ACTION_RECOMMENDED,
                   reason=f"Cited rule {cited} is in the retrieved set and the action matches the rulebook.")
