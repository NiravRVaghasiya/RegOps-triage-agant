"""Reproducible eval harness:  python -m app.eval [--min-accuracy 0.7] [--out results.json]

Runs every case in data/eval_set.jsonl through the full pipeline and reports:
  * classification accuracy  — predicted category == expected_category
  * faithfulness rate        — share of results with NO hallucinated citation: the emitted
                               cited_rule_id is null or in that case's retrieved set, and any
                               unfaithful decision is routed to NEEDS_HUMAN_REVIEW. Must be 1.0.
plus diagnostics: cited-rule accuracy, no-match handling, retrieval recall@3 and the number of
invented citations the model attempted that the guard caught.

Exits non-zero if accuracy < --min-accuracy, faithfulness < 1.0, or any no-match case is acted on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.llm import embeddings_model_name, llm_provider
from app.pipeline import run_pipeline
from app.rag import get_rule
from app.schemas import Status, TriageResult
from app.stages.decide import Decision, decide

EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_set.jsonl"


def load_cases(path: Path = EVAL_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_faithful_output(result: TriageResult) -> bool:
    """Re-check the guard's invariant from scratch (does not trust the result's own `faithful` flag).

    Actionable results must cite a retrieved rule and carry exactly that rule's rulebook action;
    anything else must be routed to human review with no citation or action surfaced.
    """
    retrieved = {r.rule_id for r in result.retrieved_rules}
    if result.status == Status.ACTION_RECOMMENDED:
        rule = get_rule(result.cited_rule_id) if result.cited_rule_id else None
        return (result.faithful and result.cited_rule_id in retrieved and rule is not None
                and result.recommended_action == rule.action)
    return not result.faithful and result.cited_rule_id is None and result.recommended_action is None


def evaluate(cases: list[dict], llm=None) -> dict:
    rows = []
    for case in cases:
        captured: list[Decision] = []

        def recording_decide(*args, **kwargs) -> Decision:
            decision = decide(*args, **kwargs)
            captured.append(decision)
            return decision

        result = run_pipeline(case["text"], llm=llm, decide_fn=recording_decide)
        expected_rule = case.get("expected_rule_id")
        retrieved = [r.rule_id for r in result.retrieved_rules]
        rows.append({
            "id": case["id"],
            "expected_category": case["expected_category"],
            "category": result.category,
            "category_ok": result.category == case["expected_category"],
            "expected_rule_id": expected_rule,
            "cited_rule_id": result.cited_rule_id,
            "rule_ok": result.cited_rule_id == expected_rule,
            "retrieved": retrieved,
            "retrieval_hit": expected_rule is None or expected_rule in retrieved,
            "status": result.status.value,
            "faithful": result.faithful,
            "faithful_output": is_faithful_output(result),
            "rejected_citation": captured[0].rejected_citation if captured else None,
        })

    n = len(rows)
    labeled = [r for r in rows if r["expected_rule_id"]]
    no_match = [r for r in rows if not r["expected_rule_id"]]
    return {
        "provider": llm_provider(),
        "embeddings": embeddings_model_name(),
        "n": n,
        "classification_accuracy": sum(r["category_ok"] for r in rows) / n,
        "faithfulness_rate": sum(r["faithful_output"] for r in rows) / n,
        "cited_rule_accuracy": sum(r["rule_ok"] for r in rows) / n,
        "no_match_escalated": (sum(r["status"] == Status.NEEDS_HUMAN_REVIEW.value for r in no_match) / len(no_match))
        if no_match else None,
        "retrieval_recall_at_3": (sum(r["retrieval_hit"] for r in labeled) / len(labeled)) if labeled else None,
        "hallucinations_caught": sum(r["rejected_citation"] is not None for r in rows),
        "rows": rows,
    }


def _mark(ok: bool) -> str:
    return "ok" if ok else "XX"


def print_report(report: dict) -> None:
    print(f"\nRegOps Triage eval  provider={report['provider']}  embeddings={report['embeddings']}  n={report['n']}\n")
    header = f"{'id':<5} {'expected_category':<23} {'predicted':<23} {'cat':<3} {'exp_rule':<8} {'cited':<8} " \
             f"{'rule':<4} {'status':<19} {'faithful'}"
    print(header)
    print("-" * len(header))
    for r in report["rows"]:
        print(f"{r['id']:<5} {r['expected_category']:<23} {r['category']:<23} {_mark(r['category_ok']):<3} "
              f"{str(r['expected_rule_id'] or '-'):<8} {str(r['cited_rule_id'] or '-'):<8} {_mark(r['rule_ok']):<4} "
              f"{r['status']:<19} {r['faithful']!s:<5}{'' if r['faithful_output'] else '  HALLUCINATED'}")
    print()
    print(f"classification accuracy : {report['classification_accuracy']:.3f}")
    print(f"faithfulness rate       : {report['faithfulness_rate']:.3f}   (no hallucinated citations; must be 1.000)")
    print(f"cited-rule accuracy     : {report['cited_rule_accuracy']:.3f}")
    if report["no_match_escalated"] is not None:
        print(f"no-match -> human review: {report['no_match_escalated']:.3f}")
    if report["retrieval_recall_at_3"] is not None:
        print(f"retrieval recall@3      : {report['retrieval_recall_at_3']:.3f}")
    print(f"invented citations caught by guard: {report['hallucinations_caught']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-set", type=Path, default=EVAL_PATH)
    parser.add_argument("--min-accuracy", type=float, default=0.7)
    parser.add_argument("--out", type=Path, help="write the full JSON report here")
    args = parser.parse_args(argv)

    report = evaluate(load_cases(args.eval_set))
    print_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    passed = (report["classification_accuracy"] >= args.min_accuracy and report["faithfulness_rate"] == 1.0
              and report["no_match_escalated"] in (None, 1.0))
    print(f"\nEVAL {'PASS' if passed else 'FAIL'} (accuracy >= {args.min_accuracy}, faithfulness == 1.0, "
          "no-match -> human review == 1.0)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
