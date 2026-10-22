# BUILD SPEC — RegOps Triage Agent

> **For the coding agent (Claude Code / Cursor / Codex):** Build this project exactly as specified. Follow the layers in order. Do NOT skip a validation gate — each gate must print PASS before you move to the next layer. Build the **walking skeleton first** (Layer 0) and keep it runnable at every step. Clone-and-run only: no PyPI packaging, no publishing.

---

## 1. One-line purpose

An agentic **LangChain** workflow that ingests a free-text data-quality / vendor-submitted exception, **classifies** it, **retrieves** the governing rule via **RAG**, **decides** the next action, and **verifies** that decision against the retrieved rule before emitting a structured, auditable result — refusing to act on a rule it cannot cite.

## 2. Non-negotiable outcomes (definition of done)

1. `POST /triage` accepts `{ "text": "<exception>" }` and returns a **Pydantic-validated** JSON `TriageResult`.
2. Pipeline has **4 named stages**: `classify → retrieve → decide → verify`.
3. **Faithfulness guard**: if the decision does not cite a real retrieved rule id, status = `NEEDS_HUMAN_REVIEW` (never a fabricated rule).
4. A **reproducible eval harness** (`python -m app.eval`) prints classification accuracy + faithfulness rate over the labeled eval set.
5. **Deployed live** on free tier (GCP Cloud Run primary; AWS App Runner path documented) with a public URL.
6. Minimal **Gradio UI** at `/` for live demo.
7. `pytest` green; `README.md` with run + deploy steps.

## 3. Tradeoff gates (decide BEFORE coding — record the choice in README)

| Decision | Default | Rationale | Only deviate if |
| --- | --- | --- | --- |
| LLM provider | **Gemini** (`gemini-1.5-flash`, free tier) via `langchain-google-genai` | Free, fast, pairs with Cloud Run | user sets `LLM_PROVIDER=openai/anthropic/ollama` |
| Provider abstraction | `get_llm()`** factory** reading `LLM_PROVIDER` env | Swappable; defends "multi-provider" claim | never hardcode a provider |
| Vector store | **ChromaDB** (local persistent dir) | Zero infra, free, embeds in container | corpus > 1k docs (N/A here) |
| Embeddings | Gemini embeddings (`text-embedding-004`) | Same key, free | provider swapped |
| Agent style | **LCEL chains** for classify/decide + explicit retrieve step; tool-calling only for the decide step's action selection | Keeps it debuggable in a 2-day build | — |
| Deploy target | **GCP Cloud Run** | Always-free, container, public URL in minutes | user wants AWS → App Runner (same Dockerfile) |

## 4. Tech stack (pin in requirements.txt)

`langchain`, `langchain-google-genai`, `langchain-community`, `chromadb`, `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `gradio`, `python-dotenv`, `pytest`, `httpx`.

## 5. Repo layout

```
regops-triage-agent/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app + Gradio mount
│   ├── schemas.py         # Pydantic: Exception, TriageResult, RuleHit
│   ├── llm.py             # get_llm() + get_embeddings() factory
│   ├── pipeline.py        # classify → retrieve → decide → verify orchestration
│   ├── rag.py             # Chroma index build + retriever
│   ├── stages/
│   │   ├── classify.py    # LCEL chain -> category + severity
│   │   ├── decide.py      # tool-calling / LCEL -> action + cited rule_id
│   │   └── verify.py      # faithfulness check against retrieved rules
│   └── eval.py            # runs eval_set.jsonl, prints metrics
├── data/
│   ├── rules.jsonl        # ~18 synthetic rules (RAG knowledge base)
│   └── eval_set.jsonl     # ~25 labeled exceptions incl. edge cases
├── tests/
│   └── test_pipeline.py
├── .env.example           # LLM_PROVIDER, GOOGLE_API_KEY, CHROMA_DIR
├── Dockerfile
├── requirements.txt
└── README.md

```

## 6. Data contracts (Pydantic — build these FIRST)

```python
# schemas.py
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional

class Severity(str, Enum):
    LOW = "LOW"; MEDIUM = "MEDIUM"; HIGH = "HIGH"

class Status(str, Enum):
    RESOLVED = "RESOLVED"
    ACTION_RECOMMENDED = "ACTION_RECOMMENDED"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"

class RuleHit(BaseModel):
    rule_id: str
    text: str
    score: float

class TriageResult(BaseModel):
    input_text: str
    category: str                       # from classify
    severity: Severity
    retrieved_rules: list[RuleHit]
    cited_rule_id: Optional[str] = None # from decide
    recommended_action: Optional[str] = None
    status: Status
    faithful: bool                      # from verify
    rationale: str

```

`rules.jsonl` record: `{"rule_id": "R-012", "category": "DOCUMENTATION_MISSING", "text": "...", "action": "..."}` `eval_set.jsonl` record: `{"id": "E-01", "text": "...", "expected_category": "...", "expected_rule_id": "R-012"}` (use `expected_rule_id: null` for deliberate no-match cases).

## 7. Build layers (DEPENDENCY-ORDERED — do in this exact sequence)

### Layer 0 — Walking skeleton ⬅ build + deploy-test this FIRST

- FastAPI app, `POST /triage` returns a **hardcoded** `TriageResult`. Gradio UI mounted at `/`. Dockerfile builds and runs locally.
- **GATE 0:** `uvicorn app.main:app` serves; `curl POST /triage` returns valid JSON matching the schema; `docker build` succeeds. Print `GATE 0 PASS`.

### Layer 1 — LLM factory + schemas

- Implement `schemas.py` and `llm.py` (`get_llm()`, `get_embeddings()` reading env; `.env.example` created).
- **GATE 1:** `get_llm().invoke("ping")` returns text; schemas import clean. Print `GATE 1 PASS`.

### Layer 2 — RAG index

- `rag.py`: load `data/rules.jsonl`, build persistent Chroma index, expose `retrieve(text, k=3) -> list[RuleHit]`.
- **GATE 2:** retrieving a known exception returns the expected `rule_id` in top-3. Print `GATE 2 PASS`.

### Layer 3 — Classify stage

- `stages/classify.py`: LCEL chain → structured `{category, severity}` (use `.with_structured_output`).
- **GATE 3:** 3 sample inputs classify into valid enum values. Print `GATE 3 PASS`.

### Layer 4 — Decide stage

- `stages/decide.py`: given input + retrieved rules, choose `recommended_action` and the `cited_rule_id`** (must be one of the retrieved ids or null)**.
- **GATE 4:** decision's `cited_rule_id` is always either a retrieved id or null — never invented. Print `GATE 4 PASS`.

### Layer 5 — Verify stage (the differentiator)

- `stages/verify.py`: `faithful = cited_rule_id in {r.rule_id for r in retrieved}`. If not faithful OR no rule cited → `status = NEEDS_HUMAN_REVIEW`. Otherwise `ACTION_RECOMMENDED`.
- **GATE 5:** inject a fake citation → pipeline flags `faithful=False` and `NEEDS_HUMAN_REVIEW`. Print `GATE 5 PASS`.

### Layer 6 — Wire pipeline + replace skeleton

- `pipeline.py` runs all 4 stages; `main.py` calls real pipeline instead of the hardcoded stub.
- **GATE 6:** end-to-end `POST /triage` on 3 real inputs returns faithful, schema-valid results. Print `GATE 6 PASS`.

### Layer 7 — Eval harness

- `eval.py`: run every `eval_set.jsonl` case, compute **classification accuracy** (category match) and **faithfulness rate** (no hallucinated citations); print a summary table + overall numbers.
- **GATE 7:** `python -m app.eval` prints accuracy ≥ 0.7 and faithfulness = 1.0 (faithfulness MUST be perfect — a hallucinated citation is a hard fail). Print `GATE 7 PASS`.

### Layer 8 — Tests + deploy

- `tests/test_pipeline.py`: schema validity, the faithfulness-guard case, a no-match edge case.
- Deploy to Cloud Run; document AWS App Runner alternative in README.
- **GATE 8:** `pytest` green; public URL returns valid triage. Print `GATE 8 PASS`.

## 8. Deploy commands (put in README)

**GCP Cloud Run (primary):**

```bash
gcloud run deploy regops-triage-agent \
  --source . --region europe-central2 \
  --allow-unauthenticated \
  --set-env-vars LLM_PROVIDER=gemini,GOOGLE_API_KEY=$GOOGLE_API_KEY

```

**AWS alternative (same Dockerfile):** push image to ECR → AWS App Runner service, set the same env vars. Document as a talking point ("identical container runs on AWS App Runner / Lambda-container").

## 9. Guardrails for the coding agent

- Keep the walking skeleton runnable at every layer — never leave the build red between gates.
- No provider hardcoding — everything through `get_llm()`/`get_embeddings()`.
- The faithfulness guard is the heart of the project — never weaken it to make a test pass.
- Secrets only via env / `.env` (gitignored); ship `.env.example` only.
- README must state the tradeoff-gate choices from §3 and show the eval output.

