# RegOps Triage Agent

An agentic **LangChain** workflow that ingests a free-text data-quality / vendor-submitted exception,
**classifies** it, **retrieves** the governing rule via **RAG**, **decides** the next action, and **verifies**
that decision against the retrieved rule before emitting a structured, auditable result — **refusing to act on a
rule it cannot cite**.

```
POST /triage {"text": "..."}
        │
        ▼
 ┌──────────┐   ┌──────────┐   ┌────────────────────┐   ┌──────────────────────┐
 │ classify │ → │ retrieve │ → │ decide (tool call) │ → │ verify (faithfulness) │ → TriageResult (Pydantic)
 └──────────┘   └──────────┘   └────────────────────┘   └──────────────────────┘
  LCEL +         Chroma top-k    cite_rule(rule_id ∈        cited id ∈ retrieved ids
  structured     over 18 rules   retrieved ids) |           AND action == rulebook action
  output                         escalate_to_human          else NEEDS_HUMAN_REVIEW
```

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (Git Bash: source .venv/Scripts/activate); Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # set GOOGLE_API_KEY, or LLM_PROVIDER=offline to run without any key

uvicorn app.main:app --reload     # UI: http://127.0.0.1:8000/   API docs: http://127.0.0.1:8000/docs
curl -s -X POST http://127.0.0.1:8000/triage -H "Content-Type: application/json" \
     -d '{"text": "Vendor submitted a power bank listing but attached no battery safety test documentation."}'

python -m app.eval                # eval harness: accuracy + faithfulness (exit 1 if targets missed)
pytest                            # 54 tests, always run on the offline provider (no key, no network)
python scripts/gates.py all       # re-run the build-spec validation gates 0–8
```

Example response:

```json
{
  "input_text": "Vendor submitted a power bank listing but attached no battery safety test documentation.",
  "category": "DOCUMENTATION_MISSING",
  "severity": "MEDIUM",
  "retrieved_rules": [
    {"rule_id": "R-001", "text": "Lithium-ion batteries or products containing them ...", "score": 0.3866},
    {"rule_id": "R-002", "text": "Cosmetic and personal-care products ...", "score": 0.2398},
    {"rule_id": "R-007", "text": "Products listed under an incorrect product-type ...", "score": 0.0847}
  ],
  "cited_rule_id": "R-001",
  "recommended_action": "Request UN38.3 test summary from vendor; hold listing until received.",
  "status": "ACTION_RECOMMENDED",
  "faithful": true,
  "rationale": "Rule R-001 ... | verify: Cited rule R-001 is in the retrieved set and the action matches the rulebook."
}
```

## Tradeoff gates (§3 of the build spec)

| Decision | Choice | Notes |
| --- | --- | --- |
| LLM provider | **Gemini** via `langchain-google-genai` (default `LLM_PROVIDER=gemini`) | Model default is **`gemini-2.5-flash`** (`GEMINI_MODEL`). *Deviation:* the spec's `gemini-1.5-flash` has been retired by Google, so it would 404; set `GEMINI_MODEL` to override. |
| Provider abstraction | `get_llm()` / `get_embeddings()` factory in `app/llm.py` reading `LLM_PROVIDER` | `gemini` · `openai` · `anthropic` · `ollama` · `offline`. No stage imports a provider SDK. Non-Gemini SDKs (`langchain-openai`, `langchain-anthropic`, `langchain-ollama`) are optional installs. |
| Vector store | **ChromaDB**, local persistent dir (`CHROMA_DIR`) | Collections are keyed by *embedding model + rules-file fingerprint*, so a changed rulebook or model gets a fresh index and a stale one can never serve deleted rules. Retrieval and the rulebook lookups used by decide/verify come from one snapshot of `rules.jsonl`. |
| Embeddings | Gemini **`models/gemini-embedding-001`** (`GEMINI_EMBEDDING_MODEL`) | *Deviation:* `text-embedding-004` is deprecated by Google; override via env if your key still has it. |
| Agent style | **LCEL chains** for classify (`prompt \| llm.with_structured_output(Classification)`) and decide; explicit retrieve step; **tool calling only for decide** | Decide binds two tools with forced tool choice: `cite_rule(rule_id: enum[retrieved ids], rationale)` and `escalate_to_human(reason)`. |
| Deploy target | **GCP Cloud Run** (same Dockerfile runs on **AWS App Runner**) | See *Deploy* below. |

### The `offline` provider (added for key-less CI / demo)

`LLM_PROVIDER=offline` is a deterministic, dependency-free stand-in implementing the same LangChain interfaces
(`invoke`, `bind_tools`, `with_structured_output`) so the **pipeline code path is identical** for every provider.
It is **not a language model**: it reads the tagged prompt blocks the stages emit (`<exception>`, `<category>`,
`<classification>`, `<rule>`) and answers by lexical similarity using hashed word-stem + character-4-gram
embeddings. It exists so that `pytest`, the gates and the eval harness run with no key and no network.
**Its eval numbers measure the pipeline + guard, not LLM quality** — the taxonomy descriptions it matches against
were written with the rulebook in view. Re-run `python -m app.eval` with `LLM_PROVIDER=gemini` for real numbers.

## The faithfulness guard

The guard is enforced three times, independently:

1. **Schema constraint (decide):** the `cite_rule` tool's `rule_id` is a JSON-schema `enum` of *only this request's
   retrieved ids*.
2. **Post-validation (decide):** if a provider ignores the enum, an unknown id is discarded (`cited_rule_id=None`,
   kept as `rejected_citation` for audit). The recommended action is **copied from the rulebook entry**, never
   free-generated.
3. **Verify (`app/stages/verify.py`):** re-derives `faithful = cited_rule_id in {r.rule_id for r in retrieved}` and
   additionally requires the action to equal the cited rule's rulebook action. Not faithful or nothing cited →
   `NEEDS_HUMAN_REVIEW`, otherwise `ACTION_RECOMMENDED`. An unfaithful decision's citation and action are **not
   surfaced** in the result; the rationale records what was rejected.

Also: user text is escaped (`<`/`>`) before entering prompts, so an exception cannot forge `<rule>` tags
(covered by `test_prompt_tag_injection_cannot_forge_a_rule`).

Status semantics: `faithful` follows the spec literally — it is `true` only when a retrieved rule is cited, so a
correct refusal on a no-match input reads `faithful=false, status=NEEDS_HUMAN_REVIEW`. `RESOLVED` is part of the
contract but the spec's verify stage never emits it (reserved for a future "no action needed" auto-close path).

## Eval

`python -m app.eval` runs all 25 labeled cases in `data/eval_set.jsonl` through the full pipeline.

* **classification accuracy** — predicted category == `expected_category`
* **faithfulness rate** — share of results with **no hallucinated citation**, re-checked from scratch rather than
  trusting the result's own `faithful` flag: every `ACTION_RECOMMENDED` result cites a retrieved rule and carries
  exactly that rule's rulebook action; everything else is `NEEDS_HUMAN_REVIEW` with no citation/action surfaced.
  Must be **1.000**. Because the guard enforces this by construction, the metric is a regression alarm for the
  guard; the adversarial run below is what shows the guard actually catching invented citations.
* diagnostics: cited-rule accuracy, no-match → human-review rate, retrieval recall@3, invented citations caught.

Output on this machine (`LLM_PROVIDER=offline`, 2026-10-22):

```
RegOps Triage eval  provider=offline  embeddings=offline/hash-1024  n=25

id    expected_category       predicted               cat exp_rule cited    rule status              faithful
-------------------------------------------------------------------------------------------------------------
E-01  DOCUMENTATION_MISSING   DOCUMENTATION_MISSING   ok  R-001    R-001    ok   ACTION_RECOMMENDED  True
...
E-22  NO_MATCH                NO_MATCH                ok  -        -        ok   NEEDS_HUMAN_REVIEW  False
E-23  NO_MATCH                NO_MATCH                ok  -        -        ok   NEEDS_HUMAN_REVIEW  False
E-24  HAZMAT                  IP_INFRINGEMENT         XX  R-015    R-016    XX   ACTION_RECOMMENDED  True
E-25  NO_MATCH                NO_MATCH                ok  -        -        ok   NEEDS_HUMAN_REVIEW  False

classification accuracy : 0.960
faithfulness rate       : 1.000   (no hallucinated citations; must be 1.000)
cited-rule accuracy     : 0.960
no-match -> human review: 1.000
retrieval recall@3      : 1.000
invented citations caught by guard: 0

EVAL PASS (accuracy >= 0.7, faithfulness == 1.0, no-match -> human review == 1.0)
```

E-24 (two issues: corrosive without ADR class *and* a copied trademark) is the deliberate hard case; the offline
stand-in picks the IP issue instead of the more safety-critical HAZMAT one. Note it still cites a *real, retrieved*
rule, so faithfulness is unaffected. Adversarial check (Gate 7): with a model that **always invents `R-999`**, the
faithfulness rate stays 1.000 and 25/25 invented citations are caught.

Gemini run: *not yet recorded — no `GOOGLE_API_KEY` on the build machine.* Run
`LLM_PROVIDER=gemini python -m app.eval --out eval_gemini.json` and paste the summary here.

## Deploy

The container listens on `$PORT` (default 8080), builds the Chroma index on startup into `/tmp/chroma`, and runs as
a non-root user.

```bash
docker build -t regops-triage-agent .
docker run --rm -p 8080:8080 -e LLM_PROVIDER=gemini -e GOOGLE_API_KEY=$GOOGLE_API_KEY regops-triage-agent
```

### GCP Cloud Run (primary)

```bash
gcloud run deploy regops-triage-agent \
  --source . --region europe-central2 \
  --allow-unauthenticated \
  --max-instances 1 --session-affinity \
  --set-env-vars LLM_PROVIDER=gemini,GOOGLE_API_KEY=$GOOGLE_API_KEY
```

`--max-instances 1 --session-affinity` (added to the spec's command) keeps the Gradio UI working: its queue
(`/gradio_api/queue/join` → `/queue/data`) lives in process memory, so a UI session must stay on one instance.
`POST /triage` is stateless and scales freely if you drop the UI. Recommended extras: `--memory 1Gi` (Gradio + Chroma exceed the 512 MiB default under load) and, for anything beyond
a demo, store the key in Secret Manager and use `--set-secrets GOOGLE_API_KEY=google-api-key:latest` instead of a
plain env var. To demo without a key, deploy with `LLM_PROVIDER=offline`.

Verify the deployment (also what Gate 8 checks when `PUBLIC_URL` is set):

```bash
PUBLIC_URL=https://regops-triage-agent-xxxx.a.run.app python scripts/gates.py 8
```

### AWS alternative — App Runner (same Dockerfile)

```bash
aws ecr create-repository --repository-name regops-triage-agent
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
# App Runner runs linux/amd64 only: --platform is required when building on Apple Silicon / ARM
docker build --platform linux/amd64 -t <acct>.dkr.ecr.<region>.amazonaws.com/regops-triage-agent:latest .
docker push <acct>.dkr.ecr.<region>.amazonaws.com/regops-triage-agent:latest
# App Runner console/CLI: source = that ECR image, port 8080,
# env LLM_PROVIDER=gemini and GOOGLE_API_KEY (prefer a Secrets Manager reference), health check path /healthz.
# App Runner has no sticky sessions, so pin one instance for the Gradio UI:
aws apprunner create-auto-scaling-configuration --auto-scaling-configuration-name regops-single \
  --min-size 1 --max-size 1     # pass the returned ARN as --auto-scaling-configuration-arn on create-service
```

Talking point: the identical container runs on Cloud Run, AWS App Runner, or Lambda (container image + a web
adapter) — the provider is a runtime env var, not a code change.

## Build gates

`python scripts/gates.py <N|all>` re-runs each layer's validation gate from the build spec. Gates print
`GATE N PASS` only when every check they can run succeeds; checks impossible on the current machine are printed as
`DEFERRED` (never silently passed).

| Gate | Checks | Status on build machine |
| --- | --- | --- |
| 0 | uvicorn serves, `POST /triage` schema-valid, UI at `/`, 422 on bad input, `docker build` | PASS; docker build **deferred** (docker not installed) |
| 1 | schemas import, `get_llm().invoke("ping")`, embeddings, structured output, Gemini wiring | PASS on `offline`; live Gemini ping **deferred** (no key) |
| 2 | expected rule in top-3; recall@3 ≥ 0.9; persistent index reload | PASS (recall@3 = 1.00) |
| 3 | 3 novel inputs classify into valid enums | PASS |
| 4 | citation always a retrieved id or null (eval set, rogue model, unretrieved real id) | PASS |
| 5 | injected fake/unretrieved/tampered citation → `faithful=False`, `NEEDS_HUMAN_REVIEW` | PASS |
| 6 | end-to-end HTTP on 3 inputs: faithful, schema-valid | PASS |
| 7 | eval accuracy ≥ 0.7, faithfulness = 1.0, no-match → human review, adversarial eval | PASS (0.960 / 1.000) |
| 8 | `pytest` green, docker build (or clean-venv container-equivalent smoke), public URL | PASS (54 tests, smoke OK); public URL **deferred** until deployed |

## Layout

```
app/
  main.py          FastAPI app + Gradio UI mounted at /
  schemas.py       Pydantic contracts: Exception (ExceptionIn), TriageResult, RuleHit, Severity, Status
  llm.py           get_llm() / get_embeddings() factory + offline provider
  rag.py           rules.jsonl loader, fingerprinted persistent Chroma index, retrieve()
  pipeline.py      classify → retrieve → decide → verify orchestration
  stages/          classify.py · decide.py · verify.py
  eval.py          python -m app.eval
data/              rules.jsonl (18 rules) · eval_set.jsonl (25 labeled cases)
tests/             test_pipeline.py (54 tests incl. guard, no-match, injection)
scripts/gates.py   build-spec validation gates
```
