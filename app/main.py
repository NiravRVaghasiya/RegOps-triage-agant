"""FastAPI app exposing POST /triage, with a Gradio demo UI mounted at /."""

import logging
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException

from app import rag
from app.llm import ProviderConfigError, embeddings_model_name, llm_provider
from app.pipeline import run_pipeline
from app.schemas import ExceptionIn, TriageResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:  # warm the rule index so the first request is fast; never block startup on it
        rag.get_index()
        log.info("rule index ready (provider=%s, embeddings=%s)", llm_provider(), embeddings_model_name())
    except Exception as exc:  # noqa: BLE001
        log.error("rule index warm-up failed: %s", exc)
    yield


app = FastAPI(title="RegOps Triage Agent", version="1.0.0", lifespan=lifespan)


def run_triage(text: str) -> TriageResult:
    try:
        return run_pipeline(text)
    except ProviderConfigError as exc:
        raise HTTPException(status_code=503, detail=f"LLM provider misconfigured: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - upstream LLM/embedding failure; never fabricate a result
        log.exception("triage failed")
        raise HTTPException(status_code=502, detail=f"Triage pipeline failed: {type(exc).__name__}") from exc


@app.get("/healthz")
def healthz() -> dict:
    try:
        provider = llm_provider()
    except ProviderConfigError as exc:
        provider = f"invalid ({exc})"
    return {"status": "ok", "llm_provider": provider}


@app.post("/triage", response_model=TriageResult)
def triage(body: ExceptionIn) -> TriageResult:
    return run_triage(body.text)


EXAMPLES = [
    "Vendor submitted a power bank listing but attached no battery safety test documentation.",
    "This herbal capsule listing claims it cures diabetes and reverses heart disease.",
    "An aerosol spray deodorant was submitted without any transport hazard class or UN number.",
    "Vendor emailed asking what time the warehouse closes for drop-off on Fridays.",
]


def _ui_triage(text: str) -> tuple[str, dict]:
    if not text or not text.strip():
        return "Enter an exception to triage.", {}
    try:
        result = run_triage(text.strip())
    except HTTPException as exc:
        return f"**Error:** {exc.detail}", {}
    badge = "✅" if result.faithful else "⚠️"
    summary = (f"### {badge} {result.status.value}\n**Category:** {result.category} · "
               f"**Severity:** {result.severity.value} · **Cited rule:** {result.cited_rule_id or '—'}\n\n"
               f"**Recommended action:** {result.recommended_action or 'Route to a human reviewer.'}")
    return summary, result.model_dump(mode="json")


with gr.Blocks(title="RegOps Triage Agent") as demo:
    gr.Markdown("# RegOps Triage Agent\n`classify → retrieve → decide → verify` — refuses to act on a rule it "
                "cannot cite.")
    inp = gr.Textbox(label="Exception text", lines=4, placeholder="Describe the data-quality / vendor exception...")
    btn = gr.Button("Triage", variant="primary")
    summary = gr.Markdown()
    out = gr.JSON(label="TriageResult")
    gr.Examples(EXAMPLES, inputs=inp)
    btn.click(_ui_triage, inputs=inp, outputs=[summary, out])
    inp.submit(_ui_triage, inputs=inp, outputs=[summary, out])

# Mount last so the API routes above take precedence over the UI's catch-all path.
app = gr.mount_gradio_app(app, demo, path="/")
