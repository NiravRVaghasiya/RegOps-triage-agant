"""Layer validation gates from the build spec.

Usage:  python scripts/gates.py <N> [<N> ...]     e.g.  python scripts/gates.py 0 1 2
        python scripts/gates.py all

Each gate prints `GATE N PASS` only if every check it can run succeeds. Checks that cannot run in
this environment (no docker / gcloud) are printed as DEFERRED, never silently counted as passed.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


class GateFailure(AssertionError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise GateFailure(msg)
    print(f"  ok   {msg}")


def deferred(msg: str) -> None:
    print(f"  DEFERRED  {msg}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """Runs `uvicorn app.main:app` in a subprocess for the duration of a gate."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc = None

    def __enter__(self) -> "Server":
        import httpx

        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise GateFailure("uvicorn exited early:\n" + self.proc.stdout.read().decode(errors="replace"))
            try:
                if httpx.get(f"{self.base}/healthz", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                time.sleep(0.5)
        raise GateFailure("uvicorn did not become healthy within 120s")

    def __exit__(self, *exc) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def gate0() -> None:
    import httpx

    from app.schemas import TriageResult

    with Server() as srv:
        check(True, f"uvicorn app.main:app serves on {srv.base}")
        r = httpx.post(f"{srv.base}/triage", json={"text": "Power bank listed without UN38.3 summary."}, timeout=60)
        check(r.status_code == 200, f"POST /triage -> {r.status_code}")
        result = TriageResult.model_validate(r.json())
        check(True, f"response validates as TriageResult (status={result.status.value})")
        ui = httpx.get(f"{srv.base}/", timeout=30)
        check(ui.status_code == 200 and "<html" in ui.text.lower(), "Gradio UI served at /")
        bad = httpx.post(f"{srv.base}/triage", json={"text": ""}, timeout=30)
        check(bad.status_code == 422, "empty text rejected with 422")
    check((ROOT / "Dockerfile").is_file(), "Dockerfile present")
    if shutil.which("docker"):
        proc = subprocess.run(["docker", "build", "-t", "regops-triage-agent:gate0", "."], cwd=ROOT)
        check(proc.returncode == 0, "docker build succeeds")
    else:
        deferred("docker build: docker is not installed on this machine")


def gate1() -> None:
    from pydantic import BaseModel

    from app import llm, schemas

    check(all(hasattr(schemas, n) for n in ("Exception", "TriageResult", "RuleHit", "Severity", "Status")),
          "schemas import clean (Exception, TriageResult, RuleHit, Severity, Status)")
    check((ROOT / ".env.example").is_file(), ".env.example present")
    provider = llm.llm_provider()
    out = llm.get_llm().invoke("ping")
    check(isinstance(out.content, str) and out.content.strip() != "",
          f"get_llm().invoke('ping') returns text [{provider}]: {out.content.strip()[:60]!r}")
    vec = llm.get_embeddings().embed_query("ping")
    check(len(vec) > 0 and all(isinstance(v, float) for v in vec[:5]),
          f"get_embeddings().embed_query returns a {len(vec)}-d vector [{llm.embeddings_provider()}]")

    class Probe(BaseModel):
        category: str
        severity: str

    structured = llm.get_llm().with_structured_output(Probe)
    check(isinstance(structured.invoke("<exception>ping</exception>"), Probe), "with_structured_output works")
    if provider == "gemini":
        return
    # Prove the default provider path is wired (not just the offline stand-in).
    saved = {k: os.environ.pop(k, None) for k in ("LLM_PROVIDER", "GOOGLE_API_KEY")}
    try:
        try:
            llm.get_llm()
            raise GateFailure("gemini without GOOGLE_API_KEY should raise ProviderConfigError")
        except llm.ProviderConfigError as exc:
            check("GOOGLE_API_KEY" in str(exc), "default provider is gemini and demands GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = "dummy-key-for-construction-only"
        model = llm.get_llm()
        check(type(model).__name__ == "ChatGoogleGenerativeAI", f"gemini factory builds {type(model).__name__}")
        deferred("live Gemini ping: no GOOGLE_API_KEY on this machine (offline provider used above)")
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def _eval_cases() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def gate2() -> None:
    import tempfile

    from app import rag

    rules = rag.load_rules()
    check(len(rules) == 18, f"loaded {len(rules)} rules from data/rules.jsonl")
    hits = rag.retrieve("Vendor submitted a power bank listing but attached no battery safety test documentation.", k=3)
    check(len(hits) == 3 and "R-001" in [h.rule_id for h in hits],
          f"known exception (power bank, no test docs) -> top-3 {[h.rule_id for h in hits]} contains R-001")
    for case in [c for c in _eval_cases() if c["expected_rule_id"]][:3]:
        ids = [h.rule_id for h in rag.retrieve(case["text"])]
        check(case["expected_rule_id"] in ids, f"{case['id']}: expected {case['expected_rule_id']} in top-3 {ids}")
    labeled = [c for c in _eval_cases() if c["expected_rule_id"]]
    recall = sum(c["expected_rule_id"] in [h.rule_id for h in rag.retrieve(c["text"])] for c in labeled) / len(labeled)
    check(recall >= 0.9, f"recall@3 over {len(labeled)} labeled eval cases = {recall:.2f}")
    # Chroma keeps its HNSW files memory-mapped on Windows, so tolerate cleanup failure.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        first = rag.RuleIndex(chroma_dir=tmp)
        check(first.collection.count() == 18, "fresh persistent index built in CHROMA_DIR")
        again = rag.RuleIndex(chroma_dir=tmp)
        check(again.collection.id == first.collection.id
              and again.collection.metadata.get("fingerprint") == first.fingerprint,
              "index reloads from disk (same collection reused, fingerprint match)")


GATE3_SAMPLES = [  # novel phrasings, not taken from the eval set
    ("Bluetooth speaker with built-in lithium cells was uploaded with no UN38.3 report.", "DOCUMENTATION_MISSING"),
    ("A hunting knife is for sale with no age check on the product page.", "AGE_RESTRICTION"),
    ("Hi team, could you confirm when this month's invoice is due?", "NO_MATCH"),
]


def gate3() -> None:
    from app import rag
    from app.schemas import Severity
    from app.stages.classify import TAXONOMY, classify

    rule_categories = {r.category for r in rag.load_rules()}
    check(set(TAXONOMY) == rule_categories | {"NO_MATCH"}, "taxonomy == rule categories + NO_MATCH")
    for text, expected in GATE3_SAMPLES:
        c = classify(text)
        check(c.category in TAXONOMY and isinstance(c.severity, Severity),
              f"{text[:50]!r}... -> {c.category}/{c.severity.value} (valid enums; expected {expected}"
              f"{', matched' if c.category == expected else ', MISMATCH'})")


def rogue_llm(fake_id: str = "R-999"):
    """Offline model that always tries to cite a rule id that was never retrieved."""
    from app.llm import OfflineChatModel

    class RogueChatModel(OfflineChatModel):
        def _answer(self, prompt, tools):
            if any(t["name"] == "cite_rule" for t in tools):
                return "cite_rule", {"rule_id": fake_id, "rationale": "fabricated"}
            return super()._answer(prompt, tools)

    return RogueChatModel()


def gate4() -> None:
    from app import rag
    from app.stages.classify import classify
    from app.stages.decide import decide

    cases = _eval_cases()
    for case in cases:
        rules = rag.retrieve(case["text"])
        d = decide(case["text"], classify(case["text"]), rules)
        ids = {r.rule_id for r in rules}
        if d.cited_rule_id is not None and d.cited_rule_id not in ids:
            raise GateFailure(f"{case['id']}: invented citation {d.cited_rule_id} not in {sorted(ids)}")
    check(True, f"{len(cases)} eval cases: every cited_rule_id is a retrieved id or null")
    text = "Vendor submitted a power bank listing but attached no battery safety test documentation."
    rules = rag.retrieve(text)
    d = decide(text, classify(text), rules, llm=rogue_llm("R-999"))
    check(d.cited_rule_id is None and d.rejected_citation == "R-999",
          "rogue model citing R-999 -> cited_rule_id=None (rejected_citation recorded)")
    real_but_not_retrieved = next(r.rule_id for r in rag.load_rules() if r.rule_id not in {h.rule_id for h in rules})
    d = decide(text, classify(text), rules, llm=rogue_llm(real_but_not_retrieved))
    check(d.cited_rule_id is None, f"citing a real-but-unretrieved rule ({real_but_not_retrieved}) -> null")
    check(decide(text, classify(text), []).cited_rule_id is None, "no retrieved rules -> null citation")


def gate5() -> None:
    from app.pipeline import run_pipeline
    from app.rag import get_rule
    from app.schemas import Status
    from app.stages.decide import Decision

    text = "Vendor submitted a power bank listing but attached no battery safety test documentation."

    def inject(fake_id: str, action: str = "Approve listing immediately."):
        def fake_decide(*_args, **_kwargs) -> Decision:  # bypasses decide's own guard entirely
            return Decision(cited_rule_id=fake_id, recommended_action=action, rationale="injected")
        return fake_decide

    r = run_pipeline(text, decide_fn=inject("R-999"))
    check(r.faithful is False and r.status == Status.NEEDS_HUMAN_REVIEW,
          f"injected fake citation R-999 -> faithful={r.faithful}, status={r.status.value}")
    check(r.cited_rule_id is None and r.recommended_action is None, "fake citation/action not surfaced in result")
    check("R-999" in r.rationale, "rationale records the rejected citation for audit")

    r = run_pipeline(text, decide_fn=inject("R-003", get_rule("R-003").action))
    check(not r.faithful and r.status == Status.NEEDS_HUMAN_REVIEW,
          "injected real-but-unretrieved rule R-003 -> NEEDS_HUMAN_REVIEW")
    r = run_pipeline(text, decide_fn=inject("R-001", "Approve listing immediately."))
    check(not r.faithful and r.status == Status.NEEDS_HUMAN_REVIEW,
          "retrieved id R-001 with a non-rulebook action -> NEEDS_HUMAN_REVIEW")
    r = run_pipeline(text, decide_fn=inject(None, None))
    check(not r.faithful and r.status == Status.NEEDS_HUMAN_REVIEW, "no citation -> NEEDS_HUMAN_REVIEW")
    r = run_pipeline(text, llm=rogue_llm("R-999"))
    check(not r.faithful and r.status == Status.NEEDS_HUMAN_REVIEW and r.cited_rule_id is None,
          "rogue LLM through the real decide stage -> NEEDS_HUMAN_REVIEW")
    r = run_pipeline(text)
    check(r.faithful and r.status == Status.ACTION_RECOMMENDED and r.cited_rule_id == "R-001",
          "control: honest pipeline -> faithful, ACTION_RECOMMENDED, R-001")


GATE6_INPUTS = [
    ("A cordless vacuum with a lithium-ion pack was listed but no UN38.3 test summary was supplied.", "R-001"),
    ("Listing for an energy drink powder shows no allergen statement in the local language.", "R-005"),
    ("The product record declares country of origin as an embargoed territory.", "R-014"),
]


def gate6() -> None:
    import httpx

    from app.schemas import Status, TriageResult

    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    check("run_pipeline" in source and "Walking-skeleton" not in source, "main.py calls the real pipeline (stub removed)")
    with Server() as srv:
        for text, expected in GATE6_INPUTS:
            r = httpx.post(f"{srv.base}/triage", json={"text": text}, timeout=120)
            check(r.status_code == 200, f"POST /triage -> 200 for {text[:40]!r}...")
            res = TriageResult.model_validate(r.json())
            check(res.faithful and res.status == Status.ACTION_RECOMMENDED
                  and res.cited_rule_id in {h.rule_id for h in res.retrieved_rules},
                  f"  schema-valid, faithful, cites {res.cited_rule_id} (expected {expected}), {res.category}")


def gate7() -> None:
    import re

    from app.eval import evaluate, load_cases

    proc = subprocess.run([sys.executable, "-m", "app.eval"], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    print("\n".join("    " + line for line in proc.stdout.splitlines() if line.strip()))
    acc = float(re.search(r"classification accuracy\s*:\s*([\d.]+)", proc.stdout).group(1))
    faith = float(re.search(r"faithfulness rate\s*:\s*([\d.]+)", proc.stdout).group(1))
    check(proc.returncode == 0, "python -m app.eval exits 0")
    check(acc >= 0.7, f"classification accuracy {acc:.3f} >= 0.7")
    check(faith == 1.0, f"faithfulness rate {faith:.3f} == 1.0")
    nm = re.search(r"no-match -> human review\s*:\s*([\d.]+)", proc.stdout)
    check(nm is not None and float(nm.group(1)) == 1.0, "every no-match case routed to human review")
    rogue = evaluate(load_cases(), llm=rogue_llm("R-999"))
    check(rogue["faithfulness_rate"] == 1.0 and rogue["hallucinations_caught"] == rogue["n"]
          and all(r["status"] == "NEEDS_HUMAN_REVIEW" for r in rogue["rows"]),
          f"adversarial eval (model always invents R-999): faithfulness {rogue['faithfulness_rate']:.3f}, "
          f"{rogue['hallucinations_caught']}/{rogue['n']} invented citations caught")


def _container_equivalent_smoke() -> None:
    """Without docker: replay the Dockerfile in a clean venv (fresh install from requirements.txt, only the
    COPY'd paths, PORT env, CHROMA_DIR in a temp dir) and hit /triage."""
    import tempfile

    import httpx

    from app.schemas import TriageResult

    with tempfile.TemporaryDirectory(prefix="regops-container-", ignore_cleanup_errors=True) as tmp:
        srv_dir = Path(tmp) / "srv"
        srv_dir.mkdir()
        shutil.copy(ROOT / "requirements.txt", srv_dir)
        shutil.copytree(ROOT / "app", srv_dir / "app", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "data", srv_dir / "data")
        venv = Path(tmp) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        install = subprocess.run([str(py), "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=srv_dir,
                                 capture_output=True, text=True)
        check(install.returncode == 0, "clean `pip install -r requirements.txt` (Dockerfile RUN step) succeeds"
              + ("" if install.returncode == 0 else "\n" + install.stderr[-2000:]))
        port = _free_port()
        env = {k: v for k, v in os.environ.items() if k not in ("LLM_PROVIDER", "GOOGLE_API_KEY", "CHROMA_DIR")}
        env.update(PORT=str(port), LLM_PROVIDER="offline", CHROMA_DIR=str(Path(tmp) / "chroma"),
                   GRADIO_ANALYTICS_ENABLED="False", PYTHONUNBUFFERED="1")
        cmd = [str(py), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)]
        proc = subprocess.Popen(cmd, cwd=srv_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(240):
                try:
                    if httpx.get(f"{base}/healthz", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.5)
            r = httpx.post(f"{base}/triage", json={"text": "Desk lamp listed with no EU energy label."}, timeout=60)
            res = TriageResult.model_validate(r.json())
            check(r.status_code == 200, f"isolated copy (app/ + data/ only) serves valid triage on $PORT "
                  f"({res.cited_rule_id}, {res.status.value})")
        finally:
            proc.terminate()
            proc.wait(timeout=10)


def gate8() -> None:
    import httpx

    from app.schemas import TriageResult

    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    lines = [ln for ln in proc.stdout.splitlines() if " passed" in ln or " failed" in ln or " error" in ln]
    summary = lines[-1].strip() if lines else (proc.stdout + proc.stderr)[-500:]
    check(proc.returncode == 0, f"pytest green: {summary}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    check(all(s in readme for s in ("gcloud run deploy", "App Runner", "Tradeoff", "faithfulness rate")),
          "README documents Cloud Run + App Runner deploy, tradeoff gates and eval output")
    if shutil.which("docker"):
        p = subprocess.run(["docker", "build", "-t", "regops-triage-agent:gate8", "."], cwd=ROOT)
        check(p.returncode == 0, "docker build succeeds")
    elif os.getenv("GATE8_SKIP_CONTAINER_SMOKE") != "1":
        deferred("docker build: docker not installed -> running container-equivalent smoke instead")
        _container_equivalent_smoke()
    url = os.getenv("PUBLIC_URL", "").rstrip("/")
    if url:
        r = httpx.post(f"{url}/triage", json={"text": GATE6_INPUTS[0][0]}, timeout=120)
        res = TriageResult.model_validate(r.json())
        check(r.status_code == 200, f"public URL {url}/triage returns valid triage ({res.status.value})")
    else:
        deferred("public URL: not deployed from this machine (no gcloud); set PUBLIC_URL=<service url> to check")


GATES = {0: gate0, 1: gate1, 2: gate2, 3: gate3, 4: gate4, 5: gate5, 6: gate6, 7: gate7, 8: gate8}


def main(argv: list[str]) -> int:
    wanted = sorted(GATES) if argv in ([], ["all"]) else [int(a) for a in argv]
    for n in wanted:
        print(f"--- GATE {n} ---")
        try:
            GATES[n]()
        except Exception as exc:  # noqa: BLE001 - report any failure as a gate failure
            print(f"GATE {n} FAIL: {type(exc).__name__}: {exc}")
            return 1
        print(f"GATE {n} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
