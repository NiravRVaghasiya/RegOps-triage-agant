"""Provider-agnostic LLM + embeddings factory.

Every stage obtains its model through `get_llm()` / `get_embeddings()`; nothing else in the code
base imports a provider SDK. Selection is driven by env vars (see .env.example):

    LLM_PROVIDER         gemini (default) | openai | anthropic | ollama | offline
    EMBEDDINGS_PROVIDER  defaults to LLM_PROVIDER (anthropic has no embeddings API -> offline)

`offline` is a deterministic, dependency-free stand-in used for CI, the test suite and key-less
demos. It is NOT a language model: it reads the tagged prompt blocks the stages emit
(<exception>, <category>, <classification>, <rule>) and answers by lexical similarity using
`OfflineHashEmbeddings`. It implements the same LangChain interfaces (`bind_tools`,
`with_structured_output`, `invoke`) so the pipeline code path is identical for every provider.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import uuid
from typing import Any, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

load_dotenv(override=False)
log = logging.getLogger(__name__)

PROVIDERS = ("gemini", "openai", "anthropic", "ollama", "offline")

# Defaults are overridable per provider. The spec's gemini-1.5-flash / text-embedding-004 have been
# retired by Google, so the current Flash / embedding models are the defaults (see README §Tradeoffs).
DEFAULT_MODELS = {
    "gemini": ("GEMINI_MODEL", "gemini-2.5-flash"),
    "openai": ("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_MODEL", "claude-haiku-5-5"),
    "ollama": ("OLLAMA_MODEL", "llama3.1"),
}
DEFAULT_EMBEDDING_MODELS = {
    "gemini": ("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
    "openai": ("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    "ollama": ("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
}


class ProviderConfigError(RuntimeError):
    """Raised when the selected provider is unknown or missing credentials/packages."""


def llm_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in PROVIDERS:
        raise ProviderConfigError(f"Unknown LLM_PROVIDER={provider!r}; expected one of {PROVIDERS}")
    return provider


def embeddings_provider() -> str:
    provider = os.getenv("EMBEDDINGS_PROVIDER", "").strip().lower()
    if not provider:
        provider = llm_provider()
        if provider == "anthropic":
            log.warning("Anthropic has no embeddings API; using offline embeddings (set EMBEDDINGS_PROVIDER).")
            provider = "offline"
    if provider not in PROVIDERS or provider == "anthropic":
        raise ProviderConfigError(f"Unsupported EMBEDDINGS_PROVIDER={provider!r}")
    return provider


def embeddings_model_name() -> str:
    """Stable identifier of the active embedding model (used to fingerprint the vector index)."""
    provider = embeddings_provider()
    if provider == "offline":
        return f"offline/hash-{OfflineHashEmbeddings.DIM}"
    env, default = DEFAULT_EMBEDDING_MODELS[provider]
    return f"{provider}/{os.getenv(env, default)}"


def _require_key(var: str, provider: str) -> None:
    if not os.getenv(var):
        raise ProviderConfigError(f"LLM_PROVIDER={provider} requires {var} (set it in .env or the environment)")


def _import_error(pkg: str, provider: str) -> ProviderConfigError:
    return ProviderConfigError(f"LLM_PROVIDER={provider} requires `pip install {pkg}`")


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    provider = llm_provider()
    if provider == "offline":
        return OfflineChatModel()
    env, default = DEFAULT_MODELS[provider]
    model = os.getenv(env, default)
    if provider == "gemini":
        _require_key("GOOGLE_API_KEY", provider)
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=3)
    if provider == "openai":
        _require_key("OPENAI_API_KEY", provider)
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise _import_error("langchain-openai", provider) from exc
        return ChatOpenAI(model=model, temperature=temperature, max_retries=3)
    if provider == "anthropic":
        _require_key("ANTHROPIC_API_KEY", provider)
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise _import_error("langchain-anthropic", provider) from exc
        # Current Claude models reject non-default sampling parameters, so temperature is not forwarded;
        # determinism comes from structured output and enum-constrained tools instead.
        return ChatAnthropic(model=model, max_retries=3)
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise _import_error("langchain-ollama", provider) from exc
    return ChatOllama(model=model, temperature=temperature)


def get_embeddings() -> Embeddings:
    provider = embeddings_provider()
    if provider == "offline":
        return OfflineHashEmbeddings()
    env, default = DEFAULT_EMBEDDING_MODELS[provider]
    model = os.getenv(env, default)
    if provider == "gemini":
        _require_key("GOOGLE_API_KEY", provider)
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model)
    if provider == "openai":
        _require_key("OPENAI_API_KEY", provider)
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise _import_error("langchain-openai", provider) from exc
        return OpenAIEmbeddings(model=model)
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise _import_error("langchain-ollama", provider) from exc
    return OllamaEmbeddings(model=model)


# --------------------------------------------------------------------------------------------------
# Offline provider
# --------------------------------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """a an and any are as at be been but by can cannot for from has have in into is it its may must
    no not of on or our such that the their them then there these this to under was were which with
    without e g eg etc per all so also any being they we you your item items product products listing
    listings listed vendor seller submitted submission""".split()
)


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ers", "es", "ed", "er", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _features(text: str) -> dict[str, float]:
    feats: dict[str, float] = {}
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]
    for w in words:
        s = _stem(w)
        feats["w:" + s] = feats.get("w:" + s, 0.0) + 1.0
        padded = f"^{s}$"
        for i in range(len(padded) - 3):  # character 4-grams give tolerance to morphology
            g = "c:" + padded[i : i + 4]
            feats[g] = feats.get(g, 0.0) + 0.25
    return feats


class OfflineHashEmbeddings(Embeddings):
    """Deterministic signed feature-hashing embeddings (word stems + char 4-grams), L2-normalised."""

    DIM = 1024

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for feat, weight in _features(text).items():
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.DIM
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * math.sqrt(weight)  # sublinear term frequency
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


_TAG = r"<{tag}\b([^>]*)>(.*?)</{tag}>"
_ATTR = re.compile(r'(\w+)="([^"]*)"')


def _blocks(text: str, tag: str) -> list[tuple[dict[str, str], str]]:
    return [
        (dict(_ATTR.findall(attrs)), body.strip())
        for attrs, body in re.findall(_TAG.format(tag=tag), text, flags=re.S)
    ]


class OfflineChatModel(BaseChatModel):
    """Deterministic stand-in chat model; see module docstring. Supports tool calling."""

    # Minimum lexical similarity for the offline model to commit to a category / rule.
    category_threshold: float = 0.12
    rule_threshold: float = 0.10

    @property
    def _llm_type(self) -> str:
        return "offline-deterministic"

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Optional[Any] = None, **kwargs: Any):
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        if not tools:
            reply = "pong (offline deterministic model)" if prompt.strip().lower() == "ping" else "ok"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])
        name, args = self._answer(prompt, [t["function"] for t in tools])
        call = {"name": name, "args": args, "id": f"call_{uuid.uuid5(uuid.NAMESPACE_OID, prompt).hex[:12]}"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])

    def _answer(self, prompt: str, tools: list[dict]) -> tuple[str, dict]:
        emb = OfflineHashEmbeddings()
        exception = " ".join(body for _, body in _blocks(prompt, "exception"))
        query = emb.embed_query(exception)
        tool_names = [t["name"] for t in tools]
        params = tools[0].get("parameters", {}).get("properties", {})

        if len(tools) == 1 and {"category", "severity"} <= params.keys():
            best = ("NO_MATCH", "LOW", 0.0)
            no_match_severity = "LOW"
            for attrs, desc in _blocks(prompt, "category"):
                sim = _cosine(query, emb.embed_query(desc))
                if attrs.get("name") == "NO_MATCH":
                    no_match_severity = attrs.get("severity", "LOW")
                    sim -= 0.02  # tie-break toward a concrete category when equally similar
                if sim > best[2]:
                    best = (attrs.get("name", "NO_MATCH"), attrs.get("severity", "MEDIUM"), sim)
            if best[2] < self.category_threshold:
                best = ("NO_MATCH", no_match_severity, best[2])
            return tool_names[0], {"category": best[0], "severity": best[1]}

        if "cite_rule" not in tool_names and "escalate_to_human" not in tool_names:
            # Unknown schema: fill each field with its first enum value or an empty string.
            return tool_names[0], {k: (v.get("enum") or [""])[0] for k, v in params.items()}

        category = next((a.get("category") for a, _ in _blocks(prompt, "classification")), None)
        candidates = []
        for attrs, body in _blocks(prompt, "rule"):
            if category and category != "NO_MATCH" and attrs.get("category") == category:
                candidates.append((_cosine(query, emb.embed_query(body)), attrs.get("id")))
        candidates.sort(reverse=True)
        if candidates and candidates[0][0] >= self.rule_threshold and "cite_rule" in tool_names:
            sim, rule_id = candidates[0]
            return "cite_rule", {
                "rule_id": rule_id,
                "rationale": f"Rule {rule_id} matches the exception's category ({category}) "
                f"with lexical similarity {sim:.2f} (offline deterministic model).",
            }
        return "escalate_to_human" if "escalate_to_human" in tool_names else tool_names[0], {
            "reason": "No retrieved rule both matches the classified category "
            f"({category}) and is lexically similar enough to the exception (offline deterministic model)."
        }
