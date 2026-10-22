"""Rule knowledge base: load data/rules.jsonl, build a persistent Chroma index, retrieve RuleHits.

Each RuleIndex reads rules.jsonl exactly once; retrieval and the rulebook lookups used by decide /
verify (`get_rule`) are served from that same snapshot, so a rule's text and its action can never
come from different versions of the file. The Chroma collection is named by (embedding model,
rules-file fingerprint): a changed rulebook or model gets a fresh collection, so a stale index can
never serve rules that no longer exist, and processes on different rule versions sharing one
CHROMA_DIR never overwrite each other's collection.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from pydantic import BaseModel

from app.llm import embeddings_model_name, get_embeddings
from app.schemas import RuleHit

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "data" / "rules.jsonl"
log = logging.getLogger(__name__)


class Rule(BaseModel):
    rule_id: str
    category: str
    text: str
    action: str


def _parse_rules(raw: bytes, path: Path) -> list[Rule]:
    rules = [Rule.model_validate_json(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len({r.rule_id for r in rules}) != len(rules):
        raise ValueError(f"Duplicate rule_id in {path}")
    return rules


def load_rules(path: Path = RULES_PATH) -> list[Rule]:
    return _parse_rules(path.read_bytes(), path)


def _document(rule: Rule) -> str:
    return f"{rule.category.replace('_', ' ').lower()}: {rule.text} Action: {rule.action}"


class RuleIndex:
    def __init__(self, rules_path: Path = RULES_PATH, chroma_dir: str | None = None) -> None:
        self.rules_path = rules_path
        raw = rules_path.read_bytes()  # single read: fingerprint, index and rulebook share this snapshot
        self.rules = _parse_rules(raw, rules_path)
        self.by_id = {r.rule_id: r for r in self.rules}
        self.embeddings = get_embeddings()
        model = embeddings_model_name()
        self.fingerprint = hashlib.sha256(raw + model.encode()).hexdigest()
        self.client = chromadb.PersistentClient(
            path=chroma_dir or os.getenv("CHROMA_DIR", str(ROOT / ".chroma")),
            settings=Settings(anonymized_telemetry=False),
        )
        self.name = f"rules_{hashlib.sha256(model.encode()).hexdigest()[:12]}_{self.fingerprint[:12]}"
        self.collection = self._load_or_build()

    def _load_or_build(self):
        try:
            existing = self.client.get_collection(self.name, embedding_function=None)
            if existing.count() == len(self.rules):
                return existing
            self.client.delete_collection(self.name)  # partial build (e.g. crashed mid-add)
        except NotFoundError:
            pass
        collection = self.client.create_collection(
            self.name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "fingerprint": self.fingerprint},
        )
        docs = [_document(r) for r in self.rules]
        collection.add(
            ids=[r.rule_id for r in self.rules],
            documents=docs,
            embeddings=self.embeddings.embed_documents(docs),
            metadatas=[{"category": r.category, "text": r.text, "action": r.action} for r in self.rules],
        )
        return collection

    def _query(self, text: str, k: int) -> dict:
        return self.collection.query(
            query_embeddings=[self.embeddings.embed_query(text)],
            n_results=k,
            include=["distances"],
        )

    def retrieve(self, text: str, k: int = 3) -> list[RuleHit]:
        k = max(1, min(k, len(self.rules)))
        try:
            res = self._query(text, k)
        except NotFoundError:  # collection removed underneath us (e.g. CHROMA_DIR wiped): rebuild once
            log.warning("rule collection %s disappeared; rebuilding", self.name)
            self.collection = self._load_or_build()
            res = self._query(text, k)
        # Rule text comes from this index's snapshot, the same one get_rule() serves to decide/verify.
        return [
            RuleHit(rule_id=rid, text=self.by_id[rid].text, score=round(1.0 - float(dist), 4))
            for rid, dist in zip(res["ids"][0], res["distances"][0])
        ]


_index: RuleIndex | None = None
_index_key: tuple | None = None
_lock = threading.Lock()


def get_index() -> RuleIndex:
    """Process-wide index, rebuilt if the embedding model or CHROMA_DIR changed since last call."""
    global _index, _index_key
    key = (embeddings_model_name(), os.getenv("CHROMA_DIR"))
    with _lock:
        if _index is None or _index_key != key:
            _index, _index_key = RuleIndex(), key
        return _index


def get_rule(rule_id: str) -> Rule | None:
    """Look up a rule in the same snapshot that retrieval serves."""
    return get_index().by_id.get(rule_id)


def retrieve(text: str, k: int = 3) -> list[RuleHit]:
    return get_index().retrieve(text, k)
