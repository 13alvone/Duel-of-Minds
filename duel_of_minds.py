#!/usr/bin/env python3
import argparse
import hashlib
import http.client
import json
import logging
import math
import os
import random
import re
import signal
import socket
import sqlite3
import sys
import time
import urllib.request
from dataclasses import dataclass
from array import array
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from topics import Theme, TopicGraph

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib  # type: ignore

# Optional scientific stacks
try:
    import numpy as _np  # type: ignore

    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# Optional: ONNX runtime + tokenizers for MiniLM embeddings
try:
    import onnxruntime as ort  # type: ignore

    _HAS_ONNXRUNTIME = True
except Exception:
    _HAS_ONNXRUNTIME = False

try:
    from tokenizers import Tokenizer  # type: ignore

    _HAS_TOKENIZERS = True
except Exception:
    _HAS_TOKENIZERS = False

# Local helpers
from migrations import run_migrations

# Optional: llama.cpp backend
try:
    from llama_cpp import Llama  # type: ignore
    _HAS_LLAMA = True
except Exception:
    _HAS_LLAMA = False

# Optional: OpenAI-compatible HTTP backend (local server)
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
MODELS_CONFIG_PATH = os.path.join(CONFIG_DIR, "models.toml")

DEFAULT_TOPIC_GRAPH = os.path.abspath(
    os.path.join(BASE_DIR, "data", "topics", "existential_pivots.yaml")
)

_ZERO_NETWORK_GUARD_ENABLED = False
_REQUESTS_GUARD_ACTIVE = False
_MODEL_CHECKSUM_CACHE: Optional[Dict[str, str]] = None


def _zero_network_error() -> RuntimeError:
    return RuntimeError(
        "Zero-network mode is enabled; outbound HTTP is blocked. "
        "Rerun with --zero-network off to permit HTTP backends."
    )


def _raise_zero_network_error(*_args: Any, **_kwargs: Any) -> None:
    raise _zero_network_error()


def enable_zero_network_guard() -> None:
    """Install runtime guards that prevent outbound HTTP access."""

    global _ZERO_NETWORK_GUARD_ENABLED, _REQUESTS_GUARD_ACTIVE

    if _ZERO_NETWORK_GUARD_ENABLED:
        return

    _ZERO_NETWORK_GUARD_ENABLED = True

    # Guard urllib (used by many stdlib callers)
    urllib.request.urlopen = _raise_zero_network_error  # type: ignore[assignment]

    def _guarded_http_request(
        self: http.client.HTTPConnection,
        method: str,
        url: str,
        body: Any = None,
        headers: Optional[Dict[str, Any]] = None,
        *,
        encode_chunked: bool = False,
    ) -> None:
        raise _zero_network_error()

    http.client.HTTPConnection.request = _guarded_http_request  # type: ignore[assignment]
    http.client.HTTPSConnection.request = _guarded_http_request  # type: ignore[assignment]

    # Guard low-level socket creation to catch direct attempts
    def _guarded_create_connection(*_args: Any, **_kwargs: Any) -> None:
        raise _zero_network_error()

    socket.create_connection = _guarded_create_connection  # type: ignore[assignment]

    if _HAS_REQUESTS:
        import requests  # type: ignore

        if not _REQUESTS_GUARD_ACTIVE:
            def _guarded_session_request(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> None:
                raise _zero_network_error()

            def _guarded_requests_call(*args: Any, **kwargs: Any) -> None:
                raise _zero_network_error()

            requests.sessions.Session.request = _guarded_session_request  # type: ignore[assignment]
            for verb in ("request", "get", "post", "put", "delete", "head", "options", "patch"):
                setattr(requests, verb, _guarded_requests_call)
            _REQUESTS_GUARD_ACTIVE = True


def _normalize_model_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _load_model_checksums(config_path: str = MODELS_CONFIG_PATH) -> Dict[str, str]:
    """Load model path -> sha256 mappings from config/models.toml."""

    global _MODEL_CHECKSUM_CACHE

    if _MODEL_CHECKSUM_CACHE is not None:
        return _MODEL_CHECKSUM_CACHE

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model checksum config not found at {config_path}. Create it to list authorized models."
        )

    with open(config_path, "rb") as fh:
        raw = tomllib.load(fh)

    entries: Dict[str, str] = {}
    models_section = raw.get("models")

    def _register_entry(entry: Dict[str, Any], index: int) -> None:
        path = entry.get("path")
        checksum = entry.get("sha256")
        if not path or not isinstance(path, str):
            raise ValueError(f"models[{index}] is missing a 'path' string entry in {config_path}.")
        if not checksum or not isinstance(checksum, str):
            raise ValueError(f"models[{index}] is missing a 'sha256' string entry in {config_path}.")
        entries[_normalize_model_path(path)] = checksum.lower()
        # Support optional aliases for quick lookup
        alias = entry.get("alias") or entry.get("id")
        if alias and isinstance(alias, str):
            entries[alias.strip().lower()] = checksum.lower()

    if isinstance(models_section, list):
        for idx, entry in enumerate(models_section):
            if not isinstance(entry, dict):
                raise ValueError(f"models[{idx}] must be a table in {config_path}.")
            _register_entry(entry, idx)
    elif isinstance(models_section, dict):
        for idx, (key, value) in enumerate(models_section.items()):
            if isinstance(value, dict):
                value.setdefault("alias", key)
                _register_entry(value, idx)
            elif isinstance(value, str):
                entries[_normalize_model_path(key)] = value.lower()
            else:
                raise ValueError(f"models.{key} must map to a string hash or table in {config_path}.")
    else:
        raise ValueError(f"config/models.toml must define a 'models' list or table.")

    if not entries:
        raise ValueError(f"No model checksum entries found in {config_path}.")

    _MODEL_CHECKSUM_CACHE = entries
    return entries


def _compute_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def verify_model_checksum(model_path: str) -> None:
    """Verify model binary integrity before loading llama.cpp."""

    normalized_path = _normalize_model_path(model_path)
    checksums = _load_model_checksums()

    expected = checksums.get(normalized_path)
    if expected is None:
        # Allow lookup by alias based on basename for convenience
        expected = checksums.get(os.path.basename(normalized_path))
    if expected is None:
        raise RuntimeError(
            f"No checksum entry found for model '{normalized_path}' in {MODELS_CONFIG_PATH}."
        )

    actual = _compute_sha256(normalized_path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"Model checksum mismatch for '{normalized_path}'. Expected "
            f"{expected.lower()}, computed {actual.lower()}."
        )

    logging.info(f"[i] Verified SHA256 for model {normalized_path}.")


# ---------------------------
# Logging setup
# ---------------------------
def setup_logging(verbosity: int) -> None:
    level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 0:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.debug("[DEBUG] Logger initialized.")


# ---------------------------
# SQLite schema and helpers
# ---------------------------


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    run_migrations(conn)
    return conn

def insert_run(conn: sqlite3.Connection, backend: str, model_name: str,
               persona_a_path: str, persona_b_path: str, params_json: str) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs (started_at_utc, backend, model_name, persona_a_path, persona_b_path, params_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), backend, model_name, persona_a_path, persona_b_path, params_json),
    )
    conn.commit()
    run_id = cur.lastrowid
    logging.info(f"[i] Started run_id={run_id}")
    return run_id

def insert_message(conn: sqlite3.Connection, run_id: int, turn_index: int, speaker: str, content: str) -> None:
    conn.execute(
        "INSERT INTO messages (run_id, turn_index, speaker, content, created_at_utc) VALUES (?, ?, ?, ?, ?)",
        (run_id, turn_index, speaker, content, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()

def fetch_last_turn(conn: sqlite3.Connection, run_id: int) -> int:
    cur = conn.cursor()
    cur.execute("SELECT MAX(turn_index) FROM messages WHERE run_id=?", (run_id,))
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else -1

def fetch_recent_context(conn: sqlite3.Connection, run_id: int, k: int) -> List[Tuple[int, str, str]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT turn_index, speaker, content FROM messages WHERE run_id=? ORDER BY turn_index DESC LIMIT ?",
        (run_id, k),
    )
    rows = cur.fetchall()
    rows.reverse()
    return [(int(row[0]), str(row[1]), str(row[2])) for row in rows]


def insert_growth_metric(
    conn: sqlite3.Connection,
    run_id: int,
    turn_index: int,
    novelty_score: float,
    max_similarity: Optional[float],
    method: str,
    threshold: Optional[float],
    attempt_count: int,
    flagged: bool,
) -> None:
    conn.execute(
        """
        INSERT INTO growth_metrics (run_id, turn_index, novelty_score, max_similarity, method, threshold, attempt_count, flagged, created_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            turn_index,
            float(novelty_score),
            None if max_similarity is None else float(max_similarity),
            method,
            None if threshold is None else float(threshold),
            int(max(attempt_count, 0)),
            1 if flagged else 0,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def upsert_fact(
    conn: sqlite3.Connection,
    run_id: int,
    subject: str,
    predicate: str,
    obj: str,
    turn_id: int,
    confidence: float,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO facts (run_id, subject, predicate, object, turn_id, confidence, created_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, subject, predicate, object)
            DO UPDATE SET
                turn_id=excluded.turn_id,
                confidence=excluded.confidence,
                updated_at_utc=excluded.updated_at_utc
            """,
            (run_id, subject, predicate, obj, turn_id, confidence, timestamp, timestamp),
        )


def upsert_theme(
    conn: sqlite3.Connection,
    run_id: int,
    theme_key: str,
    strength: float,
    turn_id: int,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO themes (run_id, theme, strength, last_seen_turn, created_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, theme)
            DO UPDATE SET
                strength=excluded.strength,
                last_seen_turn=excluded.last_seen_turn,
                updated_at_utc=excluded.updated_at_utc
            """,
            (run_id, theme_key, strength, turn_id, timestamp, timestamp),
        )


def upsert_summary(conn: sqlite3.Connection, run_id: int, step: int, content: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            """
            INSERT INTO summaries (run_id, step, content, created_at_utc)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id, step)
            DO UPDATE SET
                content=excluded.content,
                created_at_utc=excluded.created_at_utc
            """,
            (run_id, step, content, timestamp),
        )


def latest_summary(conn: sqlite3.Connection, run_id: int) -> str:
    cur = conn.cursor()
    cur.execute(
        "SELECT content FROM summaries WHERE run_id=? ORDER BY step DESC LIMIT 1",
        (run_id,),
    )
    row = cur.fetchone()
    return str(row[0]) if row else ""


def fetch_fact_snippets(conn: sqlite3.Connection, run_id: int, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    cur = conn.cursor()
    cur.execute(
        """
        SELECT subject, predicate, object, turn_id, confidence
        FROM facts
        WHERE run_id=?
        ORDER BY confidence DESC, turn_id DESC
        LIMIT ?
        """,
        (run_id, limit),
    )
    return [
        {
            "subject": str(row[0]),
            "predicate": str(row[1]),
            "object": str(row[2]),
            "turn_id": int(row[3]),
            "confidence": float(row[4]),
        }
        for row in cur.fetchall()
    ]


def fetch_theme_rows(conn: sqlite3.Connection, run_id: int, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    cur = conn.cursor()
    cur.execute(
        """
        SELECT theme, strength, last_seen_turn
        FROM themes
        WHERE run_id=?
        ORDER BY strength DESC, last_seen_turn DESC
        LIMIT ?
        """,
        (run_id, limit),
    )
    return [
        {
            "theme": str(row[0]),
            "strength": float(row[1]),
            "last_seen_turn": int(row[2]),
        }
        for row in cur.fetchall()
    ]


def fetch_recent_summaries(conn: sqlite3.Connection, run_id: int, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    cur = conn.cursor()
    cur.execute(
        """
        SELECT step, content
        FROM summaries
        WHERE run_id=?
        ORDER BY step DESC
        LIMIT ?
        """,
        (run_id, limit),
    )
    return [
        {
            "step": int(row[0]),
            "content": str(row[1]),
        }
        for row in cur.fetchall()
    ]


# ---------------------------
# Backends
# ---------------------------
class BackendBase:
    def generate(self, system_prompt: str, user_prompt: str, stop: List[str],
                 temperature: float, top_p: float, max_tokens: int,
                 repeat_penalty: float) -> str:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError

    def supports_embeddings(self) -> bool:
        return False

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError("Embedding backend not available")


class LlamaCppBackend(BackendBase):
    def __init__(self, model_path: str, ctx_size: int, gpu_layers: int, seed: Optional[int]):
        if not _HAS_LLAMA:
            logging.error("[x] llama-cpp-python not available. Install it or choose the openai backend.")
            raise RuntimeError("llama-cpp-python missing.")
        if not os.path.exists(model_path):
            logging.error(f"[x] Model path does not exist: {model_path}")
            raise FileNotFoundError(model_path)
        verify_model_checksum(model_path)
        logging.info(f"[i] Loading llama.cpp model from {model_path} (ctx={ctx_size}, gpu_layers={gpu_layers})")
        self.llm = Llama(model_path=model_path, n_ctx=ctx_size, n_gpu_layers=gpu_layers, seed=seed or -1)

    def generate(self, system_prompt: str, user_prompt: str, stop: List[str],
                 temperature: float, top_p: float, max_tokens: int,
                 repeat_penalty: float) -> str:
        # Use chat format for better steering
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        out = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
            stop=stop if stop else None,
        )
        text = out["choices"][0]["message"]["content"]
        return text.strip()

    def name(self) -> str:
        return "llamacpp"

    def supports_embeddings(self) -> bool:
        return True

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            result = self.llm.create_embedding(input=text)  # type: ignore[arg-type]
            embeddings.append(list(result["data"][0]["embedding"]))
        return embeddings


class OpenAICompatBackend(BackendBase):
    def __init__(self, api_base: str, api_key: str, model_name: str, timeout: int = 120):
        if not _HAS_REQUESTS:
            logging.error("[x] requests not available. Install it or choose the llamacpp backend.")
            raise RuntimeError("requests missing.")
        if _ZERO_NETWORK_GUARD_ENABLED:
            raise _zero_network_error()
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model_name
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str, stop: List[str],
                 temperature: float, top_p: float, max_tokens: int,
                 repeat_penalty: float) -> str:
        if _ZERO_NETWORK_GUARD_ENABLED:
            raise _zero_network_error()
        # Note: many local servers ignore repeat_penalty; still included for parity.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop if stop else None,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.api_base}/chat/completions"
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if resp.status_code != 200:
            logging.error(f"[x] OpenAI-compatible request failed: {resp.status_code} {resp.text[:200]}")
            raise RuntimeError(f"HTTP {resp.status_code}")
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return text.strip()

    def name(self) -> str:
        return f"openai_compat:{self.model}"

    def supports_embeddings(self) -> bool:
        return False


# ---------------------------
# Utilities
# ---------------------------
def read_file(path: str) -> str:
    if not os.path.exists(path):
        logging.error(f"[x] File not found: {path}")
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_system_prompt(
    persona_text: str,
    long_summary: str,
    safety_notes: str,
    topic_guidance: Optional[str] = None,
) -> str:
    # Keep this brief and directive; don’t embed copyrighted corpora here.
    lines = [
        "You are to speak in a consistent philosophical persona.",
        "Constraints:",
        "- Be concise but insightful; avoid filler.",
        "- Prefer original phrasing over quotation.",
        "- Do not reveal private data; do not output long copyrighted passages.",
        "- If asked for biographical facts, be accurate or say you are unsure.",
        "",
        "Long-term context (summarized):",
        long_summary.strip() if long_summary else "(none)",
        "",
        "Persona notes:",
        persona_text.strip(),
        "",
        "Safety notes:",
        safety_notes.strip() if safety_notes else "Do not disclose sensitive or private information.",
    ]
    if topic_guidance:
        lines.extend([
            "",
            "Topic focus:",
            topic_guidance.strip(),
        ])
    return "\n".join(lines)

def make_turn_prompt(
    history: List[Tuple[str, str]],
    next_speaker: str,
    stopwords: List[str],
    topic_guidance: Optional[str] = None,
) -> str:
    # Render short buffer
    rendered = []
    for speaker, content in history:
        rendered.append(f"{speaker}: {content}")
    rendered.append(f"{next_speaker}:")
    prompt = "\n".join(rendered)
    # Ensure stopwords don’t accidentally appear mid-turn by hinting format
    if stopwords:
        prompt += "\n\n(Respond as the next line. Avoid the explicit tokens: " + ", ".join(stopwords) + ")"
    if topic_guidance:
        prompt += "\n\n[Topic focus]\n" + topic_guidance.strip()
    return prompt

def ngram_block(text: str, prior_texts: List[str], n: int) -> str:
    """Remove repeated n-grams present in recent outputs (very simple post-filter)."""
    if n <= 1 or not prior_texts:
        return text
    prior_concat = " ".join(prior_texts)[-8000:]  # limit search window for efficiency
    words = text.split()
    if len(words) < n:
        return text
    filtered_words = []
    for i in range(len(words)):
        # build candidate ngram ending at i
        start = max(0, i - n + 1)
        ngram = " ".join(words[start:i+1])
        if ngram and ngram in prior_concat:
            # skip current token to break the repeated n-gram chain
            continue
        filtered_words.append(words[i])
    return " ".join(filtered_words).strip()

def truncate_tokens(text: str, max_chars: int) -> str:
    return text[:max_chars]

def safe_int(val: str, default: int) -> int:
    try:
        return int(val)
    except Exception:
        return default


def parse_wallclock_arg(value: str) -> Optional[float]:
    """Convert wall-clock duration strings like '24h', '30m', '90s', 'off' into seconds."""
    if value is None:
        raise ValueError("Wall-clock value is required.")
    token = value.strip().lower()
    if token in {"off", "none", "0"}:
        return None
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if token[-1] in units:
        magnitude = token[:-1]
        unit = token[-1]
    else:
        magnitude = token
        unit = "s"
    if not magnitude.isdigit():
        raise ValueError(f"Invalid wall-clock duration: {value}")
    seconds = int(magnitude) * units[unit]
    if seconds <= 0:
        raise ValueError("Wall-clock duration must be positive or 'off'.")
    return float(seconds)


def parse_pause_jitter_arg(value: str) -> Tuple[float, float]:
    """Parse pause jitter bounds expressed as 'min_ms,max_ms' into seconds."""
    if value is None:
        raise ValueError("Pause jitter value is required.")
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Pause jitter must be two comma-separated millisecond values.")
    try:
        min_ms = float(parts[0])
        max_ms = float(parts[1])
    except ValueError as exc:
        raise ValueError("Pause jitter values must be numeric.") from exc
    if min_ms < 0 or max_ms < 0 or max_ms < min_ms:
        raise ValueError("Pause jitter must be non-negative and max >= min.")
    return (min_ms / 1000.0, max_ms / 1000.0)


# ---------------------------
# Memory + vector store plumbing
# ---------------------------


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _vector_to_blob(vector: Sequence[float]) -> bytes:
    arr = array("f", [float(v) for v in vector])
    return arr.tobytes()


def _blob_to_vector(blob: bytes) -> List[float]:
    arr = array("f")
    arr.frombytes(blob)
    return list(arr)


class HashEmbeddingBackend:
    """Deterministic hashing fallback when no high-quality embedder is available."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            values: List[float] = []
            token = text.strip() or "(empty)"
            for index in range(self.dimension):
                digest = hashlib.sha256(f"{token}|{index}".encode("utf-8")).digest()
                int_val = int.from_bytes(digest[:4], "big", signed=False)
                scaled = (int_val / 0xFFFFFFFF) * 2.0 - 1.0
                values.append(float(scaled))
            vectors.append(values)
        return vectors


class OnnxMiniLMEmbedding:
    """Wrapper around an ONNX MiniLM encoder (if local files and deps are present)."""

    def __init__(self, model_path: str) -> None:
        if not (_HAS_ONNXRUNTIME and _HAS_TOKENIZERS and _HAS_NUMPY):
            raise RuntimeError("onnxruntime, tokenizers, and numpy are required for ONNX embeddings.")
        if os.path.isdir(model_path):
            model_file = os.path.join(model_path, "model.onnx")
            tokenizer_file = os.path.join(model_path, "tokenizer.json")
        else:
            base = os.path.dirname(model_path)
            model_file = model_path
            tokenizer_file = os.path.join(base, "tokenizer.json")
        if not os.path.exists(model_file):
            raise FileNotFoundError(model_file)
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(tokenizer_file)
        self._session = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])  # type: ignore[arg-type]
        self._tokenizer = Tokenizer.from_file(tokenizer_file)
        output = self._session.get_outputs()[0]
        try:
            self.dimension = int(output.shape[-1])
        except Exception:
            self.dimension = 384

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            encoded = self._tokenizer.encode(text)
            input_ids = _np.array([encoded.ids], dtype=_np.int64)  # type: ignore[call-arg]
            attention = _np.ones_like(input_ids, dtype=_np.int64)
            outputs = self._session.run(None, {"input_ids": input_ids, "attention_mask": attention})
            vector = outputs[0][0]
            vectors.append(vector.astype(_np.float32).tolist())
        return vectors


class VectorStoreBase:
    def sync(self, run_id: int, turn_index: int, embedding_blob: bytes) -> None:
        raise NotImplementedError

    def search(
        self,
        run_id: int,
        query_embedding: Sequence[float],
        topk: int,
        *,
        exclude_turn: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class SimpleVectorStore(VectorStoreBase):
    """Fallback cosine-similarity store implemented directly over SQLite."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def sync(self, run_id: int, turn_index: int, embedding_blob: bytes) -> None:
        # Data already written to message_embeddings; nothing further required.
        return

    def _fetch_rows(self, run_id: int, limit: int) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT turn_index, speaker, content, embedding FROM message_embeddings WHERE run_id=? ORDER BY turn_index DESC LIMIT ?",
            (run_id, limit),
        )
        rows = []
        for turn_index, speaker, content, blob in cur.fetchall():
            if blob is None:
                continue
            vector = _blob_to_vector(blob)
            rows.append(
                {
                    "turn_index": int(turn_index),
                    "speaker": str(speaker),
                    "content": str(content),
                    "vector": vector,
                }
            )
        return rows

    def search(
        self,
        run_id: int,
        query_embedding: Sequence[float],
        topk: int,
        *,
        exclude_turn: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if topk <= 0:
            return []
        rows = self._fetch_rows(run_id, max(topk * 8, 32))
        query = list(query_embedding)
        results: List[Dict[str, Any]] = []
        for row in rows:
            if exclude_turn is not None and row["turn_index"] == exclude_turn:
                continue
            vector = row["vector"]
            if len(vector) != len(query) or not vector:
                continue
            score = _cosine_similarity(query, vector)
            results.append(
                {
                    "turn_index": row["turn_index"],
                    "speaker": row["speaker"],
                    "content": row["content"],
                    "score": float(score),
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:topk]


class FaissVectorStore(SimpleVectorStore):
    """FAISS-accelerated similarity search if available."""

    def __init__(self, conn: sqlite3.Connection, dim: int) -> None:
        if not (_HAS_FAISS and _HAS_NUMPY):
            raise RuntimeError("FAISS requires faiss and numpy modules.")
        super().__init__(conn)
        self.dim = dim

    def search(
        self,
        run_id: int,
        query_embedding: Sequence[float],
        topk: int,
        *,
        exclude_turn: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if topk <= 0:
            return []
        rows = self._fetch_rows(run_id, max(topk * 8, 64))
        if not rows:
            return []
        matrix = _np.array([row["vector"] for row in rows], dtype=_np.float32)
        if matrix.shape[1] != self.dim:
            return super().search(run_id, query_embedding, topk, exclude_turn=exclude_turn)
        query = _np.array([query_embedding], dtype=_np.float32)
        if query.shape[1] != self.dim:
            return super().search(run_id, query_embedding, topk, exclude_turn=exclude_turn)
        faiss.normalize_L2(matrix)
        faiss.normalize_L2(query)
        index = faiss.IndexFlatIP(self.dim)
        index.add(matrix)
        scores, indices = index.search(query, min(topk, matrix.shape[0]))
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices[0]):
            row = rows[idx]
            if exclude_turn is not None and row["turn_index"] == exclude_turn:
                continue
            results.append(
                {
                    "turn_index": row["turn_index"],
                    "speaker": row["speaker"],
                    "content": row["content"],
                    "score": float(scores[0][rank]),
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:topk]


class SqliteVSSVectorStore(SimpleVectorStore):
    """SQLite-VSS wrapper (if extension available)."""

    def __init__(self, conn: sqlite3.Connection, dim: int) -> None:
        super().__init__(conn)
        self.dim = dim
        try:
            conn.enable_load_extension(True)
            conn.execute("SELECT vss_version()")
        except Exception as exc:
            raise RuntimeError("sqlite-vss extension not available") from exc
        with conn:
            conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings_vss USING vss0(embedding({dim}))")

    def _lookup_rowid(self, run_id: int, turn_index: int) -> Optional[int]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT rowid FROM message_embeddings WHERE run_id=? AND turn_index=?",
            (run_id, turn_index),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None

    def sync(self, run_id: int, turn_index: int, embedding_blob: bytes) -> None:
        rowid = self._lookup_rowid(run_id, turn_index)
        if rowid is None:
            return
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO message_embeddings_vss(rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite3.Binary(embedding_blob)),
                )
        except sqlite3.DatabaseError as exc:
            logging.debug("[DEBUG] sqlite-vss sync failed: %s", exc)

    def search(
        self,
        run_id: int,
        query_embedding: Sequence[float],
        topk: int,
        *,
        exclude_turn: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if topk <= 0:
            return []
        try:
            query_blob = sqlite3.Binary(_vector_to_blob(query_embedding))
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT m.turn_index, m.speaker, m.content, v.distance
                FROM message_embeddings_vss v
                JOIN message_embeddings m ON m.rowid = v.rowid
                WHERE m.run_id=? AND v.embedding MATCH ?
                ORDER BY v.distance ASC
                LIMIT ?
                """,
                (run_id, query_blob, max(topk * 3, topk)),
            )
            rows = cur.fetchall()
        except sqlite3.DatabaseError as exc:
            logging.debug("[DEBUG] sqlite-vss search failed: %s", exc)
            return super().search(run_id, query_embedding, topk, exclude_turn=exclude_turn)
        results: List[Dict[str, Any]] = []
        for turn_index, speaker, content, distance in rows:
            turn = int(turn_index)
            if exclude_turn is not None and turn == exclude_turn:
                continue
            score = 1.0 / (1.0 + float(distance))
            results.append(
                {
                    "turn_index": turn,
                    "speaker": str(speaker),
                    "content": str(content),
                    "score": score,
                }
            )
        if not results:
            return super().search(run_id, query_embedding, topk, exclude_turn=exclude_turn)
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:topk]


class VectorStoreFactory:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, dim: int) -> VectorStoreBase:
        try:
            store = SqliteVSSVectorStore(self.conn, dim)
            logging.info("[i] Using sqlite-vss vector store (%d dims).", dim)
            return store
        except Exception as exc:
            logging.debug("[DEBUG] sqlite-vss unavailable: %s", exc)
        if _HAS_FAISS and _HAS_NUMPY:
            try:
                store = FaissVectorStore(self.conn, dim)
                logging.info("[i] Using FAISS vector store fallback (%d dims).", dim)
                return store
            except Exception as exc:
                logging.debug("[DEBUG] FAISS fallback unavailable: %s", exc)
        logging.info("[i] Using simple cosine vector store (%d dims).", dim)
        return SimpleVectorStore(self.conn)


class MemoryManager:
    def __init__(
        self,
        conn: sqlite3.Connection,
        run_id: int,
        backend: BackendBase,
        db_path: str,
        embedding_model: str,
        recall_topk: int,
        recall_alpha: float,
    ) -> None:
        self.conn = conn
        self.run_id = run_id
        self.backend = backend
        self.db_path = db_path
        self.recall_topk = max(0, recall_topk)
        self.recall_alpha = min(max(recall_alpha, 0.0), 1.0)
        self._embedding_model = (embedding_model or "auto").strip()
        self._embedding_backend = self._init_embedding_backend()
        self._store_factory = VectorStoreFactory(conn)
        self._vector_store: Optional[VectorStoreBase] = None
        self._theme_cache: Dict[str, float] = {}

    def _init_embedding_backend(self) -> Any:
        token = self._embedding_model.lower()
        if token in {"", "off", "none"}:
            logging.info("[i] Embedding pipeline disabled.")
            return None
        if token in {"auto"}:
            if self.backend.supports_embeddings():
                logging.info("[i] Using llama.cpp embeddings via active backend.")
                return self.backend
            pipeline = self._load_default_minilm()
            if pipeline:
                logging.info("[i] Loaded default MiniLM ONNX embeddings.")
                return pipeline
        elif token in {"llamacpp"}:
            if self.backend.supports_embeddings():
                logging.info("[i] Using requested llama.cpp embedding mode.")
                return self.backend
            logging.warning("[!] llama.cpp embeddings requested but backend does not support them.")
        elif token in {"minilm-onnx", "minilm", "onnx"}:
            pipeline = self._load_default_minilm()
            if pipeline:
                logging.info("[i] Loaded MiniLM ONNX embeddings (default search).")
                return pipeline
        if token not in {"auto", "llamacpp", "minilm-onnx", "minilm", "onnx", "off", "none"}:
            candidate = self._embedding_model
            if os.path.exists(candidate):
                try:
                    pipeline = OnnxMiniLMEmbedding(candidate)
                    logging.info("[i] Loaded ONNX embeddings from %s.", candidate)
                    return pipeline
                except Exception as exc:
                    logging.warning("[!] Failed to load ONNX embeddings from %s: %s", candidate, exc)
        if self.backend.supports_embeddings():
            logging.info("[i] Falling back to llama.cpp embeddings.")
            return self.backend
        logging.warning("[!] Falling back to deterministic hash embeddings.")
        return HashEmbeddingBackend()

    def _load_default_minilm(self) -> Optional[OnnxMiniLMEmbedding]:
        candidates: List[str] = []
        env_path = os.environ.get("DOM_MINILM_PATH") or os.environ.get("DOM_MINILM_DIR")
        if env_path:
            candidates.append(env_path)
        repo_path = os.path.join(os.path.dirname(__file__), "data", "models", "minilm-onnx")
        candidates.append(repo_path)
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                return OnnxMiniLMEmbedding(path)
            except Exception as exc:
                logging.debug("[DEBUG] MiniLM load failed at %s: %s", path, exc)
        return None

    def latest_summary(self) -> str:
        return latest_summary(self.conn, self.run_id)

    def record_turn(self, turn_index: int, speaker: str, content: str, theme_key: Optional[str]) -> None:
        if not content.strip():
            return
        self._store_facts(turn_index, speaker, content)
        if theme_key:
            self._update_theme(theme_key, turn_index)
        self._store_embedding(turn_index, speaker, content)

    def record_summary(self, step: int, content: str) -> None:
        if not content.strip():
            return
        upsert_summary(self.conn, self.run_id, step, content)

    def _store_facts(self, turn_index: int, speaker: str, content: str) -> None:
        snippets = self._extract_fact_snippets(content)
        for snippet in snippets:
            confidence = 0.55 + min(len(snippet) / 400.0, 0.35)
            upsert_fact(self.conn, self.run_id, speaker, "asserts", snippet, turn_index, confidence)

    def _update_theme(self, theme_key: str, turn_index: int) -> None:
        strength = self._theme_cache.get(theme_key)
        if strength is None:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT strength FROM themes WHERE run_id=? AND theme=?",
                (self.run_id, theme_key),
            )
            row = cur.fetchone()
            strength = float(row[0]) if row else 0.0
        strength = float(strength) + 1.0
        self._theme_cache[theme_key] = strength
        upsert_theme(self.conn, self.run_id, theme_key, strength, turn_index)

    def _ensure_store(self, dim: int) -> VectorStoreBase:
        if self._vector_store is None:
            self._vector_store = self._store_factory.create(dim)
        return self._vector_store

    def _store_embedding(self, turn_index: int, speaker: str, content: str) -> None:
        if self._embedding_backend is None:
            return
        try:
            vectors = self._embedding_backend.embed_texts([content])  # type: ignore[call-arg]
        except Exception as exc:
            logging.debug("[DEBUG] Embedding generation failed: %s", exc)
            return
        if not vectors or not vectors[0]:
            return
        vector = vectors[0]
        blob = _vector_to_blob(vector)
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO message_embeddings (run_id, turn_index, speaker, content, embedding, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, turn_index)
                DO UPDATE SET speaker=excluded.speaker, content=excluded.content, embedding=excluded.embedding, created_at_utc=excluded.created_at_utc
                """,
                (self.run_id, turn_index, speaker, content, sqlite3.Binary(blob), timestamp),
            )
        store = self._ensure_store(len(vector))
        store.sync(self.run_id, turn_index, blob)

    def embed_text(self, text: str) -> Optional[List[float]]:
        if self._embedding_backend is None:
            return None
        try:
            vectors = self._embedding_backend.embed_texts([text])  # type: ignore[call-arg]
        except Exception as exc:
            logging.debug("[DEBUG] Embed text failed: %s", exc)
            return None
        if not vectors or not vectors[0]:
            return None
        return [float(v) for v in vectors[0]]

    def recent_embeddings(self, limit: int) -> List[Tuple[int, List[float]]]:
        if limit <= 0 or self._embedding_backend is None:
            return []
        cur = self.conn.cursor()
        cur.execute(
            "SELECT turn_index, embedding FROM message_embeddings WHERE run_id=? ORDER BY turn_index DESC LIMIT ?",
            (self.run_id, limit),
        )
        rows = cur.fetchall()
        results: List[Tuple[int, List[float]]] = []
        for row in rows:
            blob = row["embedding"] if isinstance(row, sqlite3.Row) else row[1]
            if blob is None:
                continue
            vector = _blob_to_vector(bytes(blob)) if not isinstance(blob, (bytes, bytearray)) else _blob_to_vector(blob)
            results.append((int(row["turn_index"] if isinstance(row, sqlite3.Row) else row[0]), vector))
        return results

    def semantic_recall(
        self,
        query_text: str,
        *,
        exclude_turn: Optional[int],
        topk: int,
    ) -> List[Dict[str, Any]]:
        if topk <= 0 or not query_text.strip():
            return []
        if self._vector_store is None or self._embedding_backend is None:
            return []
        try:
            vectors = self._embedding_backend.embed_texts([query_text])  # type: ignore[call-arg]
        except Exception as exc:
            logging.debug("[DEBUG] Query embedding failed: %s", exc)
            return []
        if not vectors or not vectors[0]:
            return []
        return self._vector_store.search(self.run_id, vectors[0], topk, exclude_turn=exclude_turn)

    def render_recall_block(self, history: Sequence[Tuple[int, str, str]], next_speaker: str) -> str:
        if self.recall_topk <= 0:
            return ""
        query_turn = history[-1][0] if history else None
        query_text = history[-1][2] if history else ""
        semantic_quota = 0
        if self._vector_store is not None and self._embedding_backend is not None:
            semantic_quota = max(0, int(round(self.recall_topk * self.recall_alpha)))
        semantic = self.semantic_recall(query_text, exclude_turn=query_turn, topk=semantic_quota) if semantic_quota else []
        fact_quota = self.recall_topk - len(semantic)
        facts = fetch_fact_snippets(self.conn, self.run_id, max(fact_quota, 0))
        themes = fetch_theme_rows(self.conn, self.run_id, 3)
        summaries = fetch_recent_summaries(self.conn, self.run_id, 2)
        sections: List[str] = []
        if semantic:
            lines = [f"- Turn {item['turn_index']} ({item['speaker']}): {self._truncate(item['content'])}" for item in semantic]
            sections.append("[Semantic recall]\n" + "\n".join(lines))
        if facts:
            lines = [
                f"- {fact['subject']} {fact['predicate']} {self._truncate(fact['object'])} (turn {fact['turn_id']})"
                for fact in facts
            ]
            sections.append("[Fact memory]\n" + "\n".join(lines))
        if themes:
            lines = [
                f"- {row['theme']} (strength {row['strength']:.1f}, last turn {row['last_seen_turn']})"
                for row in themes
            ]
            sections.append("[Theme tracker]\n" + "\n".join(lines))
        if summaries:
            lines = [f"- Turn {row['step']}: {self._truncate(row['content'], 320)}" for row in summaries]
            sections.append("[Recent summaries]\n" + "\n".join(lines))
        return "\n\n".join(section for section in sections if section).strip()

    @staticmethod
    def _truncate(text: str, limit: int = 220) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[:limit] + "…"


@dataclass
class NoveltyAssessment:
    score: float
    best_similarity: float
    method: str

    def passed(self, threshold: float) -> bool:
        return self.score >= threshold


@dataclass
class GenerationResult:
    content: str
    novelty: float
    similarity: float
    method: str
    attempts: int
    flagged: bool


class NoveltyScorer:
    def __init__(self, memory: MemoryManager, recent_window: int = 10) -> None:
        self.memory = memory
        self.recent_window = max(1, recent_window)

    def assess(self, candidate: str, recent_texts: Sequence[str]) -> NoveltyAssessment:
        cleaned = candidate.strip()
        if not cleaned:
            return NoveltyAssessment(score=0.0, best_similarity=1.0, method="empty")
        windowed_texts = [text for text in list(recent_texts)[-self.recent_window :] if text.strip()]
        assessments: List[NoveltyAssessment] = []
        embed_assessment = self._embedding_assessment(cleaned)
        if embed_assessment is not None:
            assessments.append(embed_assessment)
        hash_assessment = self._hash_assessment(cleaned, windowed_texts)
        if hash_assessment is not None:
            assessments.append(hash_assessment)
        if not assessments:
            return NoveltyAssessment(score=1.0, best_similarity=0.0, method="default")
        best = max(assessments, key=lambda item: item.score)
        if len(assessments) > 1:
            logging.debug(
                "[DEBUG] Novelty assessments %s -> selected %s",
                [(item.method, item.score, item.best_similarity) for item in assessments],
                (best.method, best.score, best.best_similarity),
            )
        return best

    def _embedding_assessment(self, candidate: str) -> Optional[NoveltyAssessment]:
        vector = self.memory.embed_text(candidate)
        if vector is None:
            return None
        previous = self.memory.recent_embeddings(self.recent_window)
        if not previous:
            return NoveltyAssessment(score=1.0, best_similarity=0.0, method="embedding")
        best_similarity = 0.0
        for _, other_vector in previous:
            if not other_vector:
                continue
            similarity = max(min(_cosine_similarity(vector, other_vector), 1.0), -1.0)
            best_similarity = max(best_similarity, similarity)
        score = max(0.0, 1.0 - best_similarity)
        return NoveltyAssessment(score=score, best_similarity=best_similarity, method="embedding")

    def _hash_assessment(self, candidate: str, recent_texts: Sequence[str]) -> Optional[NoveltyAssessment]:
        if not recent_texts:
            return NoveltyAssessment(score=1.0, best_similarity=0.0, method="hash")
        shingles = self._shingles(candidate)
        if not shingles:
            return NoveltyAssessment(score=0.0, best_similarity=1.0, method="hash")
        best_similarity = 0.0
        for text in recent_texts:
            other_shingles = self._shingles(text)
            if not other_shingles:
                continue
            union = len(shingles | other_shingles)
            if union == 0:
                continue
            similarity = len(shingles & other_shingles) / float(union)
            if similarity > best_similarity:
                best_similarity = similarity
        score = max(0.0, 1.0 - best_similarity)
        return NoveltyAssessment(score=score, best_similarity=best_similarity, method="hash")

    def _shingles(self, text: str) -> Set[str]:
        tokens = self._tokenize(text)
        if not tokens:
            return set()
        if len(tokens) < 3:
            return set(tokens)
        shingles: Set[str] = set()
        size = 3
        for idx in range(len(tokens) - size + 1):
            shingles.add(" ".join(tokens[idx : idx + size]))
        return shingles

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]

    @staticmethod
    def _extract_fact_snippets(content: str, limit: int = 2) -> List[str]:
        clean = content.replace("\n", " ").strip()
        if not clean:
            return []
        sentences: List[str] = []
        for segment in clean.replace("?", ".").split("."):
            candidate = segment.strip()
            if len(candidate) < 8:
                continue
            sentences.append(candidate)
            if len(sentences) >= limit:
                break
        if not sentences:
            sentences.append(clean)
        return sentences[:limit]


# ---------------------------
# Main conversational engine
# ---------------------------
class DuelEngine:
    def __init__(self,
                 backend: BackendBase,
                 db: sqlite3.Connection,
                 run_id: int,
                 speaker_a: str,
                 speaker_b: str,
                 persona_a: str,
                 persona_b: str,
                 stopwords: List[str],
                 short_ctx_turns: int,
                 max_reply_chars: int,
                 temperature: float,
                 top_p: float,
                 max_tokens: int,
                 repeat_penalty: float,
                 ngram_block_n: int,
                 summary_every: int,
                 summary_chars: int,
                 max_wallclock_seconds: Optional[float],
                 pause_jitter: Tuple[float, float],
                 growth_checks: bool,
                 novelty_threshold: float,
                 recall_topk: int,
                 recall_alpha: float,
                 embedding_model: str,
                 db_path: str,
                 topic_graph: Optional["TopicGraph"] = None,
                 topic_pivot_interval: int = 0):
        self.backend = backend
        self.db = db
        self.run_id = run_id
        self.speaker_a = speaker_a
        self.speaker_b = speaker_b
        self.persona_a = persona_a
        self.persona_b = persona_b
        self.stopwords = stopwords
        self.short_ctx_turns = short_ctx_turns
        self.max_reply_chars = max_reply_chars
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.ngram_block_n = ngram_block_n
        self.summary_every = summary_every
        self.summary_chars = summary_chars
        self.memory = MemoryManager(
            conn=db,
            run_id=run_id,
            backend=backend,
            db_path=db_path,
            embedding_model=embedding_model,
            recall_topk=recall_topk,
            recall_alpha=recall_alpha,
        )
        self.long_summary = self.memory.latest_summary()  # rolling summary we refresh periodically
        self._shutdown = False
        self.max_wallclock_seconds = max_wallclock_seconds
        self.pause_jitter = pause_jitter
        self.growth_checks = growth_checks
        self.novelty_threshold = novelty_threshold
        self.topic_graph = topic_graph
        self.topic_pivot_interval = (
            topic_pivot_interval if topic_graph and topic_pivot_interval > 0 else None
        )
        self._turns_since_pivot = 0
        self._current_theme_id: Optional[str] = None
        self._recent_theme_ids: Deque[str] = deque(maxlen=5)
        self._rng = random.Random()
        if self.topic_graph:
            initial_theme = self.topic_graph.pick_initial_theme(self._rng)
            self._current_theme_id = initial_theme.key
            self._recent_theme_ids.append(initial_theme.key)
            logging.info(
                f"[i] Topic focus initialized: {initial_theme.title}"
            )
        self._start_monotonic: Optional[float] = None
        self._max_regen_attempts = 2
        self._novelty_scorer = NoveltyScorer(self.memory, recent_window=12)

    def signal_handler(self, *_: Any) -> None:
        logging.warning("[!] Caught signal; finishing current turn and stopping.")
        self._shutdown = True

    def summarize_long_context(self, step: int) -> None:
        """Ask one of the personas to summarize the conversation so far, if backend available."""
        ctx = fetch_recent_context(self.db, self.run_id, k=200)
        if not ctx:
            return
        # Build a compact prompt to summarize themes and points of agreement/disagreement
        system = "You are a neutral summarizer. Extract main themes and disagreements. Be concise (<= 1200 chars)."
        transcript = "\n".join([f"{speaker}: {content}" for _, speaker, content in ctx])
        user = "Summarize the above conversation’s key ideas, agreements, and disagreements succinctly."
        try:
            summary = self.backend.generate(system, user + "\n\n" + transcript[-8000:], [], 0.2, 0.95, 512, 1.0)
            self.long_summary = truncate_tokens(summary, self.summary_chars)
            self.memory.record_summary(step, self.long_summary)
            logging.info("[i] Updated rolling summary.")
        except Exception as e:
            logging.warning(f"[!] Summary generation failed: {e}")

    def _active_theme(self) -> Optional["Theme"]:
        if not self.topic_graph or not self._current_theme_id:
            return None
        try:
            return self.topic_graph.theme(self._current_theme_id)
        except KeyError:
            return None

    def _topic_guidance_text(self) -> Optional[str]:
        theme = self._active_theme()
        if not theme:
            return None
        lines = [f"{theme.title}: {theme.summary}"]
        for cue in theme.guidance:
            lines.append(f"- {cue}")
        if self.topic_graph:
            previews = []
            for transition in self.topic_graph.preview_transitions(theme.key):
                if transition.target == theme.key:
                    continue
                try:
                    target_theme = self.topic_graph.theme(transition.target)
                except KeyError:
                    continue
                label = f"Pivot to {target_theme.title}"
                if transition.cue:
                    label += f": {transition.cue}"
                previews.append(f"- {label}")
            if previews:
                lines.append("Upcoming pivots:")
                lines.extend(previews)
        return "\n".join(lines)

    def _maybe_pivot_topic(self) -> None:
        if not self.topic_graph or self.topic_pivot_interval is None:
            return
        if self._turns_since_pivot < self.topic_pivot_interval:
            return
        next_theme = self.topic_graph.next_theme(
            self._current_theme_id,
            self._rng,
            tuple(self._recent_theme_ids),
        )
        if not next_theme:
            self._turns_since_pivot = 0
            return
        current = self._active_theme()
        if not current or current.key != next_theme.key:
            prior_title = current.title if current else "(none)"
            logging.info(
                f"[i] Pivoting topic from {prior_title} to {next_theme.title}."
            )
            self._current_theme_id = next_theme.key
            self._recent_theme_ids.append(next_theme.key)
        self._turns_since_pivot = 0

    def _system_for(self, speaker: str, topic_guidance: Optional[str]) -> str:
        persona = self.persona_a if speaker == self.speaker_a else self.persona_b
        safety = "Avoid long verbatim quotations. Prefer original synthesis. Do not invent private facts."
        return build_system_prompt(persona, self.long_summary, safety, topic_guidance)

    def _recent_texts(self) -> List[str]:
        ctx = fetch_recent_context(self.db, self.run_id, k=10)
        return [content for _, _, content in ctx]

    def _flagged_message(self, novelty: float) -> str:
        return (
            f"(novelty_flag: score={novelty:.3f} < threshold={self.novelty_threshold:.3f})"
        )

    def _generate_turn(self, next_speaker: str) -> GenerationResult:
        history_rows = fetch_recent_context(self.db, self.run_id, k=self.short_ctx_turns)
        topic_guidance = self._topic_guidance_text()
        system_prompt = self._system_for(next_speaker, topic_guidance)
        history_pairs = [(speaker, content) for _, speaker, content in history_rows]
        user_prompt = make_turn_prompt(history_pairs, next_speaker, self.stopwords, topic_guidance)
        recall_block = self.memory.render_recall_block(history_rows, next_speaker)
        if recall_block:
            user_prompt += "\n\n[Memory]\n" + recall_block
        logging.debug(f"[DEBUG] System prompt for {next_speaker} is {len(system_prompt)} chars.")
        logging.debug(f"[DEBUG] User prompt length={len(user_prompt)} chars.")

        max_attempts = 1 if not self.growth_checks else (self._max_regen_attempts + 1)
        attempts = 0
        final_content = ""
        candidate = ""
        assessment = NoveltyAssessment(score=1.0, best_similarity=0.0, method="default")
        flagged = False
        recent_texts = self._recent_texts()

        while attempts < max_attempts:
            attempts += 1
            raw = self.backend.generate(
                system_prompt,
                user_prompt,
                self.stopwords,
                self.temperature,
                self.top_p,
                self.max_tokens,
                self.repeat_penalty,
            )
            raw = truncate_tokens(raw, self.max_reply_chars)
            if self.ngram_block_n > 1:
                raw = ngram_block(raw, recent_texts, self.ngram_block_n)
            candidate = raw.strip()
            assessment = self._novelty_scorer.assess(candidate, recent_texts)
            logging.debug(
                "[DEBUG] Turn attempt %d by %s: novelty %.3f (method=%s, similarity=%.3f)",
                attempts,
                next_speaker,
                assessment.score,
                assessment.method,
                assessment.best_similarity,
            )
            if not self.growth_checks:
                final_content = candidate
                break
            if assessment.passed(self.novelty_threshold):
                final_content = candidate
                break
            if attempts >= max_attempts:
                flagged = True
                final_content = self._flagged_message(assessment.score)
                logging.warning(
                    "[!] Novelty %.3f below threshold %.3f after %d attempts; inserting flag.",
                    assessment.score,
                    self.novelty_threshold,
                    attempts,
                )
                break
            logging.warning(
                "[!] Novelty %.3f below threshold %.3f; regenerating (attempt %d/%d).",
                assessment.score,
                self.novelty_threshold,
                attempts,
                max_attempts,
            )

        if not final_content.strip():
            final_content = candidate.strip() or "(…)"

        return GenerationResult(
            content=final_content,
            novelty=assessment.score,
            similarity=assessment.best_similarity,
            method=assessment.method,
            attempts=attempts,
            flagged=flagged,
        )

    def run(self, start_turn_index: int, max_turns: int) -> None:
        # Alternate speakers: even turn -> A, odd -> B (or continue based on last message)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # If no prior messages, seed an initial question by A to start motion
        current_turn = start_turn_index + 1
        if start_turn_index < 0:
            seed = "Let us begin: what do you take to be the primordial tension between freedom and meaning?"
            insert_message(self.db, self.run_id, 0, self.speaker_a, seed)
            logging.info(f"[i] Seeded opening by {self.speaker_a}.")
            self.memory.record_turn(0, self.speaker_a, seed, self._current_theme_id)
            current_turn = 1

        self._start_monotonic = time.monotonic()
        while current_turn <= max_turns and not self._shutdown:
            if self.max_wallclock_seconds is not None and self._start_monotonic is not None:
                elapsed = time.monotonic() - self._start_monotonic
                if elapsed >= self.max_wallclock_seconds:
                    logging.info("[i] Reached max wall-clock duration; stopping run.")
                    break
            self._maybe_pivot_topic()
            next_speaker = self.speaker_a if current_turn % 2 == 0 else self.speaker_b
            try:
                result = self._generate_turn(next_speaker)
                reply = result.content.strip() or "(…)"
                insert_message(self.db, self.run_id, current_turn, next_speaker, reply)
                logging.info(f"[i] Turn {current_turn} by {next_speaker} stored ({len(reply)} chars).")
                self.memory.record_turn(current_turn, next_speaker, reply, self._current_theme_id)
                if result.flagged:
                    logging.warning(
                        "[!] Growth check flag inserted on turn %d (score %.3f < %.3f).",
                        current_turn,
                        result.novelty,
                        self.novelty_threshold,
                    )
                elif self.growth_checks:
                    logging.debug(
                        "[DEBUG] Novelty %.3f via %s (similarity=%.3f, attempts=%d) for turn %d.",
                        result.novelty,
                        result.method,
                        result.similarity,
                        result.attempts,
                        current_turn,
                    )
                insert_growth_metric(
                    self.db,
                    self.run_id,
                    current_turn,
                    result.novelty,
                    result.similarity,
                    result.method,
                    self.novelty_threshold if self.growth_checks else None,
                    result.attempts,
                    result.flagged,
                )
            except Exception as e:
                logging.error(f"[x] Generation failed on turn {current_turn}: {e}")
                fallback = f"(generation_error: {e})"
                insert_message(self.db, self.run_id, current_turn, next_speaker, fallback)
                self.memory.record_turn(current_turn, next_speaker, fallback, self._current_theme_id)
                insert_growth_metric(
                    self.db,
                    self.run_id,
                    current_turn,
                    0.0,
                    None,
                    "error",
                    self.novelty_threshold if self.growth_checks else None,
                    0,
                    False,
                )
            finally:
                self._turns_since_pivot += 1

            # Periodic summarization to keep a rolling long-term memory
            if self.summary_every > 0 and current_turn % self.summary_every == 0:
                self.summarize_long_context(current_turn)

            if self.pause_jitter[1] > 0:
                wait = random.uniform(self.pause_jitter[0], self.pause_jitter[1])
                if wait > 0:
                    logging.debug(f"[DEBUG] Sleeping for {wait:.2f}s between turns to simulate pacing.")
                    time.sleep(wait)

            current_turn += 1


# ---------------------------
# CLI
# ---------------------------
def parse_run_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate an endless philosophical dialogue between two personas, locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required positionals
    p.add_argument("db_path", help="Path to SQLite database file for transcripts.")
    p.add_argument("persona_a_path", help="File with Persona A instructions (e.g., Camus-like).")
    p.add_argument("persona_b_path", help="File with Persona B instructions (e.g., Nietzsche-like).")

    # Backend selection
    p.add_argument("--backend", choices=["llamacpp", "openai"], default="llamacpp",
                   help="Generation backend to use.")
    # llama.cpp
    p.add_argument("--model-path", default="", help="Path to .gguf model (llamacpp backend).")
    p.add_argument("--ctx-size", type=int, default=4096, help="Context size (llamacpp).")
    p.add_argument(
        "--gpu-layers",
        default="auto",
        help="GPU offload layers (llamacpp). Use 'auto' to probe hardware and choose for you."
    )
    p.add_argument("--seed", type=int, default=-1, help="Seed for reproducibility (llamacpp).")
    # OpenAI-compatible
    p.add_argument("--api-base", default="http://127.0.0.1:11434/v1", help="OpenAI-compatible base URL.")
    p.add_argument("--api-key", default="sk-local", help="API key (if required by local server).")
    p.add_argument("--model-name", default="local-model", help="Model name for OpenAI-compatible server.")

    # Conversation controls
    p.add_argument("--speaker-a", default="Camus", help="Display name for Persona A.")
    p.add_argument("--speaker-b", default="Nietzsche", help="Display name for Persona B.")
    p.add_argument("--max-turns", type=int, default=200, help="Run for N turns (use resume for longer).")
    p.add_argument("--short-ctx-turns", type=int, default=18, help="Recent turns to include in prompt.")
    p.add_argument("--max-reply-chars", type=int, default=1400, help="Truncate replies to this many chars.")
    p.add_argument("--stop", nargs="*", default=["Camus:", "Nietzsche:"], help="Stop sequences.")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    p.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    p.add_argument("--max-tokens", type=int, default=512, help="Max new tokens per reply (backend-specific).")
    p.add_argument("--repeat-penalty", type=float, default=1.1, help="Repeat penalty (if supported).")
    p.add_argument("--ngram-block", type=int, default=4, help="Block repeated n-grams against recent outputs (post-filter).")
    p.add_argument("--summary-every", type=int, default=50, help="Refresh rolling summary every N turns (0 to disable).")
    p.add_argument("--summary-chars", type=int, default=1200, help="Max chars to keep in rolling summary.")
    p.add_argument("--resume", action="store_true", help="Resume last run in the DB (same backend/model/personas).")
    p.add_argument("--max-wallclock", default="24h", help="Maximum wall-clock duration before stopping (e.g. 30m, 2h, off).")
    p.add_argument("--pause-jitter", default="700,4000", help="Pause between turns in milliseconds as 'min,max'.")
    p.add_argument("--growth-checks", choices=["on", "off"], default="on", help="Enforce simple novelty checks each turn.")
    p.add_argument("--novelty-threshold", type=float, default=0.35, help="Minimum novelty ratio required when growth checks are on.")
    p.add_argument("--recall-topk", type=int, default=6, help="Number of hybrid recall snippets to blend into prompts.")
    p.add_argument("--recall-alpha", type=float, default=0.6, help="Semantic recall weight (0..1) vs. fact/theme recalls.")
    p.add_argument(
        "--embedding-model",
        default="miniLM-onnx",
        help="Embedding backend spec (auto|minilm-onnx|llamacpp|off or local path).",
    )
    p.add_argument(
        "--topic-graph",
        default=DEFAULT_TOPIC_GRAPH,
        help="Path to topic graph YAML for guided pivots (use 'off' to disable).",
    )
    p.add_argument(
        "--topic-pivot-interval",
        type=int,
        default=6,
        help="Number of turns to sustain a theme before pivoting to the next guided topic.",
    )

    # Misc
    p.add_argument(
        "--zero-network",
        choices=["on", "off"],
        default="on",
        help="Block outbound HTTP requests (set to 'off' to permit HTTP backends).",
    )
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase logging verbosity (-v, -vv).")

    return p.parse_args(argv)


def parse_nuke_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="dom nuke",
        description="Delete run-specific transcripts and memory artifacts from the SQLite database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("db_path", help="Path to the SQLite database to purge.")
    p.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Limit deletion to a specific run id (omit to clear all runs).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Skip the confirmation guard and execute immediately.",
    )
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase logging verbosity (-v, -vv).")
    return p.parse_args(argv)


def parse_cli(argv: Optional[Sequence[str]] = None) -> Tuple[str, argparse.Namespace]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0].lower() == "nuke":
        return "nuke", parse_nuke_args(tokens[1:])
    if tokens and tokens[0].lower() == "run":
        tokens = tokens[1:]
    return "run", parse_run_args(tokens)


def handle_nuke(args: argparse.Namespace) -> None:
    if not args.force:
        logging.error("[x] Refusing to execute 'dom nuke' without --force confirmation.")
        sys.exit(2)

    db_path = os.path.abspath(args.db_path)
    if not os.path.exists(db_path):
        logging.error(f"[x] Database not found: {db_path}")
        sys.exit(2)

    try:
        conn = init_db(db_path)
    except Exception as e:
        logging.error(f"[x] Failed to open DB: {e}")
        sys.exit(2)

    tables = (
        ("messages", "run_id"),
        ("facts", "run_id"),
        ("themes", "run_id"),
        ("summaries", "run_id"),
        ("message_embeddings", "run_id"),
        ("growth_metrics", "run_id"),
    )

    scope = "all runs" if args.run_id is None else f"run {args.run_id}"
    deletions: Dict[str, int] = {}

    try:
        with conn:
            for table, column in tables:
                if args.run_id is None:
                    cur = conn.execute(f"DELETE FROM {table}")
                else:
                    cur = conn.execute(
                        f"DELETE FROM {table} WHERE {column}=?",
                        (args.run_id,),
                    )
                deleted = cur.rowcount if cur.rowcount is not None else 0
                if deleted < 0:
                    deleted = 0
                deletions[table] = deleted
        conn.execute("REINDEX;")
        conn.execute("PRAGMA optimize;")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    finally:
        conn.close()

    total_deleted = sum(deletions.values())
    logging.info(f"[i] Removed {total_deleted} row(s) across {len(deletions)} tables for {scope}.")
    for table, count in deletions.items():
        logging.debug(f"[DEBUG] {table}: deleted {count} row(s).")


def main(argv: Optional[Sequence[str]] = None) -> None:
    command, args = parse_cli(argv)
    setup_logging(getattr(args, "verbose", 1))

    if command == "nuke":
        handle_nuke(args)
        return

    try:
        max_wallclock_seconds = parse_wallclock_arg(args.max_wallclock)
    except ValueError as e:
        logging.error(f"[x] {e}")
        sys.exit(2)

    try:
        pause_jitter = parse_pause_jitter_arg(args.pause_jitter)
    except ValueError as e:
        logging.error(f"[x] Invalid --pause-jitter value: {e}")
        sys.exit(2)

    if not 0.0 <= args.novelty_threshold <= 1.0:
        logging.error("[x] --novelty-threshold must be between 0.0 and 1.0.")
        sys.exit(2)
    if args.recall_topk < 0:
        logging.error("[x] --recall-topk must be >= 0.")
        sys.exit(2)
    if not 0.0 <= args.recall_alpha <= 1.0:
        logging.error("[x] --recall-alpha must be between 0.0 and 1.0.")
        sys.exit(2)

    zero_network_enabled = args.zero_network == "on"
    if zero_network_enabled:
        enable_zero_network_guard()

    growth_checks_enabled = args.growth_checks == "on"

    # Basic validations
    if args.backend == "llamacpp" and not args.model_path:
        logging.error("[x] --model-path is required for llamacpp backend.")
        sys.exit(2)
    if args.backend == "llamacpp" and not _HAS_LLAMA:
        logging.error("[x] llama-cpp-python not installed. pip install llama-cpp-python")
        sys.exit(2)
    if args.backend == "openai" and not _HAS_REQUESTS:
        logging.error("[x] requests not installed. pip install requests")
        sys.exit(2)
    if zero_network_enabled and args.backend != "llamacpp":
        logging.error(
            "[x] Zero-network mode blocks HTTP backends. Use --zero-network off to run the OpenAI-compatible backend."
        )
        sys.exit(2)

    # Load personas
    try:
        persona_a = read_file(args.persona_a_path)
        persona_b = read_file(args.persona_b_path)
    except Exception as e:
        logging.error(f"[x] Failed to read persona files: {e}")
        sys.exit(2)

    topic_graph_path = (args.topic_graph or "").strip()
    topic_graph: Optional["TopicGraph"] = None
    topic_pivot_interval = args.topic_pivot_interval
    if topic_graph_path.lower() in {"", "off", "none"}:
        topic_graph_path = ""
        topic_pivot_interval = 0
    else:
        if not os.path.isabs(topic_graph_path):
            topic_graph_path = os.path.abspath(topic_graph_path)
        from topics import load_topic_graph

        try:
            topic_graph = load_topic_graph(topic_graph_path)
        except ModuleNotFoundError:
            logging.error("[x] Loading topic graphs requires PyYAML. pip install pyyaml")
            sys.exit(2)
        except (FileNotFoundError, ValueError) as e:
            logging.error(f"[x] Failed to load topic graph: {e}")
            sys.exit(2)
        if topic_pivot_interval <= 0:
            logging.error("[x] --topic-pivot-interval must be > 0 when --topic-graph is set.")
            sys.exit(2)
        logging.info(
            f"[i] Loaded topic graph '{topic_graph.name}' with {len(topic_graph.themes)} themes."
        )

    # Backend init
    if args.backend == "llamacpp":
        gpu_layers_arg = str(args.gpu_layers).strip()
        gpu_layers_mode = "auto" if gpu_layers_arg.lower() == "auto" else "manual"
        if gpu_layers_mode == "auto":
            try:
                from runtime.gpu_probe import auto_select_gpu_layers

                gpu_layers_value = auto_select_gpu_layers(ctx_size=args.ctx_size)
                logging.info(f"[i] Auto-selected {gpu_layers_value} GPU layers via hardware probe.")
            except Exception as e:
                logging.warning(f"[!] GPU auto-selection failed ({e}); falling back to 35 layers.")
                gpu_layers_value = 35
        else:
            try:
                gpu_layers_value = int(gpu_layers_arg)
            except ValueError:
                logging.error("[x] --gpu-layers must be an integer or 'auto'.")
                sys.exit(2)
        if gpu_layers_value < 0:
            logging.error("[x] --gpu-layers must be >= 0.")
            sys.exit(2)
        try:
            backend = LlamaCppBackend(
                model_path=args.model_path,
                ctx_size=args.ctx_size,
                gpu_layers=gpu_layers_value,
                seed=args.seed if args.seed >= 0 else None
            )
        except Exception as e:
            logging.error(f"[x] Failed to initialize llama.cpp backend: {e}")
            sys.exit(2)
        model_id = os.path.basename(args.model_path)
    else:
        try:
            backend = OpenAICompatBackend(api_base=args.api_base, api_key=args.api_key, model_name=args.model_name)
        except Exception as e:
            logging.error(f"[x] Failed to initialize OpenAI-compatible backend: {e}")
            sys.exit(2)
        model_id = args.model_name
        gpu_layers_mode = "n/a"
        gpu_layers_value = -1

    # DB
    try:
        conn = init_db(args.db_path)
    except Exception as e:
        logging.error(f"[x] Failed to open DB: {e}")
        sys.exit(2)

    params = {
        "speaker_a": args.speaker_a,
        "speaker_b": args.speaker_b,
        "short_ctx_turns": args.short_ctx_turns,
        "max_reply_chars": args.max_reply_chars,
        "stop": args.stop,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "repeat_penalty": args.repeat_penalty,
        "ngram_block": args.ngram_block,
        "summary_every": args.summary_every,
        "summary_chars": args.summary_chars,
        "max_wallclock": args.max_wallclock,
        "max_wallclock_seconds": max_wallclock_seconds,
        "pause_jitter": args.pause_jitter,
        "pause_jitter_seconds": list(pause_jitter),
        "growth_checks": args.growth_checks,
        "growth_checks_enabled": growth_checks_enabled,
        "novelty_threshold": args.novelty_threshold,
        "recall_topk": args.recall_topk,
        "recall_alpha": args.recall_alpha,
        "embedding_model": args.embedding_model,
        "gpu_layers_mode": gpu_layers_mode,
        "gpu_layers": gpu_layers_value,
        "topic_graph_path": topic_graph_path,
        "topic_pivot_interval": topic_pivot_interval,
        "topic_guidance_enabled": bool(topic_graph),
        "zero_network": args.zero_network,
        "zero_network_enabled": zero_network_enabled,
    }

    run_id: Optional[int] = None
    start_turn = -1

    if args.resume:
        # Resume the most recent run that matches backend+model+personas
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM runs WHERE backend=? AND model_name=? AND persona_a_path=? AND persona_b_path=? "
            "ORDER BY id DESC LIMIT 1",
            (backend.name(), model_id, os.path.abspath(args.persona_a_path), os.path.abspath(args.persona_b_path))
        )
        row = cur.fetchone()
        if row:
            run_id = int(row[0])
            start_turn = fetch_last_turn(conn, run_id)
            logging.info(f"[i] Resuming run_id={run_id} from turn {start_turn+1}.")
        else:
            logging.warning("[!] No matching prior run found to resume; starting a new run.")

    if run_id is None:
        try:
            run_id = insert_run(
                conn,
                backend=backend.name(),
                model_name=model_id,
                persona_a_path=os.path.abspath(args.persona_a_path),
                persona_b_path=os.path.abspath(args.persona_b_path),
                params_json=json.dumps(params, sort_keys=True),
            )
        except Exception as e:
            logging.error(f"[x] Failed to create run record: {e}")
            sys.exit(2)
        start_turn = -1

    # Launch engine
    engine = DuelEngine(
        backend=backend,
        db=conn,
        run_id=run_id,
        speaker_a=args.speaker_a,
        speaker_b=args.speaker_b,
        persona_a=persona_a,
        persona_b=persona_b,
        stopwords=args.stop,
        short_ctx_turns=args.short_ctx_turns,
        max_reply_chars=args.max_reply_chars,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repeat_penalty=args.repeat_penalty,
        ngram_block_n=args.ngram_block,
        summary_every=args.summary_every,
        summary_chars=args.summary_chars,
        max_wallclock_seconds=max_wallclock_seconds,
        pause_jitter=pause_jitter,
        growth_checks=growth_checks_enabled,
        novelty_threshold=args.novelty_threshold,
        recall_topk=args.recall_topk,
        recall_alpha=args.recall_alpha,
        embedding_model=args.embedding_model,
        db_path=os.path.abspath(args.db_path),
        topic_graph=topic_graph,
        topic_pivot_interval=topic_pivot_interval,
    )

    try:
        engine.run(start_turn_index=start_turn, max_turns=args.max_turns)
        logging.info("[i] Dialogue complete.")
    except KeyboardInterrupt:
        logging.warning("[!] Interrupted by user; exiting.")
    except Exception as e:
        logging.error(f"[x] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
