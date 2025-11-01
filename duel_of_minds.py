#!/usr/bin/env python3
import argparse
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

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
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at_utc TEXT NOT NULL,
    backend TEXT NOT NULL,
    model_name TEXT NOT NULL,
    persona_a_path TEXT NOT NULL,
    persona_b_path TEXT NOT NULL,
    params_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    turn_index INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_run_turn ON messages(run_id, turn_index);
"""

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(SCHEMA)
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

def fetch_recent_context(conn: sqlite3.Connection, run_id: int, k: int) -> List[Tuple[str, str]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT speaker, content FROM messages WHERE run_id=? ORDER BY turn_index DESC LIMIT ?",
        (run_id, k),
    )
    rows = cur.fetchall()
    rows.reverse()
    return rows


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


class LlamaCppBackend(BackendBase):
    def __init__(self, model_path: str, ctx_size: int, gpu_layers: int, seed: Optional[int]):
        if not _HAS_LLAMA:
            logging.error("[x] llama-cpp-python not available. Install it or choose the openai backend.")
            raise RuntimeError("llama-cpp-python missing.")
        if not os.path.exists(model_path):
            logging.error(f"[x] Model path does not exist: {model_path}")
            raise FileNotFoundError(model_path)
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


class OpenAICompatBackend(BackendBase):
    def __init__(self, api_base: str, api_key: str, model_name: str, timeout: int = 120):
        if not _HAS_REQUESTS:
            logging.error("[x] requests not available. Install it or choose the llamacpp backend.")
            raise RuntimeError("requests missing.")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model_name
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str, stop: List[str],
                 temperature: float, top_p: float, max_tokens: int,
                 repeat_penalty: float) -> str:
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


# ---------------------------
# Utilities
# ---------------------------
def read_file(path: str) -> str:
    if not os.path.exists(path):
        logging.error(f"[x] File not found: {path}")
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_system_prompt(persona_text: str, long_summary: str, safety_notes: str) -> str:
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
    return "\n".join(lines)

def make_turn_prompt(history: List[Tuple[str, str]], next_speaker: str, stopwords: List[str]) -> str:
    # Render short buffer
    rendered = []
    for speaker, content in history:
        rendered.append(f"{speaker}: {content}")
    rendered.append(f"{next_speaker}:")
    prompt = "\n".join(rendered)
    # Ensure stopwords don’t accidentally appear mid-turn by hinting format
    if stopwords:
        prompt += "\n\n(Respond as the next line. Avoid the explicit tokens: " + ", ".join(stopwords) + ")"
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
                 summary_chars: int):
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
        self.long_summary = ""  # rolling summary we refresh periodically
        self._shutdown = False

    def signal_handler(self, *_: Any) -> None:
        logging.warning("[!] Caught signal; finishing current turn and stopping.")
        self._shutdown = True

    def summarize_long_context(self) -> None:
        """Ask one of the personas to summarize the conversation so far, if backend available."""
        ctx = fetch_recent_context(self.db, self.run_id, k=200)
        if not ctx:
            return
        # Build a compact prompt to summarize themes and points of agreement/disagreement
        system = "You are a neutral summarizer. Extract main themes and disagreements. Be concise (<= 1200 chars)."
        transcript = "\n".join([f"{s}: {c}" for s, c in ctx])
        user = "Summarize the above conversation’s key ideas, agreements, and disagreements succinctly."
        try:
            summary = self.backend.generate(system, user + "\n\n" + transcript[-8000:], [], 0.2, 0.95, 512, 1.0)
            self.long_summary = truncate_tokens(summary, self.summary_chars)
            logging.info("[i] Updated rolling summary.")
        except Exception as e:
            logging.warning(f"[!] Summary generation failed: {e}")

    def _system_for(self, speaker: str) -> str:
        persona = self.persona_a if speaker == self.speaker_a else self.persona_b
        safety = "Avoid long verbatim quotations. Prefer original synthesis. Do not invent private facts."
        return build_system_prompt(persona, self.long_summary, safety)

    def _recent_texts(self) -> List[str]:
        ctx = fetch_recent_context(self.db, self.run_id, k=10)
        return [c for _, c in ctx]

    def _generate_turn(self, next_speaker: str) -> str:
        history = fetch_recent_context(self.db, self.run_id, k=self.short_ctx_turns)
        system_prompt = self._system_for(next_speaker)
        user_prompt = make_turn_prompt(history, next_speaker, self.stopwords)
        logging.debug(f"[DEBUG] System prompt for {next_speaker} is {len(system_prompt)} chars.")
        logging.debug(f"[DEBUG] User prompt length={len(user_prompt)} chars.")
        raw = self.backend.generate(
            system_prompt, user_prompt, self.stopwords,
            self.temperature, self.top_p, self.max_tokens, self.repeat_penalty
        )
        raw = truncate_tokens(raw, self.max_reply_chars)
        if self.ngram_block_n > 1:
            raw = ngram_block(raw, self._recent_texts(), self.ngram_block_n)
        return raw.strip()

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
            current_turn = 1

        while current_turn <= max_turns and not self._shutdown:
            next_speaker = self.speaker_a if current_turn % 2 == 0 else self.speaker_b
            try:
                reply = self._generate_turn(next_speaker)
                if not reply:
                    logging.warning("[!] Empty reply received; inserting '(…)' and continuing.")
                    reply = "(…)"
                insert_message(self.db, self.run_id, current_turn, next_speaker, reply)
                logging.info(f"[i] Turn {current_turn} by {next_speaker} stored ({len(reply)} chars).")
            except Exception as e:
                logging.error(f"[x] Generation failed on turn {current_turn}: {e}")
                # Insert an explicit defect marker and continue
                insert_message(self.db, self.run_id, current_turn, next_speaker, f"(generation_error: {e})")

            # Periodic summarization to keep a rolling long-term memory
            if self.summary_every > 0 and current_turn % self.summary_every == 0:
                self.summarize_long_context()

            current_turn += 1


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
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
    p.add_argument("--gpu-layers", type=int, default=35, help="GPU offload layers (llamacpp).")
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

    # Misc
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase logging verbosity (-v, -vv).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

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

    # Load personas
    try:
        persona_a = read_file(args.persona_a_path)
        persona_b = read_file(args.persona_b_path)
    except Exception as e:
        logging.error(f"[x] Failed to read persona files: {e}")
        sys.exit(2)

    # Backend init
    if args.backend == "llamacpp":
        backend = LlamaCppBackend(
            model_path=args.model_path,
            ctx_size=args.ctx_size,
            gpu_layers=args.gpu_layers,
            seed=args.seed if args.seed >= 0 else None
        )
        model_id = os.path.basename(args.model_path)
    else:
        backend = OpenAICompatBackend(api_base=args.api_base, api_key=args.api_key, model_name=args.model_name)
        model_id = args.model_name

    # DB
    try:
        conn = init_db(args.db_path)
    except Exception as e:
        logging.error(f"[x] Failed to open DB: {e}")
        sys.exit(2)

    # Resume or new run
    import json
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
