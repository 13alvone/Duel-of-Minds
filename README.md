# Duel of Minds — Local Two-Persona Dialogue Runner

Simulate an endless (or bounded) conversation between two philosophical personas (e.g., “Camus-like” and Nietzsche), entirely **offline** and **local**. Uses either `llama.cpp` via `llama-cpp-python` or any **OpenAI-compatible** local HTTP server. All turns are stored in **SQLite** with resumable runs, rolling summaries, and basic anti-loop controls.

---

## Features
- Local-only: `llama.cpp` or OpenAI-compatible local HTTP API.
- Robust CLI via `argparse`; resumable runs; safe interrupts.
- Transcript persistence in SQLite (`runs`, `messages` tables).
- Short-term context + rolling long-term summary.
- N-gram de-looping, repetition penalty, stop sequences.
- External persona files; kill-switch; logging.

---

## Requirements
- Python 3.9+
- One of:
	- `llama-cpp-python` and a `.gguf` model, **or**
	- A local OpenAI-compatible server (e.g., Ollama w/ OpenAI port, LM Studio, text-gen-webui API).
- Optional: `requests` (for OpenAI-compatible backend).

---

## Install
	python3 -m venv .venv && source .venv/bin/activate
	pip install --upgrade pip
	# Choose one:
	pip install llama-cpp-python
	# or, for OpenAI-compatible:
	pip install requests

---

## Persona Files
Prepare two small text files describing each persona’s voice and guardrails (do NOT paste long copyrighted passages).

	# persona_camus.txt (example snippet)
	Voice: lucid, humane, measured; explores absurdity and revolt without nihilism; favors clarity and concrete images; resists grand systems.

	# persona_nietzsche.txt (example snippet)
	Voice: incisive, aphoristic; genealogy of morals; self-overcoming; paradox that clarifies; metaphor as hammer; distrust herd morality.

---

## Quick Start (llama.cpp backend)
	./duel_of_minds.py conversations.db persona_camus.txt persona_nietzsche.txt \
		--backend llamacpp \
		--model-path /path/to/model.gguf \
		--speaker-a Camus --speaker-b Nietzsche \
		--max-turns 200 --short-ctx-turns 18 \
		--temperature 0.8 --top-p 0.95 --repeat-penalty 1.1 \
		--max-tokens 512 --max-reply-chars 1400 \
		--ngram-block 4 --stop Camus: Nietzsche: \
		--summary-every 50 --summary-chars 1200 -vv

**Notes**
- Press **Ctrl-C** anytime; the current turn finishes and the run stops cleanly.
- Re-run with `--resume` to continue the latest matching run.

---

## Quick Start (OpenAI-compatible backend)
Run your local server, then:

	./duel_of_minds.py conversations.db persona_camus.txt persona_nietzsche.txt \
		--backend openai \
		--api-base http://127.0.0.1:11434/v1 \
		--api-key sk-local \
		--model-name my-local-model \
		--max-turns 200 --resume -v

---

## Resuming & Inspecting
- **Resume:** add `--resume` to continue the last run with the same backend/model/personas.
- **Inspect transcripts:** open `conversations.db`:

	sqlite3 conversations.db
	.headers on
	.mode column
	SELECT id, started_at_utc, backend, model_name FROM runs ORDER BY id DESC LIMIT 5;
	SELECT turn_index, speaker, substr(content,1,120) AS snippet FROM messages WHERE run_id=<ID> ORDER BY turn_index;

---

## Key CLI Options (essentials)
- Backends:
	- `--backend {llamacpp|openai}`
	- Llama.cpp: `--model-path`, `--ctx-size`, `--gpu-layers`, `--seed`
	- OpenAI-compat: `--api-base`, `--api-key`, `--model-name`
- Dialogue:
	- `--speaker-a`, `--speaker-b`, `--max-turns`, `--resume`
	- Context & style: `--short-ctx-turns`, `--temperature`, `--top-p`, `--repeat-penalty`, `--max-tokens`, `--max-reply-chars`
	- De-loop & stops: `--ngram-block`, `--stop ...`
	- Summarization: `--summary-every`, `--summary-chars`
- Logging: `-v` (info), `-vv` (debug)

---

## Good Practices
- **Ethics & legality:** Nietzsche’s corpus is public domain; for living or copyright-protected authors, model *style* without copying long texts. Prefer paraphrase and public-domain sources.
- **Persona design:** Keep files concise and directive (tone, objectives, taboos). Avoid “quote dumps”.
- **Stability:** If looping or blandness appears, tune `--temperature`, `--top-p`, `--repeat-penalty`, and `--ngram-block`. Increase `--short-ctx-turns` sparingly to fit context.
- **Safety rails:** Use `--stop` tokens (`Camus:`, `Nietzsche:`) and keep persona files from inviting biographical fantasies or private claims.

---

## Troubleshooting
- **Empty or repetitive replies:** raise `--temperature` slightly, tune `--repeat-penalty` (e.g., 1.1→1.2), set `--ngram-block 4–6`.
- **Context truncation:** increase `--ctx-size` (llama.cpp) and/or lower `--short-ctx-turns` / `--max-reply-chars`.
- **HTTP errors (openai backend):** verify `--api-base` URL, `--model-name`, and server is running; check logs for 401/404/500.
- **Slow startup (llama.cpp):** first load compiles kernels/maps; subsequent runs are faster. Adjust `--gpu-layers`.

---

## Why SQLite?
- Durable transcripts, resumable runs, easy querying, compact storage, and zero external dependencies.

---

## License & Attribution
This component orchestrates local generation and stores transcripts. Ensure your **model** and **persona sources** comply with their licenses. You own your transcripts; handle them responsibly.

---
