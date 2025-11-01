# AGENTS.md — Roadmap for **Duel of Minds**

## 0) Purpose & North Star
Create a **local-only**, endlessly extensible dialogue engine that simulates two evolving philosophical personas picking up where a philosophy “left off,” generating **novel, non-repeating** discourse that grows ideas over time. Runs on **Windows 11** with **AMD AI Max 395+ (128 GB RAM, 2 TB SSD, Radeon 8060+)** and never touches the cloud.

---

## 1) Non-Negotiable Constraints
- **Local only:** no outbound network calls; models & embeddings load from disk.
- **No model weights in repo:** user config points to local files.
- **Windows-first:** PowerShell scripts & binaries tested on Win11.
- **Sane defaults:** hard wall-clock cap **≤ 24h** unless overridden.
- **Copyright & safety:** public-domain sources or lawful style prompts; no long verbatim quotes.

---

## 2) Runtime & Backends (Windows + AMD)
- **Primary:** `llama.cpp` **Vulkan** build (best AMD path on Windows).
	- **Formats:** **GGUF** (Q4_K_M, Q5_K_M, Q6_K, Q8_0, etc.).
- **Alternative:** `llama.cpp` CPU-only (compatibility mode).
- **Optional:** local OpenAI-compatible servers (LM Studio, Ollama on Windows w/ AMD support as it matures) behind a flag; default is direct `llama.cpp`.

**Rationale:** Vulkan is the most reliable cross-vendor GPU path on Win; ROCm is Linux-centric; DirectML is improving but less predictable for two-agent loads.

---

## 3) Model Handling (DeepSeek & Friends)
- **Recommended bases (examples):**
	- **DeepSeek-R1 / DeepSeek-L** (when legally available as local **GGUF**).
	- **Llama-3.x Instruct** GGUF (widely quantized; strong instruction following).
	- **Mistral-Instruct / Phi-4** GGUF (compact, fast for summarizers/helpers).
- **Allowed formats:** **GGUF** (primary). Document GPTQ/AWQ/EXL2 as “advanced/alt runtime” only—**not default**.
- **Acquisition:** download from reputable hubs; verify checksums; store under `models/` **outside** the repo root.
- **Configuration:** paths in `config/models.toml` (templated), with `sha256` fields for verification.
- **Repository policy:** `.gitignore` covers `models/**`, `*.gguf`, `*.bin`, `*.safetensors`.

---

## 4) Personas & “Pick-Up-Where-They-Left-Off”
- **Persona files** (plain text) per agent:
	- Voice & taboos (no psycho-biography, no long quotes).
	- **Continuation seed:** “Continue the unresolved questions of <Philosophy X> post-<Date/Event>; extend arguments, resolve tensions, propose testable theses.”
	- **Growth objective:** “Each turn must add at least one **new claim**, critique, or refinement.”
- **Topic graph seeds** (YAML): unresolved themes, paradoxes, canonical disputes, frontier questions. Agents pivot across this graph to avoid loops.

---

## 5) Conversation Engine (Non-Repeating by Design)
- **Turn-taking with jittered pauses:** random think time (e.g., 0.7–4.0 s) + token-rate throttling to mimic humans and reduce iGPU spikes.
- **Anti-loop controls:** n-gram blocking (pre/post), repetition penalty, semantic-novelty reward (distance vs. last N turns), hash-based duplicate filter.
- **Growth checks:** per-turn validator enforces “new information or relation” (semantic diff > threshold or at least one new proposition extracted).
- **Topic pivots:** every _M_ turns, transition via topic graph with soft constraints (coverage, freshness, difficulty).

---

## 6) Memory Architecture (Local, SQLite-centric)
Extend the existing transcript DB to structured memory:

- **Tables**
	- `facts(subject, predicate, object, turn_id, confidence)`
	- `themes(theme, strength, last_seen_turn)`
	- `summaries(step, content, created_at_utc)`
- **Vector memory (local embeddings):**
	- Embed with a small **local** model (e.g., `all-MiniLM-L6-v2` via ONNX, or llama.cpp-based embeddings) — **no network**.
	- Store vectors using **sqlite-vss** if available on Windows; else FAISS with on-disk index mirrored.
- **Recall policy:** hybrid (recent buffer + long-term summary + top-K semantic recalls + key facts). Keep context within model limits.
- **Forgetting:** periodic distillation of history into facts/themes; prune near-duplicates.

---

## 7) Scheduling, Limits & CLI (Sane Defaults)
- **Wall-clock cap:** default `--max-wallclock 24h` (graceful stop + resumable).
- **Turn caps:** default `--max-turns 2000` (override as needed).
- **Pacing:** `--pause-jitter "<min_ms>,<max_ms>"`, `--avg-tokens-per-sec` throttle.
- **Resource guards:** `--ctx-size`, `--gpu-layers`, optional `--cpu-threads`.
- **Resume semantics:** match backend+model+personas or explicit `--run-id`.

---

## 8) Evaluation & Quality Gates (Offline)
- **Leakage tests:** block long quotes; redacted probes must elicit refusal/summary.
- **Novelty metrics:** distinct-n, self-BLEU, semantic diversity (avg pairwise cosine).
- **Progress signal:** count **new facts/theses** per 100 turns; density of cross-references; resolution rate of previously flagged tensions.
- **Human review:** periodic TCR snapshots (Thesis–Critique–Revision) exported to Markdown.

---

## 9) Performance Guidance (AMD AI Max 395+)
- Prefer **Q5/Q6** GGUF for quality if latency is acceptable; otherwise **Q4_K_M** for speed.
- Use **Vulkan offload**; begin with partial GPU layers (e.g., 20–35) and profile token/s.
- **Serialize** the two agents’ generations (no simultaneous decoding) + **jittered pauses** to avoid iGPU contention.
- Enable SQLite **WAL**, set `synchronous=NORMAL`, batch inserts every few turns.

---

## 10) Security & Privacy
- **Zero-network mode** default ON; fail fast if any HTTP is attempted.
- Verify model checksums before load.
- Prompt-level PII guardrails; regex/heuristic blocklist for accidental doxxing.
- **Nuke command:** delete run-scoped data (messages/facts/themes/summaries) + rebuild vector indices on request.

---

## 11) Setup UX (Windows 11)
- **Scripts:** PowerShell helpers to fetch `llama.cpp` Vulkan build into `./bin/`, create `models/` (outside repo), write `config/models.toml`.
- **Model guidance (docs):** how to select GGUF variants, verify SHA256, place paths, and pick quantization levels for your hardware.
- **Health checks:** `dom doctor` validates binaries, model paths, GPU capability, then runs a 10-token smoke test.

**Example (indented with tabs in Markdown for code):**
	# Install llama.cpp (Vulkan) and create venv
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	pip install --upgrade pip
	# (If using Python wrapper; otherwise use bundled exe)
	# Place model outside repo: D:\ai\models\deepseek\deepseek.gguf
	# Run health check
	python dom.py doctor --model-path "D:\ai\models\deepseek\deepseek.gguf"

---

## 12) Roadmap (Phased)

### Phase 0 — Stabilize Runner (near-term)
- Add CLI: `--max-wallclock`, `--pause-jitter`, `--growth-checks on|off`, `--novelty-threshold`.
- Add Vulkan build path & **GPU-layer auto-tuner** (probe iGPU, choose layers).
- Load **topic graph** YAML + periodic pivoting.
- Ship model acquisition docs (GGUF only) + stricter `.gitignore`.

### Phase 1 — Memory & Growth (short → mid)
- Implement `facts/themes/summaries` + periodic distillation.
- Integrate **local** embeddings + SQLite-VSS (or FAISS fallback).
- Hybrid recall policy with `--recall-topk` and `--recall-alpha`.
- Enforce growth gate; regen on failed novelty checks.

### Phase 2 — Evaluation & Tooling (mid)
- Offline **leakage** & **novelty** suites (`dom eval`).
- Export TCR snapshots to `outputs/`.
- TUI transcript browser with FTS5 search.

### Phase 3 — Authoring Experience (mid → long)
- Persona authoring assistant (style coaching from lawful seeds).
- Interactive topic-weaver (frontier question curation & coverage).
- Optional multi-agent expansions (guest philosophers, debate/consensus).

---

## 13) CLI Additions (Spec)
- `--max-wallclock <dur>` (e.g., `30m|2h|24h|off`) **default `24h`**
- `--pause-jitter "<min_ms>,<max_ms>"` **default `700,4000`**
- `--growth-checks {on,off}` **default `on`**
- `--novelty-threshold <0..1>` **default `0.35`**
- `--recall-topk <int>` **default `6`**
- `--recall-alpha <0..1>` **default `0.6`**
- `--gpu-layers auto|<int>` **default `auto`**
- `--ctx-size <int>` **default `4096`**
- `--embedding-model <name|path>` **default `miniLM-onnx` (doc exact options)**
- `--zero-network {on,off}` **default `on`**

---

## 14) Testing Matrix
- **Backends:** llama.cpp CPU vs Vulkan on AMD.
- **Models:** DeepSeek-L (GGUF when available), Llama-3.x Instruct (Q4–Q6).
- **Loads:** 2h, 8h, 24h runs; record token/s, novelty, memory growth, GPU temps.
- **Failure modes:** model not found, checksum mismatch, OOM, vector index corruption, long-quote leak → fail closed with actionable logs.

---

## 15) Release Artifacts
- Updated **README** (Win11 quickstart + model guidance).
- **AGENTS.md** (this file) versioned with change log.
- `bin/` (llama.cpp Vulkan builds), `scripts/` (PowerShell helpers), `config/` templates.
- Sample **persona** files + **topic graph** seeds (lawful & short).
- Automated **health check** + **eval** reports in `outputs/`.

---

## 16) Open Questions
- Best **local** embeddings on Windows AMD (ONNX vs llama.cpp embeddings) without CUDA.
- Vulkan offload stability across driver updates.
- Heuristics vs learned scorer for growth checks (start simple; measure).

---

### Success Criteria
1) **Local-only** install on Win11 in ≤ 10 minutes (excluding model downloads).  
2) **24-hour run** completes within caps, with measurable **novelty** and **new theses** density.  
3) Clean config/code separation, zero model uploads, easy resume, safe defaults.

