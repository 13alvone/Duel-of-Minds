# Duel of Minds (Windows 11 Vulkan Edition)

Local-only dual persona dialogue engine optimized for Windows 11 + AMD Radeon platforms. Runs entirely offline using `llama.cpp` Vulkan builds or a bundled CLI fallback, with SQLite persistence and extensible personas.

---

## TL;DR Quickstart (Windows 11)
1. **Clone** the repo somewhere separate from your model storage (e.g., `D:\workspace\Duel-of-Minds`).
2. **Install Vulkan binaries**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ./scripts/Install-LlamaCppVulkan.ps1 -VersionTag b3532
   ```
   This populates `./bin` with `llama.cpp.exe` + DLLs and scaffolds `config/models.toml`.
3. **Register your models** (stored outside the repo, e.g., `D:\ai\models`):
   ```powershell
   ./scripts/New-ModelEntry.ps1 -Alias deepseek-q5 -ModelPath "D:/ai/models/deepseek/deepseek-q5.gguf"
   ```
4. **(Optional) Create a venv** and install the extras you need:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r runtime\requirements-base.txt  # edit as desired
   ```
5. **Run a smoke test**:
   ```powershell
   python .\duel_of_minds.py data\conversations.db data\personas\camus.txt data\personas\nietzsche.txt `
       --backend llamacpp `
       --model-path "D:/ai/models/deepseek/deepseek-q5.gguf" `
       --max-turns 2 `
       --max-tokens 128 `
       --gpu-layers auto `
       --zero-network on `
       -v
   ```
   If `llama-cpp-python` is installed, it will be used. Otherwise the runner spawns `./bin/llama.cpp.exe` directly via Vulkan.

Full walkthroughs live in [`docs/windows-quickstart.md`](docs/windows-quickstart.md).

---

## Model Placement & SHA256 Verification
- Keep GGUF weights on a separate drive (`D:\ai\models`).
- Declare each file in `config/models.toml` with alias, absolute path, and SHA256 hash.
- Use [`scripts/New-ModelEntry.ps1`](scripts/New-ModelEntry.ps1) to compute and append verified entries.
- Manual checksum recipes (PowerShell + Python) and storage layout tips are documented in [`docs/model-prep.md`](docs/model-prep.md).

The runner refuses to load any model not listed in the config file, preventing accidental use of tampered weights.

---

## AMD Vulkan Tuning Highlights
- Start with `--gpu-layers auto` to probe the Radeon iGPU/dGPU.
- Recommended quantizations: Q5_K_M for premium quality, Q4_K_M for extended runs.
- Use `--pause-jitter "700,4000"` and serialize turns to avoid iGPU contention.
- Additional tables, troubleshooting tips, and recovery steps are available in [`docs/amd-vulkan-tuning.md`](docs/amd-vulkan-tuning.md).

---

## CLI Essentials
Run `python .\duel_of_minds.py --help` for the full argument list. Key switches:

| Category | Flags |
|----------|-------|
| Backend  | `--backend {llamacpp|openai}`, `--model-path`, `--gpu-layers`, `--api-base`, `--model-name` |
| Dialogue | `--speaker-a`, `--speaker-b`, `--max-turns`, `--resume`, `--pause-jitter`, `--max-tokens` |
| Safety   | `--zero-network {on|off}`, `--stop`, `--ngram-block`, `--repeat-penalty` |
| Memory   | `--summary-every`, `--summary-chars`, `--ctx-size` |
| Logging  | `-v`, `-vv`, `--log-file` |

Personas live under `data/personas/`; topic graphs and seeds are in `data/topics/`.

---

## Offline & Security Guarantees
- Default zero-network guard blocks outbound HTTP; disable with `--zero-network off` only when targeting trusted local APIs.
- SQLite transcripts live in `data/` by default; rotate via the positional `db_path` argument.
- Use `scripts/Install-LlamaCppVulkan.ps1 -Force` to refresh binaries whenever AMD driver updates land.

---

## Need More?
- [Windows 11 Quickstart](docs/windows-quickstart.md)
- [Model Placement & Integrity](docs/model-prep.md)
- [AMD Vulkan Tuning Cheatsheet](docs/amd-vulkan-tuning.md)
- [AGENTS.md](AGENTS.md) for the long-term roadmap and design constraints.

Contributions should preserve the local-only, Windows-first philosophy and avoid bundling proprietary weights. Stay philosophical, stay offline.
