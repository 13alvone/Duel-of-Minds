# Duel of Minds (Windows 11 Vulkan Edition)

Local-only dual persona dialogue engine optimized for Windows 11 + AMD Radeon platforms. Runs entirely offline using llama.cpp Vulkan builds or a bundled CLI fallback, with SQLite persistence and extensible personas.

---

## TL;DR Quickstart (Windows 11)
1. **Clone** the repo somewhere separate from your model storage (e.g., `D:\workspace\Duel-of-Minds`).

2. **Install Vulkan binaries**:
		Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
		./scripts/Install-LlamaCppVulkan.ps1 -VersionTag b3532
	This populates `./bin` with `llama.cpp.exe` + DLLs and scaffolds `config/models.toml`.

3. **Download a model (GGUF)** to your models drive (example: DeepSeek Q5_K_M into `D:\ai\models\deepseek\deepseek-q5.gguf`):
	https://huggingface.co/TheBloke/deepseek-llm-7B-chat-GGUF/resolve/main/deepseek-llm-7b-chat.Q8_0.gguf?download=true
		# Create a target folder
		New-Item -ItemType Directory -Force -Path 'D:\ai\models\deepseek' | Out-Null

		# Point to a GGUF URL (Q5_K_M is a good quality/size balance).
		# Replace the URL below with your preferred trusted source for a DeepSeek GGUF:
		$URL = 'https://example.local/gguf/deepseek-7b-q5_k_m.gguf'   # <-- put your real GGUF URL here
		$DEST = 'D:\ai\models\deepseek\deepseek-q5.gguf'

		# Download the file
		Invoke-WebRequest -Uri $URL -OutFile $DEST -UseBasicParsing

		# (Optional but recommended) Verify integrity if you have a published SHA256
		# Replace the expected hash below with the publisher-provided value
		$EXPECTED = ''  # e.g., '3F0B9A...'
		if ($EXPECTED) {
			$HASH = (Get-FileHash -Path $DEST -Algorithm SHA256).Hash.ToUpperInvariant()
			if ($HASH -ne $EXPECTED.ToUpperInvariant()) {
				throw "Checksum mismatch. Expected $EXPECTED but computed $HASH."
			} else {
				Write-Host "[OK] SHA256 verified: $HASH" -ForegroundColor Green
			}
		}

	Notes:
	- Choose a **Q5_K_M** (or Q4_K_M for smaller VRAM) GGUF built for DeepSeek 7B/8B style models.
	- Keep models **outside the repo**, e.g., `D:\ai\models\...`.
	- If the download host provides a `.sha256` file, use it for verification.

4. **Register your model** (writes a verified entry to `config/models.toml`):
		./scripts/New-ModelEntry.ps1 -Alias deepseek-q5 -ModelPath "D:/ai/models/deepseek/deepseek-q5.gguf"

5. **(Optional) Create a venv** and install the extras you need:
		python -m venv .venv
		.\.venv\Scripts\Activate.ps1
		python -m pip install --upgrade pip
		python -m pip install -r runtime\requirements-base.txt

6. **Run a smoke test**:
		python .\duel_of_minds.py data\conversations.db data\personas\camus.txt data\personas\nietzsche.txt `
			--backend llamacpp `
			--model-path "D:/ai/models/deepseek/deepseek-q5.gguf" `
			--max-turns 2 `
			--max-tokens 128 `
			--gpu-layers auto `
			--zero-network on `
			-v
	If `llama-cpp-python` is installed, it will be used. Otherwise the runner spawns `.\bin\llama.cpp.exe` via Vulkan.

Full walkthroughs live in `docs/windows-quickstart.md`.

---

## Model Placement & SHA256 Verification
- Keep GGUF weights on a separate drive (`D:\ai\models`).
- Declare each file in `config/models.toml` with alias, absolute path, and SHA256 hash.
- Use `scripts/New-ModelEntry.ps1` to compute and append verified entries:
		./scripts/New-ModelEntry.ps1 -Alias deepseek-q5 -ModelPath "D:/ai/models/deepseek/deepseek-q5.gguf" -Force
- Manual verification (if you already downloaded the file):
		(Get-FileHash -Path "D:\ai\models\deepseek\deepseek-q5.gguf" -Algorithm SHA256).Hash

The runner refuses to load any model not listed in the config file, preventing accidental use of tampered weights.

---

## AMD Vulkan Tuning Highlights
- Start with `--gpu-layers auto` to probe the Radeon iGPU/dGPU.
- Recommended quantizations: Q5_K_M for premium quality, Q4_K_M for extended runs.
- Use `--pause-jitter "700,4000"` and serialize turns to avoid iGPU contention.
- Additional tables, troubleshooting tips, and recovery steps are available in `docs/amd-vulkan-tuning.md`.

---

## CLI Essentials
Run:
	python .\duel_of_minds.py --help
Key flags:

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

## Troubleshooting
- `Model path does not exist`: Ensure the GGUF is present at the path you pass to `--model-path` and to `New-ModelEntry.ps1`. Paths must be absolute (e.g., `D:/ai/models/...`).
- Permissions: If downloads fail, run the shell **as Administrator** or choose a folder your account owns (e.g., somewhere under `D:\ai\`).
- Hash mismatch: Redownload from a trusted source and re-run the hash check.
- Vulkan startup issues: Reinstall/repair the AMD driver and re-run `Install-LlamaCppVulkan.ps1 -Force`.

---

## Need More?
- Windows 11 Quickstart: `docs/windows-quickstart.md`
- Model Placement & Integrity: `docs/model-prep.md`
- AMD Vulkan Tuning Cheatsheet: `docs/amd-vulkan-tuning.md`
- Design & Roadmap: `AGENTS.md`

Contributions should preserve the local-only, Windows-first philosophy and avoid bundling proprietary weights. Stay philosophical, stay offline.
