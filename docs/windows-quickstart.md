# Windows 11 Quickstart

This walkthrough targets Windows 11 on AMD AI Max 395-class laptops/desktops. It keeps everything offline and stores only local state.

## 1. Prerequisites
- Windows 11 23H2 or newer.
- Latest AMD Radeon drivers (Adrenalin Edition) with Vulkan runtime.
- PowerShell 7+ (or Windows PowerShell 5.1 with TLS 1.2 enabled).
- Git and Python 3.11+ installed.
- At least 128 GB RAM and fast NVMe storage (per project baseline).

## 2. Clone the repository
```powershell
cd D:\workspace
git clone https://github.com/<you>/Duel-of-Minds.git
cd Duel-of-Minds
```

## 3. Fetch llama.cpp Vulkan binaries
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
./scripts/Install-LlamaCppVulkan.ps1 -VersionTag b3532
```
- Downloads the official Vulkan-enabled build.
- Extracts `llama.cpp.exe`, `llama-server.exe`, and related DLLs into `./bin`.
- Creates `config/models.toml` with placeholder entries if it is missing.

## 4. Prepare a virtual environment (optional but recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r runtime\requirements-base.txt  # optional helper list
```
> Create `runtime/requirements-base.txt` if you maintain a pinned dependency set. Otherwise install only what you need, e.g. `pip install llama-cpp-python onnxruntime tokenizers`.

## 5. Register your GGUF models
1. Download GGUF weights from a trusted source to a **separate** drive (e.g., `D:\ai\models`).
2. Calculate SHA256 and append an entry to `config/models.toml`:
```powershell
./scripts/New-ModelEntry.ps1 -Alias deepseek-q5 -ModelPath "D:/ai/models/deepseek/deepseek-q5.gguf"
```
3. Repeat for every model variant you plan to load.

## 6. Run a smoke test
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
The runner will attempt to use the Python bindings first. If they are missing it automatically falls back to the Vulkan CLI binary dropped into `./bin`.

## 7. Next steps
- Customize persona files in `data/personas/`.
- Explore advanced options with `python .\duel_of_minds.py --help`.
- Review `docs/model-prep.md` for SHA verification and storage practices.
- Review `docs/amd-vulkan-tuning.md` to optimize throughput on Radeon GPUs.
