# Model Placement & Integrity

Keep model weights outside the repository and never commit `.gguf` files. This guide summarizes the storage layout and checksum workflow expected by Duel of Minds.

## Storage layout
- Recommended root: `D:\ai\models` (fast NVMe drive).
- Organize by family and quantization, e.g.:
```
D:\ai\models\deepseek\deepseek-r1-q5_k_m.gguf
D:\ai\models\meta\llama3\llama3.1-instruct-q4_k_m.gguf
D:\ai\models\mistral\mistral-instruct-q6_k.gguf
```
- Keep this repo cloned separately (e.g., `D:\workspace\Duel-of-Minds`).

## Adding an entry to config/models.toml
1. Confirm the GGUF file resides on a trusted local drive.
2. Run the helper script to compute SHA256 and append the entry:
```powershell
./scripts/New-ModelEntry.ps1 -Alias llama3-q4 -ModelPath "D:/ai/models/meta/llama3/llama3.1-instruct-q4_k_m.gguf"
```
3. Inspect `config/models.toml` to ensure the alias, path, and checksum look correct.

## Manual SHA256 verification
If you prefer to verify checksums yourself:

### PowerShell
```powershell
Get-FileHash -Path "D:/ai/models/meta/llama3/llama3.1-instruct-q4_k_m.gguf" -Algorithm SHA256
```
Compare the `Hash` field to the value recorded in `config/models.toml`.

### Python (cross-platform)
```python
import hashlib
path = r"D:/ai/models/meta/llama3/llama3.1-instruct-q4_k_m.gguf"
sha = hashlib.sha256()
with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        sha.update(chunk)
print(sha.hexdigest())
```

## Trust but verify
- Only download from reputable mirrors (e.g., Hugging Face organizations with signatures).
- Maintain a text log (e.g., `models/README-local.md`) mapping download sources to hashes for auditability.
- Rerun the hash script whenever you relocate files or suspect corruption.

## Cleaning up
If you retire a model, remove the corresponding `[[models]]` block from `config/models.toml`. The runner will refuse to load any GGUF not declared in the file.
