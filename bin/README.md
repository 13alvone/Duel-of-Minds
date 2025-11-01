# llama.cpp Vulkan binaries

This directory is intentionally kept empty in source control. Run `./scripts/Install-LlamaCppVulkan.ps1`
from a Windows PowerShell prompt to download the official Vulkan-enabled llama.cpp executables
(`llama.cpp.exe`, `llama-server.exe`, and supporting DLLs). The runner will look here when the
`llama-cpp-python` module is unavailable and spawn the standalone binaries directly.

> **Do not** commit model weights or third-party executables to Git. The install script fetches the
> latest trusted release on demand and keeps the repository lightweight.
