"""GPU probing utilities for selecting llama.cpp GPU layer counts.

When the CLI is invoked with ``--gpu-layers=auto`` we probe the host to
identify likely GPU vendors and provide a heuristic layer count that
works well for the Vulkan build on AMD-heavy Windows systems while still
behaving sensibly on other hardware.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
from typing import List, Optional

_DEFAULT_LAYERS = 35
_AMD_WINDOWS_RECOMMENDATIONS = {
    4096: 40,
    8192: 32,
}


def _run_command(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError, OSError) as exc:
        logging.debug(f"[DEBUG] gpu_probe command failed {cmd}: {exc}")
        return None


def _parse_vulkaninfo() -> List[str]:
    if not shutil.which("vulkaninfo"):
        return []
    output = _run_command(["vulkaninfo", "--summary"])
    if not output:
        return []
    names = []
    for line in output.splitlines():
        if "GPU id" in line or "deviceName" in line:
            cleaned = line.split(":", 1)[-1].strip()
            if cleaned:
                names.append(cleaned)
    return names


def _parse_wmic() -> List[str]:
    if platform.system().lower() != "windows":
        return []
    output = _run_command(["wmic", "path", "win32_VideoController", "get", "Name"])
    if not output:
        return []
    names = [line.strip() for line in output.splitlines() if line.strip() and "Name" not in line]
    return names


def _parse_lspci() -> List[str]:
    if not shutil.which("lspci"):
        return []
    output = _run_command(["lspci"])
    if not output:
        return []
    pattern = re.compile(r"VGA compatible controller: (.+)")
    names = []
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            names.append(match.group(1))
    return names


def _collect_gpu_names() -> List[str]:
    names: List[str] = []
    env_hint = os.environ.get("DOM_GPU_NAME")
    if env_hint:
        names.append(env_hint)
    names.extend(_parse_vulkaninfo())
    names.extend(_parse_wmic())
    names.extend(_parse_lspci())
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def _infer_vendor(names: List[str]) -> Optional[str]:
    for name in names:
        lowered = name.lower()
        if "amd" in lowered or "radeon" in lowered:
            return "AMD"
        if "nvidia" in lowered or "geforce" in lowered:
            return "NVIDIA"
        if "intel" in lowered:
            return "INTEL"
    return None


def auto_select_gpu_layers(ctx_size: int = 4096, default: int = _DEFAULT_LAYERS) -> int:
    """Heuristically choose a GPU layer count based on detected hardware."""

    names = _collect_gpu_names()
    vendor = _infer_vendor(names) or "UNKNOWN"
    system = platform.system().lower()

    logging.debug(f"[DEBUG] GPU probe detected vendor={vendor}, system={system}, names={names}")

    if vendor == "AMD" and system == "windows":
        for max_ctx, layers in _AMD_WINDOWS_RECOMMENDATIONS.items():
            if ctx_size <= max_ctx:
                return layers
        return 28

    if vendor == "AMD":
        return max(24, min(40, default))

    if vendor == "NVIDIA":
        return max(0, min(60, default + 5))

    if vendor == "INTEL":
        return max(0, min(20, default // 2))

    # If we reached here, either we detected no GPU or the vendor is unknown.
    # Fall back to a conservative default that still favors partial offload.
    return default


__all__ = ["auto_select_gpu_layers"]
