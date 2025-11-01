# AMD Vulkan Tuning Cheatsheet

Target hardware: AMD AI Max 395+ platforms with integrated or discrete Radeon GPUs.

## 1. Probe and baseline
- Start with `--gpu-layers auto`. The runtime calls `runtime/gpu_probe.py` to pick a safe offload count.
- Monitor token throughput from the console (tokens/sec) and Radeon overlay (ALT+R).
- Keep system on AC power; disable battery saver and USB power throttling.

## 2. Layer offload strategy
| Model Size | Recommended Quantization | Starting GPU Layers | Notes |
|------------|-------------------------|---------------------|-------|
| 7B         | Q5_K_M / Q6_K           | 28–32               | Full offload usually viable on Radeon 780M/7600M.
| 13B        | Q4_K_M / Q5_K_M         | 22–28               | Watch VRAM utilization; drop to 20 if paging occurs.
| 70B (draft)| Q4_K_M                  | 8–12                | Requires large pagefile + patience; consider CPU-first fallback.

Adjust upward if frametimes remain low (<10 ms) and VRAM headroom > 1 GB. Drop down if you observe driver resets or throttling.

## 3. Precision & quantization tips
- Prefer Q5_K_M for dialectical quality; switch to Q4_K_M for long runs (>8h) if heat builds up.
- When mixing agent models (e.g., summarizer + debater), pin the heavier model to GPU and leave the lighter on CPU by launching two runs sequentially.

## 4. Temperature and pacing
- Use `--temperature 0.75` and `--top-p 0.92` for balanced novelty.
- Set `--pause-jitter "700,4000"` to desynchronize GPU spikes between agents.
- Cap `--max-tokens` to 512–768 to avoid runaway contexts; rely on summaries for memory.

## 5. Logging & diagnostics
- Enable debug logs (`-vv`) to capture GPU layer decisions.
- When the CLI backend is active, check `bin/llama.cpp.log` (if you pass `--log` flags manually) for Vulkan warnings.
- Keep Radeon Software set to "Standard" tuning profile; custom undervolts can destabilize long runs.

## 6. Recovery playbook
- If the CLI backend crashes, re-run `./scripts/Install-LlamaCppVulkan.ps1 -Force` to refresh binaries.
- If you see checksum mismatches, rerun `./scripts/New-ModelEntry.ps1 -Force ...` to update stale hashes.
- For persistent hangs, fall back to CPU-only mode by adding `--gpu-layers 0` temporarily.
