# Evaluation Workflow

The evaluation toolchain is designed to replay stored Duel of Minds transcripts and emit
metrics that measure novelty, semantic spread, and potential leakage. All artifacts are kept
under `outputs/` so that they can be archived or inspected manually after long runs.

## 1. Collecting Data

1. Launch a dialogue run with `dom run` and allow it to proceed for the desired wall-clock
   duration. When running on Windows + AMD Vulkan you can reuse the same llama.cpp Vulkan
   build described in [windows-quickstart](./windows-quickstart.md) and
   [amd-vulkan-tuning](./amd-vulkan-tuning.md).
2. For the testing matrix we operate three canonical budgets:
   - **2h smoke** — confirm short-run stability after persona or model changes.
   - **8h soak** — exercise nightly/weekly schedules and ensure novelty metrics remain healthy.
   - **24h endurance** — full validation after major releases or backend upgrades.
3. Use `--max-wallclock` (`2h`, `8h`, `24h`) when running to guarantee consistent cut-offs.
4. When a run completes (or you interrupt with `Ctrl+C`), confirm that
   `runtime/dom.db` contains the new run id via `python duel_of_minds.py run runtime/dom.db ... --resume`.

## 2. Running the Evaluator

### Basic usage

```powershell
# Evaluate the latest run in the database and drop artifacts under outputs/run_<id>_<timestamp>
python .\duel_of_minds.py eval runtime\dom.db
```

Key options:

- `--run-id <id>` — evaluate a specific run. Repeat to evaluate multiple runs.
- `--all` — replay every run stored in the database.
- `--matrix-label <slot>` — tag the output folder (e.g., `2h`, `8h`, `24h`) so it lines up with the testing matrix.
- `--limit-turns <N>` — cap the replay to the first `N` turns when you only need an early slice.
- `--quote-threshold <chars>` — adjust how many quoted characters trigger a leakage flag (default `200`).
- `--tcr-stride <turns>` — change the window used to assemble Thesis–Critique–Revision snapshots (default triplets).

Artifacts generated per run:

- `metrics.json` — machine-readable output including distinct-n, self-BLEU, semantic diversity, leakage summary, and run metadata.
- `report.md` — human-friendly summary with metric tables and leakage flags.
- `tcr.md` — Thesis–Critique–Revision snapshots built from sliding windows of the transcript.

### Testing matrix helper

The repository ships with `scripts/Invoke-EvaluationMatrix.ps1` to keep CI/manual testing aligned
with the 2h/8h/24h cadence.

```powershell
# Evaluate the latest run three times, tagging each slot.
# Pass -AllRuns to sweep the entire database instead.
PowerShell -ExecutionPolicy Bypass -File scripts\Invoke-EvaluationMatrix.ps1 -DatabasePath runtime\dom.db
```

You can forward `-RunIds 12,13,14` to pin individual runs or use `-LimitTurns` to trim evaluation
for quick feedback loops.

## 3. Interpreting Metrics

- **distinct-n (n = 1..3)** — ratio of unique n-grams to total n-grams across all turns. Values close to `1.0`
  mean that repetition is low; drops below `0.5` warrant review.
- **self-BLEU** — average BLEU score treating each turn as a hypothesis compared to the rest. Lower scores
  indicate that the conversation continues to introduce novel phrasing.
- **semantic diversity** — computed from stored message embeddings when available (falls back to deterministic
  hashes). Reported as `1 - average cosine similarity`; higher numbers signal a wider semantic spread.
- **Leakage checks** — the evaluator scans for unusually long quoted passages and copyright notices. Adjust
  `--quote-threshold` if your personas intentionally cite longer fragments.
- **TCR snapshots** — each window of turns is rendered as Thesis, Critique, Revision (or sequential fallbacks)
  to aid human reviewers in spotting regressions or stalled debates.

## 4. Failure Handling

- **Missing runs** — `dom eval` exits with a non-zero status if the requested run id does not exist. Verify the
  `runtime/dom.db` path and rerun `dom run` with `--resume` to populate transcripts.
- **Corrupt embeddings** — if message embeddings are missing the evaluator switches to the deterministic hash
  fallback and notes it in `report.md`. Re-run the dialogue with embeddings enabled (`--embedding-model` not `off`)
  for higher fidelity semantic metrics.
- **Leakage violations** — flagged turns appear in the Markdown report. Review persona prompts or backend configs;
  rerun after tightening guardrails (e.g., shorter `--max-reply-chars`, adjust growth checks).
- **CI aborts** — the PowerShell helper throws when any evaluation fails. Inspect the console output, fix the
  underlying issue (DB path, permissions, or run coverage), and rerun the script.

## 5. Archiving Results

Evaluation artifacts are meant to accumulate under `outputs/`. Keep the directory in version control so that CI
can cache or publish reports, but add subdirectories to your release notes or issue tracker as needed.

