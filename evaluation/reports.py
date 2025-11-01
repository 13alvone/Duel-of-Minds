"""Report generation helpers for Duel of Minds evaluations."""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Dict, List, Optional, Sequence, Tuple

from .metrics import compute_distinct_n, compute_self_bleu, compute_semantic_diversity


@dataclass
class ReplayTurn:
    turn_index: int
    speaker: str
    content: str


@dataclass
class LeakageFinding:
    turn_index: int
    speaker: str
    reason: str
    snippet: str


@dataclass
class LeakageReport:
    findings: List[LeakageFinding] = field(default_factory=list)
    total_messages: int = 0
    quote_threshold: int = 200

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_messages": int(self.total_messages),
            "quote_threshold": int(self.quote_threshold),
            "finding_count": int(len(self.findings)),
            "findings": [
                {
                    "turn_index": finding.turn_index,
                    "speaker": finding.speaker,
                    "reason": finding.reason,
                    "snippet": finding.snippet,
                }
                for finding in self.findings
            ],
        }


@dataclass
class EvaluationConfig:
    output_root: str
    limit_turns: Optional[int] = None
    matrix_label: Optional[str] = None
    quote_threshold: int = 200
    tcr_stride: int = 3
    timestamp: Optional[str] = None


@dataclass
class EvaluationResult:
    run_id: int
    output_dir: str
    transcript: List[ReplayTurn]
    metrics: Dict[str, object]
    leakage: LeakageReport


_COPYRIGHT_PATTERNS = [
    re.compile(r"copyright\s+\d{4}", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"permission is granted", re.IGNORECASE),
]


def load_transcript(conn: sqlite3.Connection, run_id: int, limit_turns: Optional[int] = None) -> List[ReplayTurn]:
    """Return ordered turns for the specified run."""

    query = "SELECT turn_index, speaker, content FROM messages WHERE run_id=? ORDER BY turn_index ASC"
    params: List[object] = [run_id]
    if limit_turns is not None and limit_turns > 0:
        query += " LIMIT ?"
        params.append(limit_turns)
    cur = conn.cursor()
    cur.execute(query, params)
    turns = [ReplayTurn(int(row[0]), str(row[1]), str(row[2])) for row in cur.fetchall()]
    return turns


def _blob_to_vector(blob: bytes) -> List[float]:
    from array import array

    arr = array("f")
    arr.frombytes(blob)
    return list(arr)


def _hash_embedding(text: str, dimension: int = 64) -> List[float]:
    token = text.strip() or "(empty)"
    values: List[float] = []
    for index in range(dimension):
        digest = sha256(f"{token}|{index}".encode("utf-8")).digest()
        int_val = int.from_bytes(digest[:4], "big", signed=False)
        scaled = (int_val / 0xFFFFFFFF) * 2.0 - 1.0
        values.append(float(scaled))
    return values


def _load_embeddings(
    conn: sqlite3.Connection,
    run_id: int,
    turns: Sequence[ReplayTurn],
    *,
    fallback_dimension: int = 64,
) -> Tuple[List[List[float]], str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT turn_index, embedding FROM message_embeddings WHERE run_id=? ORDER BY turn_index ASC",
        (run_id,),
    )
    stored: Dict[int, List[float]] = {}
    for turn_index, blob in cur.fetchall():
        if blob is None:
            continue
        stored[int(turn_index)] = _blob_to_vector(blob)
    vectors: List[List[float]] = []
    missing = 0
    for turn in turns:
        vector = stored.get(turn.turn_index)
        if vector is None:
            missing += 1
            vector = _hash_embedding(turn.content, fallback_dimension)
        vectors.append(vector)
    if not stored:
        source = "hash-fallback"
    elif missing == 0:
        source = "stored"
    else:
        source = "mixed"
    return vectors, source


def _detect_leakage(turns: Sequence[ReplayTurn], *, quote_threshold: int) -> LeakageReport:
    report = LeakageReport(total_messages=len(turns), quote_threshold=quote_threshold)
    quote_pattern = re.compile(r'"([^"\n]{%d,})"' % max(quote_threshold, 32))
    for turn in turns:
        content = turn.content
        for match in quote_pattern.finditer(content):
            excerpt = match.group(1).strip()
            snippet = (excerpt[:160] + "…") if len(excerpt) > 160 else excerpt
            report.findings.append(
                LeakageFinding(
                    turn_index=turn.turn_index,
                    speaker=turn.speaker,
                    reason="long_quote",
                    snippet=snippet,
                )
            )
        if any(pattern.search(content) for pattern in _COPYRIGHT_PATTERNS):
            sample = content.strip().splitlines()
            snippet = sample[0][:160] if sample else content[:160]
            report.findings.append(
                LeakageFinding(
                    turn_index=turn.turn_index,
                    speaker=turn.speaker,
                    reason="copyright_notice",
                    snippet=snippet,
                )
            )
    return report


def _render_metrics_table(metrics: Dict[str, object]) -> str:
    distinct = metrics.get("distinct_n", {})
    semantic = metrics.get("semantic_diversity", {})
    lines = ["| Metric | Value |", "| --- | --- |"]
    if distinct:
        for n, payload in distinct.items():
            ratio = payload.get("ratio", 0.0)
            lines.append(f"| distinct-{int(n)} | {ratio:.4f} |")
    self_bleu = metrics.get("self_bleu", {})
    if self_bleu:
        lines.append(f"| self-BLEU (avg) | {self_bleu.get('average', 0.0):.4f} |")
    if semantic:
        lines.append(
            "| semantic diversity | "
            f"{semantic.get('semantic_diversity', 0.0):.4f} (1 - avg cosine {semantic.get('average_cosine_similarity', 0.0):.4f}) |"
        )
    leakage = metrics.get("leakage", {})
    if leakage:
        lines.append(
            f"| leakage flags | {int(leakage.get('finding_count', 0))} finding(s) across {int(leakage.get('total_messages', 0))} messages |"
        )
    return "\n".join(lines)


def _render_tcr(turns: Sequence[ReplayTurn], stride: int) -> str:
    if stride <= 0:
        stride = 3
    blocks: List[str] = []
    for index in range(0, len(turns), stride):
        window = turns[index : index + stride]
        if len(window) < 2:
            continue
        title = f"## Turns {window[0].turn_index}–{window[-1].turn_index}"
        lines = [title]
        labels = ["Thesis", "Critique", "Revision"]
        for idx, turn in enumerate(window):
            label = labels[idx] if idx < len(labels) else f"Turn {turn.turn_index}"
            excerpt = turn.content.strip()
            if len(excerpt) > 400:
                excerpt = excerpt[:400] + "…"
            lines.append(f"### {label} — {turn.speaker} (turn {turn.turn_index})")
            lines.append("")
            lines.append(excerpt)
            lines.append("")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def generate_reports(
    conn: sqlite3.Connection,
    run_id: int,
    config: EvaluationConfig,
) -> EvaluationResult:
    turns = load_transcript(conn, run_id, config.limit_turns)
    if not turns:
        raise RuntimeError(f"No transcript data found for run_id={run_id}.")
    cur = conn.cursor()
    cur.execute(
        "SELECT started_at_utc, backend, model_name, persona_a_path, persona_b_path, params_json FROM runs WHERE id=?",
        (run_id,),
    )
    run_row = cur.fetchone()
    run_metadata: Dict[str, object] = {}
    if run_row:
        started_at_utc, backend, model_name, persona_a_path, persona_b_path, params_json = run_row
        run_metadata = {
            "started_at_utc": started_at_utc,
            "backend": backend,
            "model_name": model_name,
            "persona_a_path": persona_a_path,
            "persona_b_path": persona_b_path,
        }
        try:
            run_metadata["params"] = json.loads(params_json)
        except Exception:
            run_metadata["params"] = params_json
    messages = [turn.content for turn in turns]
    distinct = compute_distinct_n(messages)
    self_bleu = compute_self_bleu(messages)
    embeddings, embedding_source = _load_embeddings(conn, run_id, turns)
    semantic = compute_semantic_diversity(embeddings)
    semantic["embedding_source"] = embedding_source
    leakage = _detect_leakage(turns, quote_threshold=config.quote_threshold)
    metrics_payload: Dict[str, object] = {
        "run_id": run_id,
        "turn_count": len(turns),
        "distinct_n": {str(k): v for k, v in distinct.items()},
        "self_bleu": self_bleu,
        "semantic_diversity": semantic,
        "leakage": leakage.to_dict(),
    }
    if run_metadata:
        metrics_payload["run"] = run_metadata
    timestamp = config.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dir_name = f"run_{run_id}"
    if config.matrix_label:
        dir_name += f"_{config.matrix_label}"
    dir_name += f"_{timestamp}"
    output_dir = os.path.join(config.output_root, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2, ensure_ascii=False)
    report_lines = ["# Duel of Minds — Evaluation Report", ""]
    report_lines.append(f"- Run ID: {run_id}")
    report_lines.append(f"- Turns analysed: {len(turns)}")
    report_lines.append(f"- Embedding source: {semantic.get('embedding_source', 'unknown')}")
    report_lines.append("")
    report_lines.append(_render_metrics_table(metrics_payload))
    report_lines.append("")
    if leakage.findings:
        report_lines.append("## Leakage Findings")
        for finding in leakage.findings:
            report_lines.append(
                f"- Turn {finding.turn_index} ({finding.speaker}): {finding.reason} — {finding.snippet}"
            )
        report_lines.append("")
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))
    tcr_content = _render_tcr(turns, config.tcr_stride)
    tcr_path = os.path.join(output_dir, "tcr.md")
    with open(tcr_path, "w", encoding="utf-8") as fh:
        fh.write(tcr_content)
    metrics_payload["report_path"] = report_path
    metrics_payload["tcr_path"] = tcr_path
    return EvaluationResult(
        run_id=run_id,
        output_dir=output_dir,
        transcript=turns,
        metrics=metrics_payload,
        leakage=leakage,
    )
