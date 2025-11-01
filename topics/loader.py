"""Utilities for loading topic graphs from YAML files."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

from .model import Theme, TopicGraph, Transition


def _ensure_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field '{field}' must be a non-empty string.")
    return value.strip()


def _as_tuple(items: Any, field: str) -> Sequence[str]:
    if items is None:
        return ()
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        raise ValueError(f"Field '{field}' must be a list of strings.")
    cleaned: List[str] = []
    for entry in items:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"Entries in '{field}' must be non-empty strings.")
        cleaned.append(entry.strip())
    return tuple(cleaned)


def _parse_transition(raw: Dict[str, Any]) -> Transition:
    target = _ensure_str(raw.get("to"), "transitions[].to")
    cue_value = raw.get("cue")
    if cue_value is None:
        cue_text = ""
    elif isinstance(cue_value, str):
        cue_text = cue_value.strip()
    else:
        raise ValueError("Transition 'cue' must be a string if provided.")
    kind_value = raw.get("kind", "bridge")
    if kind_value is None:
        kind_text = "bridge"
    elif isinstance(kind_value, str):
        kind_text = kind_value.strip() or "bridge"
    else:
        raise ValueError("Transition 'kind' must be a string if provided.")
    weight = raw.get("weight", 1.0)
    if not isinstance(weight, (int, float)):
        raise ValueError("Transition 'weight' must be numeric.")
    tags = _as_tuple(raw.get("tags"), "transitions[].tags")
    return Transition(target=target, cue=cue_text, kind=kind_text, weight=float(weight), tags=tags)


def load_topic_graph(path: str) -> TopicGraph:
    resource = Path(path)
    if not resource.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Topic graph YAML must define a mapping at the top level.")

    name = _ensure_str(data.get("name", resource.stem), "name")
    entry = data.get("entry") or data.get("start")
    if entry is not None:
        entry = _ensure_str(entry, "entry")

    themes_raw = data.get("themes")
    if not isinstance(themes_raw, list) or not themes_raw:
        raise ValueError("Topic graph must provide a non-empty list of themes.")

    themes: Dict[str, Theme] = {}
    for raw_theme in themes_raw:
        if not isinstance(raw_theme, dict):
            raise ValueError("Each theme entry must be a mapping.")
        key = _ensure_str(raw_theme.get("id") or raw_theme.get("key"), "themes[].id")
        title = _ensure_str(raw_theme.get("title", key), "themes[].title")
        summary = _ensure_str(raw_theme.get("summary"), "themes[].summary")
        guidance = _as_tuple(raw_theme.get("guidance"), "themes[].guidance")
        tags = _as_tuple(raw_theme.get("tags"), "themes[].tags")
        transitions_raw = raw_theme.get("transitions") or []
        if not isinstance(transitions_raw, list):
            raise ValueError("Theme 'transitions' must be a list of mappings.")
        transitions = tuple(_parse_transition(item) for item in transitions_raw)
        themes[key] = Theme(
            key=key,
            title=title,
            summary=summary,
            guidance=guidance,
            transitions=transitions,
            tags=tags,
        )

    if entry and entry not in themes:
        raise ValueError(f"Entry theme '{entry}' not found in graph.")

    for theme in themes.values():
        for transition in theme.transitions:
            if transition.target not in themes:
                raise ValueError(
                    f"Transition from '{theme.key}' targets unknown theme '{transition.target}'."
                )

    graph = TopicGraph(name=name, entry_theme=entry, themes=themes)
    return graph
