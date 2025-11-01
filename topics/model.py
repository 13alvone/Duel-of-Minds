"""Data structures describing dialogue topic graphs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import random


@dataclass(frozen=True)
class Transition:
    """An edge from one theme to another."""

    target: str
    cue: str = ""
    kind: str = "bridge"
    weight: float = 1.0
    tags: Tuple[str, ...] = ()

    def normalized_weight(self) -> float:
        return self.weight if self.weight > 0 else 1.0


@dataclass(frozen=True)
class Theme:
    """A topic focus area within the conversation graph."""

    key: str
    title: str
    summary: str
    guidance: Tuple[str, ...] = ()
    transitions: Tuple[Transition, ...] = ()
    tags: Tuple[str, ...] = ()


@dataclass
class TopicGraph:
    """A directed graph describing allowable topic pivots."""

    name: str
    entry_theme: Optional[str]
    themes: Dict[str, Theme]

    def theme(self, key: str) -> Theme:
        if key not in self.themes:
            raise KeyError(f"Unknown theme: {key}")
        return self.themes[key]

    def available_theme_ids(self) -> List[str]:
        return list(self.themes.keys())

    def pick_initial_theme(self, rng: random.Random) -> Theme:
        if self.entry_theme and self.entry_theme in self.themes:
            return self.themes[self.entry_theme]
        if not self.themes:
            raise ValueError("Topic graph has no themes to choose from.")
        return rng.choice(list(self.themes.values()))

    def next_theme(
        self,
        current: Optional[str],
        rng: random.Random,
        recent: Sequence[str] = (),
    ) -> Optional[Theme]:
        if not self.themes:
            return None
        if current and current not in self.themes:
            current = None
        if current is None:
            return self.pick_initial_theme(rng)
        theme = self.themes[current]
        weighted: List[Tuple[Transition, float]] = []
        for transition in theme.transitions:
            if transition.target not in self.themes:
                continue
            penalty = 0.5 if transition.target in recent else 1.0
            weight = max(transition.normalized_weight() * penalty, 0.0)
            if weight > 0:
                weighted.append((transition, weight))
        if not weighted:
            pool = [t for t in self.themes.values() if t.key != theme.key]
            if not pool:
                return theme
            return rng.choice(pool)
        total = sum(weight for _, weight in weighted)
        pick = rng.random() * total if total > 0 else 0
        upto = 0.0
        for transition, weight in weighted:
            upto += weight
            if pick <= upto:
                return self.themes[transition.target]
        return self.themes[weighted[-1][0].target]

    def preview_transitions(self, key: str) -> Tuple[Transition, ...]:
        if key not in self.themes:
            return ()
        return self.themes[key].transitions
