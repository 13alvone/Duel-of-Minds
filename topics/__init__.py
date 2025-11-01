"""Topic graph data structures and loader utilities."""
from .model import Theme, Transition, TopicGraph


def load_topic_graph(path: str) -> TopicGraph:
    """Load a topic graph from a YAML file."""
    from .loader import load_topic_graph as _load  # Imported lazily to avoid heavy deps at import time.

    return _load(path)


__all__ = ["Theme", "Transition", "TopicGraph", "load_topic_graph"]
