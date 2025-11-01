"""Core evaluation metrics for Duel of Minds transcripts."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Tokenize text into a simple word/punctuation stream."""

    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def _extract_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_distinct_n(messages: Iterable[str], max_n: int = 3) -> Dict[int, Dict[str, float]]:
    """Compute distinct-n ratios for the provided messages."""

    tokenized = [tokenize(message) for message in messages]
    results: Dict[int, Dict[str, float]] = {}
    for n in range(1, max(1, max_n) + 1):
        ngram_list: List[Tuple[str, ...]] = []
        for tokens in tokenized:
            ngram_list.extend(_extract_ngrams(tokens, n))
        total = len(ngram_list)
        unique = len(set(ngram_list)) if ngram_list else 0
        ratio = (unique / total) if total else 0.0
        results[n] = {
            "total": float(total),
            "unique": float(unique),
            "ratio": float(ratio),
        }
    return results


def _ngram_counter(tokens: Sequence[str], n: int) -> Counter:
    return Counter(_extract_ngrams(tokens, n))


def _modified_precision(hypothesis: Sequence[str], references: Sequence[Sequence[str]], n: int) -> float:
    hyp_counts = _ngram_counter(hypothesis, n)
    if not hyp_counts:
        return 0.0
    max_ref_counts: Counter = Counter()
    for reference in references:
        ref_counter = _ngram_counter(reference, n)
        for ngram, count in ref_counter.items():
            max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)
    clipped = 0
    for ngram, count in hyp_counts.items():
        clipped += min(count, max_ref_counts.get(ngram, 0))
    smoothing = 1.0
    total = sum(hyp_counts.values())
    return (clipped + smoothing) / (total + smoothing)


def _brevity_penalty(hypothesis_len: int, reference_lens: Sequence[int]) -> float:
    if not reference_lens:
        return 1.0
    if hypothesis_len == 0:
        return 0.0
    closest_ref = min(reference_lens, key=lambda ref_len: (abs(ref_len - hypothesis_len), ref_len))
    if hypothesis_len > closest_ref:
        return 1.0
    return math.exp(1.0 - float(closest_ref) / float(hypothesis_len))


def compute_self_bleu(messages: Sequence[str], max_n: int = 4) -> Dict[str, float]:
    """Compute self-BLEU by treating each message as the hypothesis against all others."""

    tokenized = [tokenize(message) for message in messages]
    count = len(tokenized)
    if count <= 1:
        return {
            "average": 0.0,
            "count": float(count),
            "max": 0.0,
            "min": 0.0,
        }
    weights = [1.0 / max_n] * max_n
    scores: List[float] = []
    for idx, hypothesis in enumerate(tokenized):
        references = [tokens for j, tokens in enumerate(tokenized) if j != idx and tokens]
        if not references or not hypothesis:
            scores.append(0.0)
            continue
        precisions = []
        for n in range(1, max_n + 1):
            precisions.append(max(_modified_precision(hypothesis, references, n), 1e-9))
        geo_mean = math.exp(sum(weight * math.log(p) for weight, p in zip(weights, precisions)))
        bp = _brevity_penalty(len(hypothesis), [len(ref) for ref in references])
        score = bp * geo_mean
        scores.append(float(score))
    average = sum(scores) / len(scores) if scores else 0.0
    return {
        "average": float(average),
        "count": float(len(scores)),
        "max": float(max(scores) if scores else 0.0),
        "min": float(min(scores) if scores else 0.0),
    }


def compute_semantic_diversity(vectors: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Compute average pairwise cosine similarity and derived diversity score."""

    vector_list = [list(vec) for vec in vectors if vec]
    pair_count = 0
    cosine_total = 0.0
    for i in range(len(vector_list)):
        vec_a = vector_list[i]
        norm_a = math.sqrt(sum(x * x for x in vec_a))
        if norm_a == 0.0:
            continue
        for j in range(i + 1, len(vector_list)):
            vec_b = vector_list[j]
            norm_b = math.sqrt(sum(y * y for y in vec_b))
            if norm_b == 0.0:
                continue
            dot = sum(x * y for x, y in zip(vec_a, vec_b))
            cosine = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
            cosine_total += cosine
            pair_count += 1
    if pair_count == 0:
        return {
            "average_cosine_similarity": 0.0,
            "pair_count": 0.0,
            "semantic_diversity": 0.0,
        }
    average_cosine = cosine_total / pair_count
    diversity = 1.0 - average_cosine
    return {
        "average_cosine_similarity": float(average_cosine),
        "pair_count": float(pair_count),
        "semantic_diversity": float(diversity),
    }
