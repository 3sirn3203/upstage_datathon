"""
retriever_bm25.py — Lightweight in-memory BM25 retriever.

This module intentionally avoids heavy dependencies. It builds BM25 statistics
from chunk dictionaries and returns ranked chunk candidates for hybrid retrieval.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-_][A-Za-z0-9]+)*|\d+(?:[.,:-]\d+)*|[가-힣]+")


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 30


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]], config: BM25Config | None = None):
        self.config = config or BM25Config()
        self.chunks = chunks
        self.doc_tokens = [tokenize(chunk_text(chunk)) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freqs = self._build_doc_freqs()
        self.idf = self._build_idf()

    def _build_doc_freqs(self) -> dict[str, int]:
        doc_freqs: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freqs[token] += 1
        return dict(doc_freqs)

    def _build_idf(self) -> dict[str, float]:
        total_docs = len(self.doc_tokens)
        return {
            token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in self.doc_freqs.items()
        }

    def search(self, query: str | list[str], top_k: int | None = None) -> list[dict[str, Any]]:
        query_tokens = tokenize_query(query)
        if not query_tokens:
            return []

        scores = []
        query_counts = Counter(query_tokens)
        for doc_id, term_freq in enumerate(self.term_freqs):
            score = self._score_doc(term_freq, self.doc_lengths[doc_id], query_counts)
            if score > 0:
                scores.append((score, doc_id))

        limit = top_k or self.config.top_k
        scores.sort(key=lambda item: item[0], reverse=True)
        results = []
        for rank, (score, doc_id) in enumerate(scores[:limit], start=1):
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "chunk": self.chunks[doc_id],
                    "retriever": "bm25",
                }
            )
        return results

    def _score_doc(self, term_freq: Counter, doc_len: int, query_counts: Counter) -> float:
        if not self.avgdl:
            return 0.0

        score = 0.0
        k1 = self.config.k1
        b = self.config.b
        length_norm = k1 * (1 - b + b * doc_len / self.avgdl)

        for token, query_weight in query_counts.items():
            freq = term_freq.get(token, 0)
            if freq == 0:
                continue
            idf = self.idf.get(token, 0.0)
            score += query_weight * idf * (freq * (k1 + 1)) / (freq + length_norm)
        return score


def tokenize_query(query: str | list[str]) -> list[str]:
    if isinstance(query, list):
        query = " ".join(str(item) for item in query)
    return tokenize(str(query))


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        tokens.append(token)
        if is_hangul_token(token) and len(token) >= 4:
            tokens.extend(hangul_bigrams(token))
    return tokens


def is_hangul_token(token: str) -> bool:
    return all("가" <= char <= "힣" for char in token)


def hangul_bigrams(token: str) -> list[str]:
    return [token[idx : idx + 2] for idx in range(len(token) - 1)]


def chunk_text(chunk: dict[str, Any]) -> str:
    metadata = " ".join(
        str(chunk.get(key, ""))
        for key in ("source", "section", "kind")
        if chunk.get(key)
    )
    return f"{metadata}\n{chunk.get('text', '')}"


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def config_from_dict(config: dict[str, Any]) -> BM25Config:
    bm25 = config.get("indexing", {}).get("bm25", {})
    return BM25Config(
        k1=float(bm25.get("k1", 1.5)),
        b=float(bm25.get("b", 0.75)),
        top_k=int(bm25.get("top_k", 30)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect BM25 retrieval results.")
    parser.add_argument("chunks", type=Path)
    parser.add_argument("query", nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = BM25Retriever(load_chunks(args.chunks))
    for result in retriever.search(" ".join(args.query), top_k=args.top_k):
        chunk = result["chunk"]
        print(
            f"{result['rank']:>2}. {result['score']:.4f} "
            f"{chunk['chunk_id']} {chunk.get('source')} p{chunk.get('page')} {chunk.get('section')}"
        )


if __name__ == "__main__":
    main()
