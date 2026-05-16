"""
retriever_dense.py — FAISS dense retriever for query subqueries.

Each subquery is embedded with the configured local embedding model, searched
against the prebuilt FAISS index, and returned as its own ranked result list.
Hybrid merging is intentionally left to baseline_rag.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .index_corpus import DenseIndexConfig, config_from_dict, embed_queries
    from .parse_corpus import DEFAULT_CONFIG_PATH, load_config
except ImportError:
    from index_corpus import DenseIndexConfig, config_from_dict, embed_queries
    from parse_corpus import DEFAULT_CONFIG_PATH, load_config


class DenseRetriever:
    def __init__(
        self,
        *,
        faiss_path: str | Path,
        metadata_path: str | Path,
        config: DenseIndexConfig | None = None,
    ):
        self.config = config or DenseIndexConfig()
        self.faiss_path = Path(faiss_path)
        self.metadata_path = Path(metadata_path)

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "Dense retrieval requires faiss-cpu. Install project requirements or set "
                "indexing.dense.enabled: false for BM25-only retrieval."
            ) from exc

        self.index = faiss.read_index(str(self.faiss_path))
        self.metadata = load_metadata(self.metadata_path)

        if self.config.dimension > 0 and self.index.d != self.config.dimension:
            raise ValueError(
                f"FAISS dimension ({self.index.d}) does not match configured embedding dimension "
                f"({self.config.dimension}) for {self.config.model_name}"
            )
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"FAISS vectors ({self.index.ntotal}) and metadata rows ({len(self.metadata)}) differ"
            )

    def search(self, subquery: str, top_k: int = 10, label: str | None = None) -> list[dict[str, Any]]:
        query = subquery.strip()
        if not query or top_k <= 0:
            return []

        query_vector = embed_queries([query], self.config)
        scores, ids = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for rank, (score, vector_id) in enumerate(zip(scores[0], ids[0]), start=1):
            if vector_id < 0:
                continue
            chunk = self.metadata[int(vector_id)]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "chunk": chunk,
                    "retriever": "dense",
                    "retriever_name": label or "dense",
                    "query": query,
                }
            )
        return results

    def search_many(self, subqueries: list[str], top_k: int = 10) -> list[list[dict[str, Any]]]:
        result_lists = []
        for idx, subquery in enumerate(dedupe_queries(subqueries)):
            label = "dense:original" if idx == 0 else f"dense:sq{idx}"
            result_lists.append(self.search(subquery, top_k=top_k, label=label))
        return result_lists


def dedupe_queries(subqueries: Iterable[str]) -> list[str]:
    deduped = []
    seen = set()
    for subquery in subqueries:
        text = str(subquery).strip()
        if not text or text in seen:
            continue
        deduped.append(text)
        seen.add(text)
    return deduped


def load_metadata(path: Path) -> list[dict[str, Any]]:
    metadata = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))
    return metadata


def default_dense_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    parsing = config.get("parsing", {})
    dense = config.get("indexing", {}).get("dense", {})
    backend = parsing.get("backend", "pdfplumber")
    output_dir = Path(parsing.get("output_dir", "parsed_corpus")) / backend
    return (
        output_dir / dense.get("faiss_filename", "dense.faiss"),
        output_dir / dense.get("metadata_filename", "dense_metadata.jsonl"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dense FAISS retrieval results.")
    parser.add_argument("subquery", nargs="+")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--faiss", type=Path, dest="faiss_path")
    parser.add_argument("--metadata", type=Path, dest="metadata_path")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    raw_config = load_config(args.config)
    default_faiss_path, default_metadata_path = default_dense_paths(raw_config)
    retriever = DenseRetriever(
        faiss_path=args.faiss_path or default_faiss_path,
        metadata_path=args.metadata_path or default_metadata_path,
        config=config_from_dict(raw_config),
    )

    for results in retriever.search_many([" ".join(args.subquery)], top_k=args.top_k):
        for result in results:
            chunk = result["chunk"]
            print(
                f"{result['rank']:>2}. {result['score']:.4f} "
                f"{chunk['chunk_id']} {chunk.get('source')} p{chunk.get('page')} {chunk.get('section')}"
            )


if __name__ == "__main__":
    main()
