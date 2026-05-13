"""
retriever_merge.py — Merge BM25 and dense retrieval candidates.

The goal is not simply "highest scoring passage top-k". Without a reranker,
we build a balanced evidence set for generation by preserving provenance,
rewarding consensus, covering subquery facets, and limiting repeated context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MergeConfig:
    rrf_k: int = 30
    max_candidates: int = 40
    max_chunks_per_doc: int = 4
    max_chunks_per_section: int = 2
    coverage_weight: float = 0.25
    doc_repeat_penalty: float = 0.10


def config_from_dict(config: dict[str, Any]) -> MergeConfig:
    merge = config.get("retrieval", {}).get("merge", {})
    return MergeConfig(
        rrf_k=int(merge.get("rrf_k", 30)),
        max_candidates=int(merge.get("max_candidates", 40)),
        max_chunks_per_doc=int(merge.get("max_chunks_per_doc", 4)),
        max_chunks_per_section=int(merge.get("max_chunks_per_section", 2)),
        coverage_weight=float(merge.get("coverage_weight", 0.25)),
        doc_repeat_penalty=float(merge.get("doc_repeat_penalty", 0.10)),
    )


def merge_retrieval_results(
    *,
    bm25_results: list[dict[str, Any]],
    dense_result_lists: list[list[dict[str, Any]]],
    top_k: int,
    query_plan: dict[str, Any] | None = None,
    config: MergeConfig | None = None,
) -> list[dict[str, Any]]:
    """Return final ranked retrieval results for context construction."""
    del query_plan

    merge_config = config or MergeConfig()
    candidates = union_candidates(bm25_results, dense_result_lists)
    if not candidates:
        return []

    add_rrf_scores(candidates, merge_config.rrf_k)
    add_base_scores(candidates)
    candidates.sort(key=lambda candidate: candidate["base_score"], reverse=True)
    candidates = candidates[: merge_config.max_candidates]

    selected = coverage_aware_select(candidates, top_k, merge_config)
    selected = order_for_generation(selected)
    return [candidate_to_result(candidate, rank) for rank, candidate in enumerate(selected, start=1)]


def union_candidates(
    bm25_results: list[dict[str, Any]],
    dense_result_lists: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidate_map: dict[str, dict[str, Any]] = {}

    add_result_list(candidate_map, bm25_results, "bm25")
    for idx, result_list in enumerate(dense_result_lists):
        default_name = "dense:original" if idx == 0 else f"dense:sq{idx}"
        add_result_list(candidate_map, result_list, default_name)

    candidates = list(candidate_map.values())
    for candidate in candidates:
        candidate["covered_queries"] = compute_covered_queries(candidate)
        candidate["features"] = overlap_features(candidate)
    return candidates


def add_result_list(
    candidate_map: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    retriever_name: str,
) -> None:
    for result in results:
        chunk = result.get("chunk", {})
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue

        candidate = candidate_map.setdefault(
            chunk_id,
            {
                "chunk": chunk,
                "retrieved_from": {},
                "best_score": result.get("score", 0.0),
            },
        )
        source_name = result.get("retriever_name") or retriever_name
        candidate["retrieved_from"][source_name] = {
            "rank": int(result.get("rank", 9999)),
            "score": float(result.get("score", 0.0)),
            "query": result.get("query"),
        }
        candidate["best_score"] = max(float(candidate["best_score"]), float(result.get("score", 0.0)))


def add_rrf_scores(candidates: list[dict[str, Any]], rrf_k: int) -> None:
    rrf_values = []
    for candidate in candidates:
        rrf = sum(
            1.0 / (rrf_k + info["rank"])
            for info in candidate["retrieved_from"].values()
        )
        candidate["rrf"] = rrf
        rrf_values.append(rrf)

    min_rrf = min(rrf_values)
    max_rrf = max(rrf_values)
    denom = max_rrf - min_rrf + 1e-9
    for candidate in candidates:
        candidate["rrf_norm"] = (candidate["rrf"] - min_rrf) / denom


def overlap_features(candidate: dict[str, Any]) -> dict[str, Any]:
    sources = candidate["retrieved_from"].keys()
    dense_hits = [source for source in sources if source.startswith("dense:")]
    has_bm25 = "bm25" in sources
    return {
        "hit_count": len(candidate["retrieved_from"]),
        "has_bm25": has_bm25,
        "dense_hit_count": len(dense_hits),
        "hybrid_overlap": has_bm25 and bool(dense_hits),
        "multi_dense_overlap": len(dense_hits) >= 2,
        "original_query_hit": "dense:original" in sources,
    }


def compute_covered_queries(candidate: dict[str, Any]) -> set[str]:
    covered = set()
    for source in candidate["retrieved_from"]:
        if source == "bm25":
            covered.add("keyword")
        elif source == "dense:original":
            covered.add("original")
        elif source.startswith("dense:"):
            covered.add(source.replace("dense:", "", 1))
    return covered


def add_base_scores(candidates: list[dict[str, Any]]) -> None:
    for candidate in candidates:
        features = candidate["features"]
        dense_count = features["dense_hit_count"]
        candidate["base_score"] = (
            0.45 * candidate["rrf_norm"]
            + 0.20 * float(features["hybrid_overlap"])
            + 0.15 * min(dense_count / 3.0, 1.0)
            + 0.10 * float(features["original_query_hit"])
            + 0.10 * float(features["has_bm25"])
        )


def coverage_aware_select(
    candidates: list[dict[str, Any]],
    top_k: int,
    config: MergeConfig,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    covered_queries: set[str] = set()
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:
        best = max(
            remaining,
            key=lambda candidate: selection_score(candidate, selected, covered_queries, config),
        )
        selected.append(best)
        covered_queries |= best["covered_queries"]
        remaining.remove(best)
        remaining = [
            candidate
            for candidate in remaining
            if within_diversity_limits(candidate, selected, config)
        ]

    return selected


def selection_score(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    covered_queries: set[str],
    config: MergeConfig,
) -> float:
    uncovered = candidate["covered_queries"] - covered_queries
    coverage_bonus = sum(query_weight(query) for query in uncovered)
    same_doc_count = sum(source_doc(item) == source_doc(candidate) for item in selected)
    same_doc_penalty = max(0, same_doc_count - 1)
    return (
        candidate["base_score"]
        + config.coverage_weight * coverage_bonus
        - config.doc_repeat_penalty * same_doc_penalty
    )


def query_weight(query: str) -> float:
    if query == "original":
        return 1.0
    if query == "keyword":
        return 0.8
    return 0.7


def within_diversity_limits(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    config: MergeConfig,
) -> bool:
    doc = source_doc(candidate)
    section = section_key(candidate)
    same_doc = sum(source_doc(item) == doc for item in selected)
    same_section = sum(section_key(item) == section for item in selected)
    return same_doc < config.max_chunks_per_doc and same_section < config.max_chunks_per_section


def order_for_generation(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: (
            generation_group(candidate),
            -candidate["base_score"],
            source_doc(candidate),
            int(candidate["chunk"].get("page") or 0),
        ),
    )


def generation_group(candidate: dict[str, Any]) -> int:
    features = candidate["features"]
    if features["hybrid_overlap"]:
        return 0
    if features["original_query_hit"]:
        return 1
    if features["dense_hit_count"]:
        return 2
    if features["has_bm25"]:
        return 3
    return 4


def candidate_to_result(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": candidate["base_score"],
        "chunk": candidate["chunk"],
        "retriever": "hybrid",
        "retrieved_from": candidate["retrieved_from"],
        "covered_queries": sorted(candidate["covered_queries"]),
        "rrf": candidate["rrf"],
        "features": candidate["features"],
    }


def source_doc(candidate: dict[str, Any]) -> str:
    return str(candidate["chunk"].get("source", ""))


def section_key(candidate: dict[str, Any]) -> tuple[str, int, str]:
    chunk = candidate["chunk"]
    return (
        str(chunk.get("source", "")),
        int(chunk.get("page") or 0),
        str(chunk.get("section", "")),
    )
