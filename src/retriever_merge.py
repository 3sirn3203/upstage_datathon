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
    iteration_decay: float = 1.0
    missing_need_bonus: float = 0.0
    cross_iteration_bonus: float = 0.0
    min_followup_chunks: int = 0


def config_from_dict(config: dict[str, Any]) -> MergeConfig:
    retrieval = config.get("retrieval", {})
    merge = retrieval.get("merge", {})
    iterative = retrieval.get("iterative", {})
    return MergeConfig(
        rrf_k=int(merge.get("rrf_k", 30)),
        max_candidates=int(merge.get("max_candidates", 40)),
        max_chunks_per_doc=int(merge.get("max_chunks_per_doc", 4)),
        max_chunks_per_section=int(merge.get("max_chunks_per_section", 2)),
        coverage_weight=float(merge.get("coverage_weight", 0.25)),
        doc_repeat_penalty=float(merge.get("doc_repeat_penalty", 0.10)),
        iteration_decay=float(iterative.get("iteration_decay", 1.0)),
        missing_need_bonus=float(iterative.get("missing_need_bonus", 0.0)),
        cross_iteration_bonus=float(iterative.get("cross_iteration_bonus", 0.0)),
        min_followup_chunks=int(iterative.get("min_followup_chunks", 0)),
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

    add_rrf_scores(candidates, merge_config.rrf_k, merge_config.iteration_decay)
    add_base_scores(candidates, merge_config)
    candidates.sort(key=lambda candidate: candidate["base_score"], reverse=True)
    candidates = retain_candidate_pool(candidates, merge_config)

    selected = coverage_aware_select(candidates, top_k, merge_config)
    selected = ensure_min_followup_chunks(selected, candidates, top_k, merge_config)
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


def add_rrf_scores(candidates: list[dict[str, Any]], rrf_k: int, iteration_decay: float = 1.0) -> None:
    rrf_values = []
    for candidate in candidates:
        rrf = sum(
            source_iteration_weight(source, iteration_decay) / (rrf_k + info["rank"])
            for source, info in candidate["retrieved_from"].items()
        )
        candidate["rrf"] = rrf
        rrf_values.append(rrf)

    min_rrf = min(rrf_values)
    max_rrf = max(rrf_values)
    denom = max_rrf - min_rrf + 1e-9
    for candidate in candidates:
        candidate["rrf_norm"] = (candidate["rrf"] - min_rrf) / denom


def overlap_features(candidate: dict[str, Any]) -> dict[str, Any]:
    sources = list(candidate["retrieved_from"].keys())
    dense_hits = [source for source in sources if is_dense_source(source)]
    has_bm25 = any(is_bm25_source(source) for source in sources)
    matched_iterations = sorted({source_iteration(source) for source in sources})
    return {
        "hit_count": len(candidate["retrieved_from"]),
        "has_bm25": has_bm25,
        "dense_hit_count": len(dense_hits),
        "hybrid_overlap": has_bm25 and bool(dense_hits),
        "multi_dense_overlap": len(dense_hits) >= 2,
        "original_query_hit": any(is_original_dense_source(source) for source in sources),
        "missing_need_hit": any(is_missing_need_source(source) for source in sources),
        "followup_hit": any(iteration > 0 for iteration in matched_iterations),
        "cross_iteration_hit": len(matched_iterations) >= 2,
        "matched_iterations": matched_iterations,
    }


def compute_covered_queries(candidate: dict[str, Any]) -> set[str]:
    covered = set()
    for source in candidate["retrieved_from"]:
        if is_bm25_source(source):
            covered.add("keyword")
        elif is_dense_source(source):
            covered.add(dense_query_label(source))
    return covered


def add_base_scores(candidates: list[dict[str, Any]], config: MergeConfig) -> None:
    for candidate in candidates:
        features = candidate["features"]
        dense_count = features["dense_hit_count"]
        candidate["base_score"] = (
            0.45 * candidate["rrf_norm"]
            + 0.20 * float(features["hybrid_overlap"])
            + 0.15 * min(dense_count / 3.0, 1.0)
            + 0.10 * float(features["original_query_hit"])
            + 0.10 * float(features["has_bm25"])
            + config.missing_need_bonus * float(features["missing_need_hit"])
            + config.cross_iteration_bonus * float(features["cross_iteration_hit"])
        )


def retain_candidate_pool(candidates: list[dict[str, Any]], config: MergeConfig) -> list[dict[str, Any]]:
    if len(candidates) <= config.max_candidates:
        return candidates

    retained = candidates[: config.max_candidates]
    required_followups = min(
        int(config.min_followup_chunks),
        len({candidate_chunk_id(candidate) for candidate in candidates if candidate["features"]["followup_hit"]}),
    )
    if required_followups <= 0 or count_followup_chunks(retained) >= required_followups:
        return retained

    retained_ids = {candidate_chunk_id(candidate) for candidate in retained}
    for candidate in candidates[config.max_candidates :]:
        if count_followup_chunks(retained) >= required_followups:
            break
        if not candidate["features"]["followup_hit"]:
            continue
        candidate_id = candidate_chunk_id(candidate)
        if candidate_id in retained_ids:
            continue
        replace_idx = find_followup_replacement_index(retained)
        if replace_idx is None:
            break
        replaced_id = candidate_chunk_id(retained[replace_idx])
        retained[replace_idx] = candidate
        retained_ids.discard(replaced_id)
        retained_ids.add(candidate_id)

    return retained


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


def ensure_min_followup_chunks(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    top_k: int,
    config: MergeConfig,
) -> list[dict[str, Any]]:
    required = int(config.min_followup_chunks)
    if required <= 0 or top_k <= 0:
        return selected

    followup_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("features", {}).get("followup_hit")
    ]
    if not followup_candidates:
        return selected

    required = min(required, len({candidate_chunk_id(candidate) for candidate in followup_candidates}))
    selected_ids = {candidate_chunk_id(candidate) for candidate in selected}
    followup_count = count_followup_chunks(selected)
    if followup_count >= required:
        return selected

    ranked_followups = sorted(followup_candidates, key=lambda item: item["base_score"], reverse=True)
    for candidate in ranked_followups:
        if followup_count >= required:
            break
        candidate_id = candidate_chunk_id(candidate)
        if candidate_id in selected_ids:
            continue

        if len(selected) < top_k and within_diversity_limits(candidate, selected, config):
            selected.append(candidate)
            selected_ids.add(candidate_id)
            followup_count += 1
            continue

        replace_idx = find_followup_replacement_index(selected)
        if replace_idx is None:
            continue
        trial_selected = selected[:replace_idx] + selected[replace_idx + 1 :]
        if not within_diversity_limits(candidate, trial_selected, config):
            continue
        replaced_id = candidate_chunk_id(selected[replace_idx])
        selected[replace_idx] = candidate
        selected_ids.discard(replaced_id)
        selected_ids.add(candidate_id)
        followup_count += 1

    return selected


def count_followup_chunks(candidates: list[dict[str, Any]]) -> int:
    return sum(1 for candidate in candidates if candidate.get("features", {}).get("followup_hit"))


def find_followup_replacement_index(selected: list[dict[str, Any]]) -> int | None:
    replaceable = [
        (candidate.get("base_score", 0.0), idx)
        for idx, candidate in enumerate(selected)
        if not candidate.get("features", {}).get("followup_hit")
    ]
    if not replaceable:
        return None
    return min(replaceable)[1]


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
    if query.startswith("missing"):
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


def candidate_chunk_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("chunk", {}).get("chunk_id", ""))


def source_iteration(source: str) -> int:
    for part in str(source).split(":"):
        if part.startswith("iter") and part[4:].isdigit():
            return int(part[4:])
    return 0


def source_iteration_weight(source: str, decay: float) -> float:
    iteration = source_iteration(source)
    if iteration <= 0:
        return 1.0
    return max(float(decay), 0.0) ** iteration


def is_bm25_source(source: str) -> bool:
    return str(source).split(":")[-1] == "bm25"


def is_dense_source(source: str) -> bool:
    return "dense" in str(source).split(":")


def dense_query_label(source: str) -> str:
    parts = str(source).split(":")
    if "dense" not in parts:
        return "dense"
    dense_idx = parts.index("dense")
    if dense_idx + 1 >= len(parts):
        return "dense"
    return parts[dense_idx + 1]


def is_original_dense_source(source: str) -> bool:
    return is_dense_source(source) and dense_query_label(source) == "original"


def is_missing_need_source(source: str) -> bool:
    return is_dense_source(source) and dense_query_label(source).startswith("missing")
