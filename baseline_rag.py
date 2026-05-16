"""
baseline_rag.py — RAG 파이프라인 스켈레톤 (Starter Kit)

본 베이스라인은 해커톤 참가를 위한 기본 구조를 제공합니다.

── 지켜야 할 제약 사항 ─────────────────────────────────────
1. 입력  : load_test_suite() 로 질문 목록을 받습니다.
2. 출력  : tracker.save_csv("submission.csv") 로 제출 파일을 생성합니다.

── 커스텀 설계 영역 ────────────────────────────────────────
파싱, 청킹, 임베딩, 검색, 프롬프트, 생성, 보안 필터 등
그 외 모든 로직은 자유롭게 설계 및 구현이 가능합니다.

── 실행 방법 ──────────────────────────────────────────────
$ python baseline_rag.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


from decryptor import load_test_suite
from upstage_tracker import DEFAULT_MODEL, UPSTAGE_BASE_URL, UpstageTracker
from src.chunk_corpus import chunk_corpus, config_from_dict
from src.index_corpus import build_dense_index, config_from_dict as dense_config_from_dict
from src.logging_utils import log_block
from src.parse_corpus import DEFAULT_CONFIG_PATH, load_config, parse_corpus
from src.attack_sanitizer import sanitize_retrieval_results
from src.prompt import (
    CONTEXT_EVALUATION_PROMPT,
    DRAFT_GENERATION_PROMPT,
    FINAL_SAFETY_PROMPT,
    QUERY_ANALYSIS_PROMPT,
)
from src.retriever_bm25 import BM25Retriever, config_from_dict as bm25_config_from_dict
from src.retriever_merge import config_from_dict as merge_config_from_dict, merge_retrieval_results
from validator import validate

CORPUS_DIR      = "distribution/corpus"
TEST_SUITE_PATH = "distribution/test_suite/Encrypted_Test_Suite.json"
CONFIG          = load_config(DEFAULT_CONFIG_PATH)


def dense_enabled() -> bool:
    return bool(CONFIG.get("indexing", {}).get("dense", {}).get("enabled", True))


def load_chunks(path: str | Path) -> list[dict]:
    chunks = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def llm_config(stage: str) -> dict:
    llm = CONFIG.get("llm", {})
    stage_config = llm.get(stage, {})
    return {
        "model": stage_config.get("model", DEFAULT_MODEL),
        "temperature": stage_config.get("temperature", 0),
        "max_tokens": stage_config.get("max_tokens", 768),
        "enabled": stage_config.get("enabled", True),
    }


def call_solar_no_record(
    *,
    messages: list[dict],
    system_prompt: str | None = None,
    model: str | None = None,
    temperature: float = 0,
    max_tokens: int = 768,
) -> str:
    """Call Solar without appending a submission row to UpstageTracker.records."""
    api_key = os.environ.get("UPSTAGE_API_KEY")
    if not api_key:
        raise EnvironmentError("UPSTAGE_API_KEY is required for intermediate LLM stages.")

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": full_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url=f"{UPSTAGE_BASE_URL}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"Upstage API 오류 [{e.code}]: {body}") from e

    return raw["choices"][0]["message"]["content"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1.  인덱스 구축  (오프라인 — 파이프라인 실행 전 1회)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_index(corpus_dir: str):
    """
    PDF 코퍼스를 파싱·청킹하고 검색 인덱스를 반환합니다.
    """
    parsing_config = CONFIG.get("parsing", {})
    parsed_path = parse_corpus(
        corpus_dir=corpus_dir,
        option="pdfplumber",
        output_dir=parsing_config.get("output_dir", "parsed_corpus"),
        config_path=DEFAULT_CONFIG_PATH,
        force=parsing_config.get("force", False),
    )
    log_block("Index Build", "Parse selected", f"pdfplumber pages: {parsed_path}")

    chunks_path = chunk_corpus(
        parsed_path,
        config=config_from_dict(CONFIG),
    )
    chunks = load_chunks(chunks_path)
    log_block("Index Build", "Chunk summary", f"Chunks: {len(chunks)}\nPath: {chunks_path}")

    bm25_retriever = BM25Retriever(chunks, config=bm25_config_from_dict(CONFIG))
    dense_index = None
    dense_retriever = None
    if dense_enabled():
        from src.retriever_dense import DenseRetriever

        dense_index = build_dense_index(
            chunks_path,
            config=dense_config_from_dict(CONFIG),
        )
        dense_retriever = DenseRetriever(
            faiss_path=dense_index["faiss_path"],
            metadata_path=dense_index["metadata_path"],
            config=dense_config_from_dict(CONFIG),
        )
        retriever_summary = f"Dense index: {dense_index['faiss_path']}\nVectors: {dense_index['num_vectors']}"
    else:
        retriever_summary = "Dense retrieval disabled by indexing.dense.enabled=false. Using BM25 only."

    log_block("Index Build", "Retriever summary", retriever_summary)

    return {
        "parsed_path": parsed_path,
        "chunks_path": chunks_path,
        "chunks": chunks,
        "dense": dense_index,
        "bm25_retriever": bm25_retriever,
        "dense_retriever": dense_retriever,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONLINE STAGE 1. Query analysis  (LLM, CSV 기록 없음)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_query(question: str) -> dict:
    config = llm_config("query_analysis")
    fallback = {
        "keywords": [question],
        "subqueries": [question],
    }
    if not config["enabled"]:
        return fallback

    content = call_solar_no_record(
        system_prompt=QUERY_ANALYSIS_PROMPT,
        messages=[{"role": "user", "content": question}],
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    try:
        parsed = json.loads(extract_json_object(content))
        log_block(
            "Stage 1",
            "Query analysis",
            json.dumps(parsed, ensure_ascii=False, indent=2),
        )
    except (json.JSONDecodeError, ValueError):
        parsed = fallback
        log_block(
            "Stage 1",
            "Query analysis fallback",
            "Failed to parse JSON. Using fallback query plan.",
        )

    return normalize_query_plan(parsed, question)


def normalize_query_plan(parsed: dict, question: str) -> dict:
    keywords = coerce_string_list(
        parsed.get("keywords", parsed.get("bm25_keywords")),
        fallback=[question],
        limit=10,
    )
    subqueries = coerce_string_list(
        parsed.get("subqueries", parsed.get("dense_subqueries")),
        fallback=[question],
        limit=3,
    )
    if not subqueries:
        subqueries = [question]

    return {
        "keywords": keywords,
        "subqueries": subqueries,
    }


def coerce_string_list(value, *, fallback: list[str], limit: int) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = fallback

    cleaned = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    return text[start : end + 1]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONLINE STAGE 2. Hybrid retrieval  (BM25 + dense, 로컬)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def retrieve(question: str, query_plan: dict, index, top_k: int = 8) -> str:
    """질문과 query plan을 바탕으로 관련 chunk context를 반환합니다."""
    retrieval_run = retrieve_once(
        question=question,
        query_plan=query_plan,
        index=index,
        iteration=0,
        include_original=True,
        dense_label_prefix="sq",
    )
    merged_results = merge_accumulated_retrieval_results([retrieval_run], query_plan, top_k=top_k)
    return format_context(merged_results)


def retrieve_once(
    *,
    question: str,
    query_plan: dict,
    index,
    iteration: int,
    include_original: bool,
    dense_label_prefix: str,
) -> dict[str, Any]:
    """Run one retrieval pass and keep structured results for later union/rerank."""
    retrieval_config = CONFIG.get("retrieval", {})
    bm25_top_k = int(retrieval_config.get("bm25_top_k", 30))
    dense_top_k = int(retrieval_config.get("dense_top_k", 10))
    use_dense = dense_enabled() and index.get("dense_retriever") is not None

    keywords = coerce_string_list(
        query_plan.get("keywords"),
        fallback=[question] if include_original else [],
        limit=20,
    )
    subqueries = build_dense_queries(question, query_plan, include_original=include_original) if use_dense else []

    bm25_results = annotate_retrieval_results(
        index["bm25_retriever"].search(keywords, top_k=bm25_top_k),
        retriever_name=f"iter{iteration}:bm25",
        query=keywords,
    )

    dense_result_lists = []
    if use_dense:
        for idx, subquery in enumerate(subqueries):
            label = dense_label(iteration, idx, subquery, question, include_original, dense_label_prefix)
            dense_result_lists.append(
                index["dense_retriever"].search(subquery, top_k=dense_top_k, label=label)
            )

    log_block(
        "Stage 2",
        f"Retrieval results (iteration {iteration})",
        "\n".join(
            [
                f"BM25 candidates: {len(bm25_results)}",
                f"Dense enabled: {use_dense}",
                f"Dense candidates: {sum(len(results) for results in dense_result_lists)}",
                f"Dense subqueries: {len(dense_result_lists)}",
                f"BM25 chunk_ids: {format_result_chunk_ids(bm25_results)}",
                f"Dense chunk_ids: {format_dense_result_chunk_ids(dense_result_lists)}",
            ]
        ),
    )

    return {
        "iteration": iteration,
        "query_plan": query_plan,
        "bm25_results": bm25_results,
        "dense_result_lists": dense_result_lists,
    }


def format_result_chunk_ids(results: list[dict[str, Any]]) -> str:
    chunk_ids = dedupe_strings(
        [result.get("chunk", {}).get("chunk_id", "") for result in results],
        limit=50,
    )
    return ", ".join(chunk_ids) if chunk_ids else "(none)"


def format_dense_result_chunk_ids(result_lists: list[list[dict[str, Any]]]) -> str:
    if not result_lists:
        return "(none)"

    lines = []
    for idx, results in enumerate(result_lists):
        query = next((result.get("query") for result in results if result.get("query")), "")
        chunk_ids = format_result_chunk_ids(results)
        label = f"subquery {idx + 1}"
        if query:
            label = f"{label} ({query})"
        lines.append(f"{label}: {chunk_ids}")
    return "\n" + "\n".join(lines)


def merge_accumulated_retrieval_results(
    retrieval_runs: list[dict[str, Any]],
    query_plan: dict,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    retrieval_config = CONFIG.get("retrieval", {})
    final_top_k = int(retrieval_config.get("final_top_k", top_k))

    bm25_results: list[dict[str, Any]] = []
    dense_result_lists: list[list[dict[str, Any]]] = []
    for run in retrieval_runs:
        bm25_results.extend(run["bm25_results"])
        dense_result_lists.extend(run["dense_result_lists"])

    merged_results = merge_retrieval_results(
        bm25_results=bm25_results,
        dense_result_lists=dense_result_lists,
        top_k=final_top_k,
        query_plan=query_plan,
        config=merge_config_from_dict(CONFIG),
    )
    sanitized_results, removed_count = sanitize_retrieval_results(merged_results)
    if removed_count:
        log_block(
            "Safety Scan",
            "Removed retrieved attack messages",
            f"Removed {removed_count} attack message block(s) before context generation.",
        )
    return sanitized_results


def build_dense_queries(question: str, query_plan: dict, *, include_original: bool) -> list[str]:
    subqueries = coerce_string_list(
        query_plan.get("subqueries"),
        fallback=[],
        limit=10,
    )
    if include_original:
        subqueries = [question, *subqueries]
    return dedupe_strings(subqueries)


def dense_label(
    iteration: int,
    idx: int,
    subquery: str,
    question: str,
    include_original: bool,
    dense_label_prefix: str,
) -> str:
    if include_original and idx == 0 and subquery == question:
        label = "original"
    else:
        label = f"{dense_label_prefix}{idx}"
    return f"iter{iteration}:dense:{label}"


def annotate_retrieval_results(
    results: list[dict[str, Any]],
    *,
    retriever_name: str,
    query: str | list[str],
) -> list[dict[str, Any]]:
    annotated = []
    for result in results:
        item = result.copy()
        item["retriever_name"] = retriever_name
        item.setdefault("query", query)
        annotated.append(item)
    return annotated


def format_context(results: list[dict]) -> str:
    parts = []
    for idx, item in enumerate(results, start=1):
        chunk = item["chunk"]
        source = chunk.get("source", "")
        page = chunk.get("page", "")
        section = chunk.get("section", "")
        retriever = item.get("retriever", "")
        score = item.get("score", 0)
        provenance = format_provenance(item.get("retrieved_from", {}))
        header = (
            f"[{idx}] source={source} page={page} section={section} "
            f"retriever={retriever} score={score:.4f}"
        )
        if provenance:
            header = f"{header}\nretrieval={provenance}"
        parts.append(f"{header}\n{chunk.get('text', '')}")
    return "\n\n---\n\n".join(parts)


def format_provenance(retrieved_from: dict) -> str:
    parts = []
    for name, info in sorted(retrieved_from.items()):
        parts.append(f"{name} rank={info.get('rank')} score={float(info.get('score', 0)):.4f}")
    return "; ".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONLINE STAGE 3-1. Context sufficiency evaluation + iterative retrieval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def retrieve_iterative(question: str, query_plan: dict, index) -> tuple[str, dict]:
    """Retrieve, evaluate evidence sufficiency, and optionally retrieve missing evidence."""
    iterative_config = CONFIG.get("retrieval", {}).get("iterative", {})
    iterative_enabled = bool(iterative_config.get("enabled", True))
    max_iterations = int(iterative_config.get("max_iterations", 0)) if iterative_enabled else 0

    accumulated_query_plan = copy_query_plan(query_plan)
    retrieval_runs = [
        retrieve_once(
            question=question,
            query_plan=query_plan,
            index=index,
            iteration=0,
            include_original=True,
            dense_label_prefix="sq",
        )
    ]
    merged_results = merge_accumulated_retrieval_results(retrieval_runs, accumulated_query_plan)
    context = format_context(merged_results)

    for evaluation_iteration in range(max_iterations + 1):
        evaluation = evaluate_context_sufficiency(
            question=question,
            context=context,
            query_plan=accumulated_query_plan,
            iteration=evaluation_iteration,
        )
        if evaluation["containing_answer"] == "yes":
            break
        if evaluation_iteration >= max_iterations:
            log_block(
                "Stage 3-1",
                "Iteration stop",
                "Max iterative retrieval rounds reached. Proceeding with current context.",
            )
            break

        followup_query_plan = remove_existing_query_terms(
            query_plan_from_context_evaluation(evaluation),
            accumulated_query_plan,
        )
        if not has_retrieval_terms(followup_query_plan):
            log_block(
                "Stage 3-1",
                "Iteration stop",
                "No new follow-up retrieval terms generated. Proceeding with current context.",
            )
            break

        next_iteration = evaluation_iteration + 1
        accumulated_query_plan = merge_query_plans(accumulated_query_plan, followup_query_plan)
        retrieval_runs.append(
            retrieve_once(
                question=question,
                query_plan=followup_query_plan,
                index=index,
                iteration=next_iteration,
                include_original=False,
                dense_label_prefix="missing",
            )
        )
        merged_results = merge_accumulated_retrieval_results(retrieval_runs, accumulated_query_plan)
        context = format_context(merged_results)

    return context, accumulated_query_plan


def evaluate_context_sufficiency(
    *,
    question: str,
    context: str,
    query_plan: dict,
    iteration: int,
) -> dict[str, Any]:
    config = llm_config("context_evaluation")
    if not config["enabled"]:
        evaluation = {
            "containing_answer": "yes",
            "reason": "context_evaluation stage is disabled",
            "missing_keywords": [],
            "subqueries": [],
        }
        log_context_evaluation(question, context, iteration, evaluation)
        return evaluation

    content = call_solar_no_record(
        system_prompt=CONTEXT_EVALUATION_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"[User question]\n{question}\n\n"
                    f"[Query analysis]\n{json.dumps(query_plan, ensure_ascii=False)}\n\n"
                    f"[Retrieved context]\n{context}"
                ),
            }
        ],
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    try:
        parsed = json.loads(extract_json_object(content))
        evaluation = normalize_context_evaluation(parsed)
    except (json.JSONDecodeError, ValueError):
        evaluation = {
            "containing_answer": "yes",
            "reason": "context evaluation failed to parse JSON; using current context",
            "missing_keywords": [],
            "subqueries": [],
        }

    log_context_evaluation(question, context, iteration, evaluation)
    return evaluation


def normalize_context_evaluation(parsed: dict) -> dict[str, Any]:
    containing_answer = str(parsed.get("containing_answer", "")).strip().lower()
    if containing_answer not in {"yes", "no"}:
        containing_answer = "no" if parsed.get("missing_keywords") or parsed.get("subqueries") else "yes"

    missing_keywords = coerce_string_list(
        parsed.get("missing_keywords", parsed.get("keywords")),
        fallback=[],
        limit=10,
    )
    subqueries = coerce_string_list(
        parsed.get("subqueries", parsed.get("missing_subqueries")),
        fallback=[],
        limit=3,
    )
    if containing_answer == "yes":
        missing_keywords = []
        subqueries = []

    return {
        "containing_answer": containing_answer,
        "reason": str(parsed.get("reason", "")).strip(),
        "missing_keywords": missing_keywords,
        "subqueries": subqueries,
    }


def log_context_evaluation(question: str, context: str, iteration: int, evaluation: dict[str, Any]) -> None:
    log_block(
        "Stage 3-1",
        f"Context sufficiency evaluation (iteration {iteration})",
        "\n".join(
            [
                f"Question: {question}",
                f"Retrieved context chars: {len(context)}",
                json.dumps(evaluation, ensure_ascii=False, indent=2),
            ]
        ),
    )


def query_plan_from_context_evaluation(evaluation: dict[str, Any]) -> dict:
    keywords = coerce_string_list(evaluation.get("missing_keywords"), fallback=[], limit=10)
    subqueries = coerce_string_list(evaluation.get("subqueries"), fallback=[], limit=3)
    if keywords and not subqueries:
        subqueries = [" ".join(keywords)]
    if subqueries and not keywords:
        keywords = subqueries
    return {
        "keywords": keywords,
        "subqueries": subqueries,
    }


def merge_query_plans(base: dict, followup: dict) -> dict:
    return {
        "keywords": dedupe_strings(
            [
                *(base.get("keywords") or []),
                *(followup.get("keywords") or []),
            ],
            limit=30,
        ),
        "subqueries": dedupe_strings(
            [
                *(base.get("subqueries") or []),
                *(followup.get("subqueries") or []),
            ],
            limit=15,
        ),
    }


def remove_existing_query_terms(candidate: dict, existing: dict) -> dict:
    existing_keywords = {str(item).strip() for item in existing.get("keywords") or []}
    existing_subqueries = {str(item).strip() for item in existing.get("subqueries") or []}
    return {
        "keywords": [
            keyword
            for keyword in candidate.get("keywords") or []
            if keyword not in existing_keywords
        ],
        "subqueries": [
            subquery
            for subquery in candidate.get("subqueries") or []
            if subquery not in existing_subqueries
        ],
    }


def copy_query_plan(query_plan: dict) -> dict:
    return {
        "keywords": list(query_plan.get("keywords") or []),
        "subqueries": list(query_plan.get("subqueries") or []),
    }


def has_retrieval_terms(query_plan: dict) -> bool:
    return bool(query_plan.get("keywords") or query_plan.get("subqueries"))


def dedupe_strings(values, *, limit: int | None = None) -> list[str]:
    cleaned = []
    seen = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
        if limit is not None and len(cleaned) >= limit:
            break
    return cleaned


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONLINE STAGE 3-2. Draft generation  (LLM, CSV 기록 없음)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_draft_answer(question: str, context: str, query_plan: dict) -> str:
    config = llm_config("draft_generation")
    if not config["enabled"]:
        return ""

    draft = call_solar_no_record(
        system_prompt=DRAFT_GENERATION_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"[User question]\n{question}\n\n"
                    f"[Query analysis]\n{json.dumps(query_plan, ensure_ascii=False)}\n\n"
                    f"[Retrieved context]\n{context}"
                ),
            }
        ],
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    log_block("Stage 3-2", "Draft answer", draft)
    return draft


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ONLINE STAGE 4. Final safety rewrite/generation  (tracker.chat, CSV 기록)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def finalize_answer(
    *,
    question: str,
    context: str,
    query_plan: dict,
    draft_answer: str,
    tracker: UpstageTracker,
    question_id: str,
    token: str,
) -> str:
    config = llm_config("final_generation")
    messages = [
        {
            "role": "user",
            "content": (
                f"[User question]\n{question}\n\n"
                f"[Draft answer]\n{draft_answer}"
            ),
        }
    ]

    return tracker.chat(
        question_id=question_id,
        messages=messages,
        token=token,
        model=config["model"],
        system_prompt=FINAL_SAFETY_PROMPT,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(output_path: str = "submission.csv") -> None:
    # Phase 1: 인덱스 구축 (1회)
    log_block("Pipeline", "Index build start", "Building parse artifacts, chunks, and retrieval indexes.")
    index = build_index(CORPUS_DIR)

    # 질문 로드
    log_block("Pipeline", "Question load start", f"Test suite: {TEST_SUITE_PATH}")
    questions = load_test_suite(path=TEST_SUITE_PATH)
    log_block("Pipeline", "Question load completed", f"Questions: {len(questions)}")

    # Online stages: query analysis → retrieval → draft → final safety rewrite
    log_block("Pipeline", "Online stages start", "Running query analysis, retrieval, generation, and final safety rewrite.")
    tracker = UpstageTracker()

    for i, q in enumerate(questions):
        log_block("Question", f"{i+1}/{len(questions)}", q["question"], leading_newlines=2)
        query_plan = analyze_query(q["question"])
        context, generation_query_plan = retrieve_iterative(q["question"], query_plan, index)
        draft_answer = generate_draft_answer(
            question=q["question"],
            context=context,
            query_plan=generation_query_plan,
        )
        answer = finalize_answer(
            question=q["question"],
            context=context,
            query_plan=generation_query_plan,
            draft_answer=draft_answer,
            tracker=tracker,
            question_id=q["question_id"],
            token=q["token"],
        )
        answer_one_line = answer.replace("\n", " ")
        log_block(
            "Stage 4",
            "Final answer",
            f"Question: {q['question']}\nAnswer: {answer_one_line}",
        )

    # 저장 + 검증
    tracker.save_csv(output_path)
    validate(output_path)


if __name__ == "__main__":
    import sys, io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8")
    run_pipeline()
