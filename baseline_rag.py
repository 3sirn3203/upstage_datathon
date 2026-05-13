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
import urllib.error
import urllib.request


from decryptor import load_test_suite
from upstage_tracker import DEFAULT_MODEL, UPSTAGE_BASE_URL, UpstageTracker
from src.chunk_corpus import chunk_corpus, config_from_dict
from src.index_corpus import build_dense_index, config_from_dict as dense_config_from_dict
from src.parse_corpus import DEFAULT_CONFIG_PATH, load_config, parse_corpus
from src.prompt import DRAFT_GENERATION_PROMPT, FINAL_SAFETY_PROMPT, QUERY_ANALYSIS_PROMPT
from src.retriever_bm25 import BM25Retriever, config_from_dict as bm25_config_from_dict
from src.retriever_dense import DenseRetriever
from src.retriever_merge import config_from_dict as merge_config_from_dict, merge_retrieval_results
from validator import validate

CORPUS_DIR      = "distribution/corpus"
TEST_SUITE_PATH = "distribution/test_suite/Encrypted_Test_Suite.json"
CONFIG          = load_config(DEFAULT_CONFIG_PATH)


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
        option=parsing_config.get("backend", "pdfplumber"),
        output_dir=parsing_config.get("output_dir", "parsed_corpus"),
        config_path=DEFAULT_CONFIG_PATH,
        force=parsing_config.get("force", False),
    )
    print(f"  → parsed corpus: {parsed_path}")

    chunks_path = chunk_corpus(
        parsed_path,
        config=config_from_dict(CONFIG),
    )
    chunks = load_chunks(chunks_path)
    print(f"  → chunks: {chunks_path} ({len(chunks)} chunks)")

    dense_index = build_dense_index(
        chunks_path,
        config=dense_config_from_dict(CONFIG),
    )
    print(f"  → dense index: {dense_index['faiss_path']} ({dense_index['num_vectors']} vectors)")

    bm25_retriever = BM25Retriever(chunks, config=bm25_config_from_dict(CONFIG))
    dense_retriever = DenseRetriever(
        faiss_path=dense_index["faiss_path"],
        metadata_path=dense_index["metadata_path"],
        config=dense_config_from_dict(CONFIG),
    )

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
        print(f"Stage 1: Query analysis output:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except (json.JSONDecodeError, ValueError):
        parsed = fallback
        print(f"Stage 1 - Query analysis failed to parse JSON. Using fallback:")

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
    """질문과 query plan을 바탕으로 관련 chunk context를 반환합니다.

    TODO:
    - BM25/dense 후보를 RRF 또는 weighted score로 merge
    - source/page/section 중복 제거 및 context budget 적용
    """
    retrieval_config = CONFIG.get("retrieval", {})
    bm25_top_k = int(retrieval_config.get("bm25_top_k", 30))
    dense_top_k = int(retrieval_config.get("dense_top_k", 10))
    final_top_k = int(retrieval_config.get("final_top_k", top_k))

    keywords = query_plan.get("keywords") or [question]
    subqueries = [question, *(query_plan.get("subqueries") or [])]

    bm25_results = index["bm25_retriever"].search(keywords, top_k=bm25_top_k)
    dense_result_lists = index["dense_retriever"].search_many(subqueries, top_k=dense_top_k)

    print(
        "  [Retrieval] "
        f"BM25={len(bm25_results)} candidates | "
        f"Dense={sum(len(results) for results in dense_result_lists)} candidates "
        f"from {len(dense_result_lists)} subqueries"
    )

    merged_results = merge_retrieval_results(
        bm25_results=bm25_results,
        dense_result_lists=dense_result_lists,
        top_k=final_top_k,
        query_plan=query_plan,
        config=merge_config_from_dict(CONFIG),
    )
    return format_context(merged_results)


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
# ONLINE STAGE 3. Draft generation  (LLM, CSV 기록 없음)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_draft_answer(question: str, context: str, query_plan: dict) -> str:
    config = llm_config("draft_generation")
    if not config["enabled"]:
        return ""

    return call_solar_no_record(
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
                f"[Query analysis]\n{json.dumps(query_plan, ensure_ascii=False)}\n\n"
                f"[Retrieved context]\n{context}\n\n"
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
    print("[1/3] 인덱스 구축 중...")
    corpus_dir = CONFIG.get("corpus", {}).get("dir", CORPUS_DIR)
    index = build_index(corpus_dir)

    # 질문 로드
    print("[2/3] 질문 로드 중...")
    questions = load_test_suite(path=TEST_SUITE_PATH)
    print(f"  → {len(questions)}개 질문\n")

    # Online stages: query analysis → retrieval → draft → final safety rewrite
    print("[3/3] 파이프라인 실행 중...")
    tracker = UpstageTracker()

    for i, q in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}: {q['question']}")
        query_plan = analyze_query(q["question"])
        context = retrieve(q["question"], query_plan, index)
        draft_answer = generate_draft_answer(
            question=q["question"],
            context=context,
            query_plan=query_plan,
        )
        answer = finalize_answer(
            question=q["question"],
            context=context,
            query_plan=query_plan,
            draft_answer=draft_answer,
            tracker=tracker,
            question_id=q["question_id"],
            token=q["token"],
        )
        print(f"Question: {q['question']}")
        print(f"Answer: {answer.strip()}")
        print("-" * 80)
        print()

    # 저장 + 검증
    print()
    tracker.save_csv(output_path)
    print()
    validate(output_path)


if __name__ == "__main__":
    import sys, io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8")
    run_pipeline()
