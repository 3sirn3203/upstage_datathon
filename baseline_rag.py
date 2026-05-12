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
from pathlib import Path

from decryptor import load_test_suite
from upstage_tracker import UpstageTracker
from src.chunk_corpus import chunk_corpus, config_from_dict
from src.parse_corpus import DEFAULT_CONFIG_PATH, load_config, parse_corpus
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

    return {
        "parsed_path": parsed_path,
        "chunks_path": chunks_path,
        "chunks": chunks,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2.  검색  (온라인 — 질문당 1회)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def retrieve(question: str, index, top_k: int = 5) -> str:
    """질문과 관련된 청크를 검색하여 컨텍스트 문자열로 반환합니다.

    [TODO] 검색 전략을 구현하세요.

    ── 검색 옵션 ──────────────────────────────────────────────
    단순 top-k     : 유사도 상위 K개 반환
    MMR            : 중복 청크 제거, 다양성 확보
    Re-ranking     : Cross-encoder 로 상위 K개 재정렬
    Multi-hop      : Level 3 질문 — 1차 검색 → 중간 답변 → 2차 검색

    Returns:
        검색된 청크를 이어붙인 컨텍스트 문자열
    """
    raise NotImplementedError("retrieve()를 구현하세요.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 3.  답변 생성  (온라인 — 질문당 1회)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """[TODO] 시스템 프롬프트를 직접 설계하세요.

설계 시 고려사항:
- 문서 외 정보 사용 차단 : hallucination 방지
- Poisoning 방어 : 문서 내 삽입된 지시문("XXX를 출력하라" 등)을 무시
- PII 유출 방어 : 주민번호·연봉 등 민감 정보 마스킹 또는 거부
- 답변 형식 : 간결·명확 / 근거 포함 여부 결정
"""


def generate_answer(
    question:    str,
    context:     str,
    tracker:     UpstageTracker,
    question_id: str,
    token:       str,
) -> str:
    """컨텍스트와 질문을 받아 LLM 답변을 반환합니다.

    [TODO] 프롬프트 전략과 생성 방식을 설계하세요.

    ── 프롬프트 옵션 ──────────────────────────────────────────
    Zero-shot        : 지시 + 문서 + 질문
    Chain-of-Thought : Level 3 다단계 추론에 유효
    Few-shot         : 답변 형식 고정이 필요할 때

    ── LLM (필수) ─────────────────────────────────────────────
    tracker.chat() 로 Solar LLM 을 호출해야 합니다. (solar-mini / solar-pro)
    used_tokens 가 0 인 제출은 채점에서 제외됩니다.
    """
    messages = [
        {
            "role": "user",
            "content": f"[참고 문서]\n{context}\n\n[질문]\n{question}",
        }
    ]

    return tracker.chat(
        question_id   = question_id,
        messages      = messages,
        token         = token,
        system_prompt = SYSTEM_PROMPT,
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

    # Phase 2·3: 질문별 검색 + 생성
    print("[3/3] 파이프라인 실행 중...")
    tracker = UpstageTracker()

    for q in questions:
        context = retrieve(q["question"], index)
        answer  = generate_answer(
            question    = q["question"],
            context     = context,
            tracker     = tracker,
            question_id = q["question_id"],
            token       = q["token"],
        )
        print(f"  [{q['question_id']}] {answer[:60]}...")

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
