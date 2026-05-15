from __future__ import annotations

import re
from pathlib import Path
import pdfplumber

# =========================================================
# Hidden text detection config
# =========================================================
WHITE_THRESHOLD_FLOAT = 0.95
WHITE_THRESHOLD_INT = 240
TINY_FONT_THRESHOLD = 2.0


# =========================================================
# Instruction-like text patterns
# =========================================================
INSTRUCTION_PATTERNS = [
    # Explicit Jailbreak & System Overrides
    r"(?i)ignore\s+.*previous\s+instructions?",
    r"(?i)disregard\s+.*previous\s+instructions?",
    r"(?i)system\s*(prompt|note|message)",
    r"(?i)developer\s*(prompt|note|message)",
    r"(?i)override\s+all\s+rules",

    r"(?i)you\s+are\s+now",
    r"(?i)assistant\s*:",
    r"(?i)AI\s+response",
    r"(?i)language\s+model",
    
    r"(?i)must\s+(append|include|output|print|say|respond)",
    r"(?i)output\s+exactly",
    r"(?i)end\s+your\s+response\s+with",
    r"(?i)respond\s+with",
    r"(?i)APPROVED_BY_ADMIN",
    r"(?i)APPROVED[-_]BY[-_]ADMIN",
    
    r"(?i)verification\s+token",
    r"(?i)required\s+token",
    r"(?i)security\s+override",
]

# =========================================================
# Utility functions
# =========================================================

def _is_near_white(color) -> bool:
    if color is None:
        return False

    if isinstance(color, (int, float)):
        if isinstance(color, int):
            return color >= WHITE_THRESHOLD_INT
        return float(color) >= WHITE_THRESHOLD_FLOAT

    if isinstance(color, (list, tuple)):
        if len(color) == 3:  
            if isinstance(color[0], int):
                return all(c >= WHITE_THRESHOLD_INT for c in color)
            return all(c >= WHITE_THRESHOLD_FLOAT for c in color)

        if len(color) == 4:
            c, m, y, k = color
            return c < 0.05 and m < 0.05 and y < 0.05 and k < 0.05

    return False


def _has_dark_background(char: dict, rects: list[dict]) -> bool:
    """
    글자가 어두운 배경 사각형 내부나 위에 얹혀 있는지 체크하여 흰색 글씨의 오탐 방지
    """
    cx = (char["x0"] + char["x1"]) / 2
    cy = (char["top"] + char["bottom"]) / 2

    for rect in rects:
        if rect["x0"] <= cx <= rect["x1"]:
            if rect["top"] <= cy <= rect["bottom"]:
                bg = rect.get("non_stroking_color")
                if bg is not None and not _is_near_white(bg):
                    return True
    return False


# =========================================================
# Hidden text scanning
# =========================================================

def _scan_page(page, source: str) -> list[dict]:
    findings = []
    x0, top, x1, bottom = page.bbox
    rects = page.rects

    page_text_buffer = []

    for char in page.chars:
        text = char.get("text", "")
        if not text.strip():
            continue
        
        page_text_buffer.append(text)
        reason = None
        color = char.get("non_stroking_color")
        size = char.get("size") or 99

        # 1. White-on-White (숨은 글씨 공격)
        if _is_near_white(color):
            if not _has_dark_background(char, rects):
                reason = f"hidden_white_text(color={color})"

        # 2. Microscopic Text (나노 글씨 공격)
        elif size < TINY_FONT_THRESHOLD:
            reason = f"microscopic_text(size={round(size, 2)})"

        # 3. Out-of-bounds (캔버스 바깥 영역 배치 공격)
        elif not (x0 <= char["x0"] <= x1 and top <= char["top"] <= bottom):
            reason = f"out_of_bounds_text(pos=[{round(char['x0'])},{round(char['top'])}])"

        if reason:
            findings.append({
                "source": source,
                "page": page.page_number,
                "reason": reason,
                "text": text.strip(),
            })

    return findings


def scan_pdf_hidden_text(path: Path) -> list[dict]:
    findings = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                findings.extend(_scan_page(page, path.name))
    except Exception as e:
        print(f"[sanitizer error] Failed to scan geometry for {path.name}: {e}")
    return findings


# =========================================================
# Instruction pattern detection
# =========================================================

def detect_instruction_patterns(text: str) -> dict:
    """
    문서 내 텍스트에서 프롬프트 인젝션 의심 지시 패턴을 탐지합니다.
    """
    matched_patterns = []

    for pattern in INSTRUCTION_PATTERNS:
        if re.search(pattern, text):
            matched_patterns.append(pattern)

    suspicious = len(matched_patterns) > 0

    return {
        "suspicious": suspicious,
        "score": len(matched_patterns),
        "reasons": matched_patterns,
    }

# =========================================================
# Corpus-wide security map
# =========================================================

def build_security_map(corpus_dir: str | Path) -> dict:
    """
    전체 Corpus 폴더를 미리 싹 스캔해서 메모리에 보안 맵을 구축합니다.
    결과 구조: security_map[파일명][페이지번호] = { ...보안 데이터... }
    """
    corpus_path = Path(corpus_dir)
    security_map = {}

    if not corpus_path.exists():
        print(f"[sanitizer warning] Corpus directory {corpus_dir} does not exist.")
        return security_map

    pdf_files = sorted(corpus_path.glob("*.pdf"))
    if not pdf_files:
        print(f"[sanitizer warning] No PDF files found in {corpus_dir}")
        return security_map

    for pdf_path in pdf_files:
        hidden_findings = scan_pdf_hidden_text(pdf_path)
        
        hidden_by_page = {}
        for finding in hidden_findings:
            hidden_by_page.setdefault(finding["page"], []).append(finding)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_num = page.page_number
                    text = page.extract_text() or ""

                    instruction_result = detect_instruction_patterns(text)
                    hidden_page_findings = hidden_by_page.get(page_num, [])

                    reasons = []
                    if hidden_page_findings:
                        reasons.append("hidden_text_attack")
                    if instruction_result["suspicious"]:
                        reasons.append("prompt_injection_pattern")

                    suspicious = bool(reasons)

                    security_map.setdefault(pdf_path.name, {})[page_num] = {
                        "suspicious": suspicious,
                        "score": len(hidden_page_findings) + (instruction_result["score"] * 5), # 인젝션 문구 발견 시 가중치 부여
                        "reasons": reasons,
                        "hidden_findings_count": len(hidden_page_findings),
                        "instruction_patterns": instruction_result["reasons"],
                    }
            
            suspicious_pages = sum(
                1 for p_data in security_map[pdf_path.name].values() if p_data["suspicious"]
            )
            if suspicious_pages > 0:
                print(f"⚠️ [Security Scan] '{pdf_path.name}' -> Found {suspicious_pages} suspicious pages.")
        
        except Exception as e:
            print(f"[sanitizer error] Failed to process security map for {pdf_path.name}: {e}")

    return security_map


# =========================================================
# Chunk-level lookup & Sanitization
# =========================================================

def lookup_chunk_security(chunk: dict, security_map: dict) -> dict:
    """
    데이터 가공 파이프라인에서 각 chunk 엘리먼트에 보안 필드를 채워줍니다.
    """
    source = chunk.get("source")
    page = chunk.get("page") or chunk.get("page_number")

    if source and ("/" in source or "\\" in source):
        source = Path(source).name

    if not source or not page:
        return {"suspicious": False, "score": 0, "reasons": [], "hidden_findings_count": 0, "instruction_patterns": []}

    return security_map.get(source, {}).get(
        int(page),
        {"suspicious": False, "score": 0, "reasons": [], "hidden_findings_count": 0, "instruction_patterns": []}
    )