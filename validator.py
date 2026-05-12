"""
validator.py — submission.csv 제출 파일 사전 검증기

제출 전 반드시 실행하여 스키마 오류를 사전에 확인하세요.

사용법:
    python validator.py                        # 기본 경로 (submission.csv)
    python validator.py path/to/my_result.csv  # 직접 경로 지정
"""

import sys
import os
import pandas as pd

REQUIRED_COLUMNS = {"question_id", "answer", "used_tokens", "inference_time", "token"}
QID_PATTERN = r"^Q_\d{3,}$"


def validate(path: str = "submission.csv") -> bool:
    """submission.csv 규격을 검증합니다.

    Args:
        path: 검증할 CSV 파일 경로

    Returns:
        True  — 모든 검사 통과 (제출 가능)
        False — 하나 이상의 검사 실패
    """
    errors = []
    warnings = []

    # ── 1. 파일 존재 여부 ────────────────────────────────────────────────
    if not os.path.exists(path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {path}")
        return False

    # ── 2. UTF-8 인코딩 로드 ─────────────────────────────────────────────
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        errors.append("파일 인코딩이 UTF-8이 아닙니다. (저장 시 UTF-8로 내보내기 필요)")
        _print_result(errors, warnings)
        return False
    except Exception as e:
        errors.append(f"CSV 파싱 실패: {e}")
        _print_result(errors, warnings)
        return False

    # ── 3. 필수 컬럼 존재 여부 ───────────────────────────────────────────
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"필수 컬럼 누락: {sorted(missing_cols)}")

    # 이하 검사는 필수 컬럼이 모두 있을 때만 의미 있음
    if errors:
        _print_result(errors, warnings)
        return False

    # ── 4. question_id 검사 ──────────────────────────────────────────────
    qid_series = df["question_id"].astype(str)
    invalid_ids = qid_series[~qid_series.str.match(QID_PATTERN)].unique().tolist()
    if invalid_ids:
        errors.append(f"유효하지 않은 question_id 형식 ({len(invalid_ids)}개): {invalid_ids[:5]}")

    duplicate_ids = qid_series[qid_series.duplicated()].tolist()
    if duplicate_ids:
        errors.append(f"question_id 중복: {duplicate_ids}")

    # ── 5. 빈 값 검사 (answer, token) ────────────────────────────────────
    for col in ("answer", "token"):
        empty_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        empty_ids = df.loc[empty_mask, "question_id"].tolist()
        if empty_ids:
            errors.append(f"'{col}' 빈 값 발견 ({len(empty_ids)}개): {empty_ids[:5]}")

    # ── 6. 데이터 타입 검사 ──────────────────────────────────────────────
    try:
        df["used_tokens"].astype(int)
    except (ValueError, TypeError):
        errors.append("'used_tokens' 컬럼에 정수로 변환 불가한 값이 있습니다.")

    try:
        df["inference_time"].astype(float)
    except (ValueError, TypeError):
        errors.append("'inference_time' 컬럼에 실수로 변환 불가한 값이 있습니다.")

    # ── 7. 경고 (오류는 아니지만 확인 권장) ─────────────────────────────
    zero_token_rows = df[df["used_tokens"].astype(float) == 0]["question_id"].tolist()
    if zero_token_rows:
        warnings.append(
            f"'used_tokens'가 0인 항목 ({len(zero_token_rows)}개): {zero_token_rows[:5]} "
            "— UpstageTracker가 정상 연결되었는지 확인하세요."
        )

    median_time = df["inference_time"].astype(float).median()
    if median_time > 15:
        warnings.append(f"중간값 응답 시간 {median_time:.1f}초 — 30% 감점 구간입니다.")
    elif median_time > 7:
        warnings.append(f"중간값 응답 시간 {median_time:.1f}초 — 15% 감점 구간입니다.")
    elif median_time > 3:
        warnings.append(f"중간값 응답 시간 {median_time:.1f}초 — 5% 감점 구간입니다.")

    _print_result(errors, warnings, df_len=len(df))
    return len(errors) == 0


def _print_result(errors: list, warnings: list, df_len: int = None) -> None:
    print("=" * 55)
    print("  submission.csv 검증 결과")
    print("=" * 55)

    if df_len is not None:
        print(f"  총 행 수: {df_len}개\n")

    if errors:
        print(f"  [FAIL] 오류 {len(errors)}건")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print("  [PASS] 스키마 검사 통과 — 제출 가능합니다.")

    if warnings:
        print(f"\n  [WARN] 경고 {len(warnings)}건")
        for w in warnings:
            print(f"    △ {w}")

    print("=" * 55)


if __name__ == "__main__":
    import io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8")
    target = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    ok = validate(target)
    sys.exit(0 if ok else 1)
