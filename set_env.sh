#!/usr/bin/env bash
# 환경변수 설정 스크립트 (참가자용)
#
# 반드시 source 명령으로 실행해야 현재 셸에 변수가 유지됩니다:
#   source set_env.sh
#   . set_env.sh          (동일)
#
# 인자로 직접 전달할 수도 있습니다:
#   source set_env.sh <HACKATHON_KEY> <UPSTAGE_API_KEY>

# ── source 여부 감지 ─────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[오류] 이 스크립트는 반드시 source 로 실행해야 합니다."
    echo "  사용법: source set_env.sh"
    exit 1
fi

# ── HACKATHON_KEY ────────────────────────────────────────────────────────
if [[ -n "$1" ]]; then
    HACKATHON_KEY="$1"
else
    read -r -p "HACKATHON_KEY 입력 (대회 당일 공지): " HACKATHON_KEY
fi

# ── UPSTAGE_API_KEY ──────────────────────────────────────────────────────
if [[ -n "$2" ]]; then
    UPSTAGE_API_KEY_INPUT="$2"
elif [[ -n "$UPSTAGE_API_KEY" ]]; then
    echo "UPSTAGE_API_KEY: 기존 환경변수를 그대로 사용합니다."
    UPSTAGE_API_KEY_INPUT="$UPSTAGE_API_KEY"
else
    read -r -p "UPSTAGE_API_KEY 입력: " UPSTAGE_API_KEY_INPUT
fi

# ── 현재 세션에 즉시 적용 ────────────────────────────────────────────────
export HACKATHON_KEY="$HACKATHON_KEY"
export UPSTAGE_API_KEY="$UPSTAGE_API_KEY_INPUT"
export PYTHONUTF8=1

# ── Shell profile에 영구 저장 ────────────────────────────────────────────
if [[ -n "$ZSH_VERSION" ]]; then
    PROFILE="$HOME/.zshrc"
else
    PROFILE="$HOME/.bashrc"
fi

_upsert_env() {
    local key="$1" val="$2"
    local tmp
    tmp=$(mktemp)
    grep -v "^export ${key}=" "$PROFILE" > "$tmp" 2>/dev/null && mv "$tmp" "$PROFILE" || true
    echo "export ${key}=\"${val}\"" >> "$PROFILE"
}

_upsert_env "HACKATHON_KEY"   "$HACKATHON_KEY"
_upsert_env "UPSTAGE_API_KEY" "$UPSTAGE_API_KEY_INPUT"
_upsert_env "PYTHONUTF8"      "1"

echo ""
echo "환경변수 설정 완료:"
echo "  HACKATHON_KEY    = ${HACKATHON_KEY:0:4}****"
echo "  UPSTAGE_API_KEY  = ${UPSTAGE_API_KEY:0:4}****"
echo "  영구 저장 → $PROFILE"
echo ""
echo "이제 python baseline_rag.py 를 실행할 수 있습니다."
