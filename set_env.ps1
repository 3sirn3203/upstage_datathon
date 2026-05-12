# 환경변수 설정 스크립트 (참가자용)
#
# 반드시 dot-sourcing 으로 실행해야 현재 셸에 변수가 유지됩니다:
#   . .\set_env.ps1
#
# 인자로 직접 전달할 수도 있습니다:
#   . .\set_env.ps1 -HackathonKey <KEY> -UpstageApiKey <KEY>

param(
    [string]$HackathonKey   = "",
    [string]$UpstageApiKey  = ""
)

# ── HACKATHON_KEY ────────────────────────────────────────────────────────
if (-not $HackathonKey) {
    $HackathonKey = Read-Host "HACKATHON_KEY 입력 (대회 당일 공지)"
}

# ── UPSTAGE_API_KEY ──────────────────────────────────────────────────────
if (-not $UpstageApiKey) {
    if ($env:UPSTAGE_API_KEY) {
        Write-Host "UPSTAGE_API_KEY: 기존 환경변수를 그대로 사용합니다."
        $UpstageApiKey = $env:UPSTAGE_API_KEY
    } else {
        $UpstageApiKey = Read-Host "UPSTAGE_API_KEY 입력"
    }
}

# ── 현재 세션에 즉시 적용 ────────────────────────────────────────────────
$env:HACKATHON_KEY   = $HackathonKey
$env:UPSTAGE_API_KEY = $UpstageApiKey
$env:PYTHONUTF8      = "1"

# ── Windows 사용자 환경변수에 영구 저장 ──────────────────────────────────
[Environment]::SetEnvironmentVariable("HACKATHON_KEY",   $HackathonKey, "User")
[Environment]::SetEnvironmentVariable("UPSTAGE_API_KEY", $UpstageApiKey, "User")
[Environment]::SetEnvironmentVariable("PYTHONUTF8",      "1",           "User")

$maskedHackathon = $HackathonKey.Substring(0, [Math]::Min(4, $HackathonKey.Length)) + "****"
$maskedUpstage   = $UpstageApiKey.Substring(0, [Math]::Min(4, $UpstageApiKey.Length)) + "****"

Write-Host ""
Write-Host "환경변수 설정 완료:"
Write-Host "  HACKATHON_KEY    = $maskedHackathon"
Write-Host "  UPSTAGE_API_KEY  = $maskedUpstage"
Write-Host "  영구 저장 → Windows 사용자 환경변수 (시스템 속성에서 확인 가능)"
Write-Host ""
Write-Host "이제 python baseline_rag.py 를 실행할 수 있습니다."
