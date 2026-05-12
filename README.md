# Starter Kit — 운영 가이드

## 시작하기

### 1. 코드 가져오기

이 저장소를 로컬 환경에 복제합니다.

```bash
git clone https://github.com/UAI-2026-Spring-Hackathon/tech-starterkit
cd tech-starterkit
```

## 폴더 구조

```
starter_kit/
├── README.md                # 이 문서
├── Dockerfile               # 권장 실행 환경
├── .dockerignore
├── baseline_rag.py          # RAG 파이프라인 스켈레톤 (여기에 구현)
├── decryptor.py             # 암호화된 테스트 셋 복호화
├── upstage_tracker.py       # Solar LLM 호출 및 submission.csv 생성
├── validator.py             # 제출 파일 사전 검증
├── requirements.txt         # 의존성 패키지
├── set_env.sh               # 환경변수 설정 스크립트 (Linux/Mac)
├── set_env.ps1              # 환경변수 설정 스크립트 (Windows)
└── distribution/
    ├── corpus/              # PDF 코퍼스 (대회 당일 주최 측 배포)
    └── test_suite/
        └── Encrypted_Test_Suite.json   # 암호화된 질문 셋 (대회 당일 배포)
```

> `distribution/corpus/`와 `distribution/test_suite/`의 파일은 대회 당일 주최 측이 배포합니다.

---

## 제공 데이터 안내

**corpus 및 테스트 질문은 모두 영어로 제공됩니다.**

개발 편의를 위해 `decryptor.py`에 포함된 샘플 더미 데이터는 한국어로 작성되어 있습니다.
대회 당일 실제 데이터로 교체되면 영어 문서와 영어 질문을 처리하게 됩니다.

---

## 환경 설정

### 1. Python 환경 생성 (Conda 권장)

```bash
conda create -n hackathon-rag python=3.12
conda activate hackathon-rag
pip install -r requirements.txt
```

### 2. API 키 설정

대회 중 두 개의 키가 필요합니다.

| 환경변수 | 용도 | 발급 시점 |
|---|---|---|
| `UPSTAGE_API_KEY` | Solar LLM 호출 | 대회 시작 전 발급 |
| `HACKATHON_KEY` | 테스트 셋 복호화 | 대회 당일 공지 |

> 💡 **API 크레딧 관련 안내**
> - **사전 개발 기간**: **개인 API Key**를 사용해야 합니다. 레퍼럴 이벤트를 통해 크레딧을 취득하실 수 있으며, 개발과정에서 크레딧이 부족한 경우 운영진에게 개인적으로 문의주시면 도움드리겠습니다.
> - **본행사 당일**: 대회 당일 평가에만 사용할 수 있는 **별도의 크레딧이 당일 배부**될 예정입니다.

제공된 스크립트를 사용하면 대화형으로 키를 입력하고 바로 환경변수로 설정됩니다.

```bash
# Linux / Mac — 반드시 source 로 실행 (bash set_env.sh 는 동작하지 않음)
source set_env.sh

# Windows PowerShell — 반드시 dot-sourcing 으로 실행
. .\set_env.ps1
```

실행하면 각 키를 순서대로 입력받으며, 현재 세션에 즉시 적용되고 **다음에도 유지되도록 영구 저장**됩니다.
- Linux/Mac: `~/.bashrc` 또는 `~/.zshrc` 에 추가
- Windows: 사용자 환경변수(레지스트리)에 저장

이미 `UPSTAGE_API_KEY`가 설정되어 있으면 재입력을 건너뜁니다.

> **개발 중**: `HACKATHON_KEY` 없이 실행하면 샘플 더미 질문 5개로 동작합니다.
> `UPSTAGE_API_KEY`는 개발 단계부터 필요합니다.

---

## Docker (권장 환경)

패키지 충돌 없이 Python 3.12 환경을 보장하려면 Docker를 사용할 수 있습니다.

> 추가 패키지(`pypdf`, `chromadb` 등)를 사용할 경우 `requirements.txt`에서 해당 줄의 주석을 해제한 뒤 이미지를 빌드하세요.

### 1단계 — 빌드 (대회 전 미리 준비)

키 없이 빌드할 수 있습니다. 패키지 설치만 수행하며 키는 필요하지 않습니다.

```bash
docker build -t hackathon-rag .
```

### 2단계 — 실행

#### 개발 중 (HACKATHON_KEY 없이)

`HACKATHON_KEY` 없이 실행하면 샘플 더미 질문 5개로 자동 동작합니다.
파이프라인 구현과 테스트는 대회 전에 미리 진행할 수 있습니다.

```bash
# Linux / Mac
docker run --rm \
  -e UPSTAGE_API_KEY=<your_key> \
  -v "$(pwd):/workspace" \
  hackathon-rag

# Windows PowerShell
docker run --rm `
  -e UPSTAGE_API_KEY=<your_key> `
  -v "${PWD}:/workspace" `
  hackathon-rag
```

#### 대회 당일 (HACKATHON_KEY 공지 후)

```bash
# Linux / Mac
docker run --rm \
  -e HACKATHON_KEY=<공지된_키> \
  -e UPSTAGE_API_KEY=<your_key> \
  -v "$(pwd):/workspace" \
  hackathon-rag

# Windows PowerShell
docker run --rm `
  -e HACKATHON_KEY=<공지된_키> `
  -e UPSTAGE_API_KEY=<your_key> `
  -v "${PWD}:/workspace" `
  hackathon-rag
```

`-v "$(pwd):/workspace"` 마운트로 코드 변경이 즉시 반영되며, 실행 후 `submission.csv`가 현재 디렉토리에 생성됩니다.

---

## 실행

```bash
# starter_kit/ 루트에서 실행
python baseline_rag.py
```

실행 후 `submission.csv`가 생성되며, 자동으로 `validator.py` 검증이 실행됩니다.

제출 전 별도로 검증하려면:

```bash
python validator.py                     # 기본값: submission.csv
python validator.py path/to/result.csv  # 직접 경로 지정
```

---

## 채점 기준

### 응답 시간 감점 구간 (중간값 기준)

| 중간값 응답 시간 | 감점 |
|---|---|
| 3초 이하 | 없음 |
| 3초 초과 ~ 7초 이하 | 5% |
| 7초 초과 ~ 15초 이하 | 15% |
| 15초 초과 | 30% |

### 기타 유의사항

- `used_tokens`가 0인 제출 항목은 채점에서 **제외**됩니다.
  Solar LLM(`tracker.chat()`)을 반드시 사용해야 합니다.
- `submission.csv`는 UTF-8 인코딩이어야 합니다.
- `question_id`는 중복 없이 `Q_XXX` 형식이어야 합니다.

---

## 파일별 역할

### `baseline_rag.py`

구현할 RAG 파이프라인의 뼈대입니다. 세 함수를 구현하세요.

| 함수 | 역할 | 상태 |
|---|---|---|
| `build_index(corpus_dir)` | PDF 파싱·청킹·인덱싱 | **TODO** |
| `retrieve(question, index)` | 질문 관련 청크 검색 | **TODO** |
| `generate_answer(...)` | Solar LLM 답변 생성 | 뼈대 제공 |

`SYSTEM_PROMPT`도 직접 설계해야 합니다. 할루시네이션 방지, Prompt Injection 방어, PII 유출 방지를 고려하세요.

### `decryptor.py`

암호화된 테스트 셋을 복호화합니다.
- `HACKATHON_KEY` 미설정 시 → 샘플 더미 5개 반환 (개발용, **한국어**)
- 대회 당일 키 설정 후 → 실제 암호화 질문 복호화 (**영어**)

### `upstage_tracker.py`

Solar LLM 호출을 추적하고 `submission.csv`를 자동 생성합니다.
답변 생성 시 반드시 `tracker.chat()`을 사용해야 합니다.

### `distribution/corpus/`

대회 PDF 코퍼스가 배치되는 폴더입니다. **실제 문서는 영어로 작성됩니다.**
일부 문서에는 의도적으로 삽입된 지시문(Prompt Injection)이 포함되어 있습니다.
시스템 프롬프트에서 이를 무시하도록 설계하세요.
