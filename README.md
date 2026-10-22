# Policy Agent

규정 문서를 검색하고 신청 자격·서류·절차를 근거와 함께 안내하는 FastAPI + LangGraph + MCP + Vue 서비스입니다. 초기 범위는 학사 규정이며, 같은 인덱스 구조로 사내 규정도 검색할 수 있습니다.

**이 저장소에는 실제 규정 문서가 없습니다.** 문서를 직접 등록하기 전에는 신청 가능 여부를 판단하지 않습니다. `tests/fixtures/`의 가상 한빛대학교 및 가상 회사 문서는 자동 테스트 전용이며 실제 서비스 데이터가 아닙니다.

## 구성

- `rag/`: PDF/TXT/Markdown 로딩, 조항 중심 청킹, 로컬 다국어 임베딩, Chroma 영속 저장, 도메인 검색
- `mcp_server.py` / `agent/mcp_client.py`: stdio MCP Tool, Resource, Prompt와 연결
- `agent/graph.py`: 분류, 맥락, 계획, 검색, 제한적 재검색, 판단, 작성, 검증
- `server.py`: `/chat`, `/chat/stream`, SQLite 대화 이력, LangSmith 추적
- `web_chat.html` / `web_demo_server.py`: Vue 3 채팅과 SSE 프록시
- `application_forms/`: 빈 신청서 검색, 입력 항목 확인, 작성 초안 생성
- `scripts/evaluate.py`: 단일 검색 RAG와 Agentic RAG 평가

## 설치 및 환경변수

Python 3.11 이상이 필요합니다. 첫 설치에는 PyPI 패키지 다운로드가 필요합니다. 기본 임베딩은 로컬 Ollama의 `qwen3-embedding:0.6b`를 사용합니다. 대안으로 `./.venv/bin/python3 -m pip install -r requirements-huggingface.txt`를 실행하고 `POLICY_EMBEDDING_BACKEND=huggingface`, `POLICY_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`를 설정할 수 있으며 첫 실행에 모델 다운로드가 필요합니다. 임베딩 백엔드나 모델을 바꾸면 기존 인덱스와 차원이 다를 수 있으므로 별도 `POLICY_CHROMA_DIR`을 사용하세요.

```bash
python3 -m venv .venv
./.venv/bin/python3 -m pip install -r requirements.txt
cp -n .env.example .env
```

`.env`에서 `LLM_BACKEND=ollama` 또는 `openai`를 선택합니다. Ollama는 해당 모델을 로컬에 설치하고 서버를 띄워야 합니다. OpenAI는 본인의 `OPENAI_API_KEY`가 필요하며 과금될 수 있습니다. `LLM_BACKEND=fake`는 연결 시험용으로, 실제 규정 판단을 하지 않습니다. `.env`는 Git에서 제외됩니다.

웹 화면의 **답변 모델** 메뉴에서 Ollama와 OpenAI를 질문마다 바꿀 수 있습니다. OpenAI 항목은 서버의 `.env`에 `OPENAI_API_KEY`가 있을 때만 활성화됩니다. 키를 추가한 뒤에는 API 서버를 재시작하세요. 학사/사내 분야와 답변 모델마다 대화 이력을 분리합니다. 답변 모델을 OpenAI로 바꿔도 규정 검색 임베딩은 `POLICY_EMBEDDING_BACKEND` 설정을 사용하므로, 현재 Ollama 색인을 쓰는 경우 Ollama 서버도 계속 필요합니다. OpenAI 선택 시 질문과 검색된 규정 발췌가 OpenAI API로 전송되며 API 사용 요금이 발생할 수 있습니다.

## 문서 등록

실제 문서를 `data/policies/academic/` 또는 `data/policies/corporate/`에 넣습니다. PDF, UTF-8 TXT, Markdown을 지원합니다. 문서의 `시행일: YYYY-MM-DD` 줄은 메타데이터로 저장됩니다. 스캔 이미지 PDF는 OCR이 없어 텍스트를 추출할 수 없습니다.

```bash
./.venv/bin/python3 -m scripts.ingest_policies
```

기본 Chroma 경로는 `data/chroma/`이며 재시작 후 재사용합니다. 같은 파일은 SHA-256 해시와 색인 규칙 버전이 같으면 건너뛰고, 수정된 파일이나 색인 규칙이 바뀐 파일은 해당 문서 청크를 교체합니다. 50개 이상 청크로 나뉘는 긴 PDF에서는 `제N조` 형태의 조항이 있는 청크만 색인하고, 조항이 전체의 10% 미만인 파일은 교과목 개요 등 부록으로 판단해 제외합니다. 짧은 독립 PDF와 TXT/Markdown은 조항 제목이 없어도 색인합니다. 긴 PDF의 조항 외 내용도 검색해야 한다면 해당 부분을 별도 TXT/Markdown 파일로 등록하세요. 검색 품질을 위해 실제 적용 규정의 버전과 시행일을 문서에 명시하세요. 오래된 문서가 함께 검색되면 적용 여부를 검토해야 합니다.

학칙을 새 파일로 완전히 교체할 때는 기존 파일을 `data/policies/academic/`에서 빼고 새 파일을 넣은 다음 다음 명령을 실행합니다. 현재 폴더의 학사 문서를 먼저 읽고 임베딩할 수 있는지 확인한 후, **학사 규정 검색 색인만** 초기화해 다시 등록합니다. 새 폴더에 색인 가능한 문서가 없으면 초기화를 거부합니다. 사내 규정 색인, 신청서 양식, 채팅 기록은 유지됩니다.

```bash
./.venv/bin/python3 -m scripts.ingest_policies --reset-domain academic
```

사내 규정만 교체하려면 `--reset-domain corporate`를 사용합니다. 일반 색인 명령은 삭제된 파일의 기존 검색 내용을 제거하지 않으므로, 파일명까지 바꾸는 교체 작업에는 위 명령을 사용하세요. 색인을 교체하는 동안에는 API 서버에서 질문을 처리하지 않도록 잠시 중지했다가 완료 후 다시 실행하세요.

## 신청서 양식과 작성 초안

빈 신청서 파일은 웹 화면의 **빈 양식 추가** 버튼으로 등록하거나 `data/forms/academic/`(학사) 또는 `data/forms/corporate/`(사내)에 넣습니다. `.hwp`, `.hwpx`, `.docx`를 지원하며 최대 10MB입니다. 등록 후 재색인할 필요 없이 목록과 관련 질문의 검색 결과에 나타납니다. 규정 문서는 위의 `data/policies/`에 별도로 등록하고 색인해야 합니다. 신청서가 있다는 사실만으로 신청 자격이 확인되지는 않습니다.

양식에서 성명·학번 등의 항목을 찾으면 **초안 작성** 화면에 필요한 입력칸이 나타납니다. DOCX/HWPX 원본에 `{{student_name}}`, `{{student_id}}`처럼 표시를 넣으면 해당 위치를 채운 사본을 다운로드합니다. 표시가 없는 DOCX/HWPX와 HWP는 원본을 그대로 두고 입력 내용을 정리한 별도 DOCX 초안을 만듭니다. 특히 HWP 원본을 직접 수정한 파일은 생성하지 않습니다. 서명과 최종 제출은 본인이 원본 양식 및 최신 공고와 대조해 처리해야 합니다.

자동으로 입력칸을 찾지 못한 양식은 같은 폴더에 `파일명.hwp.form.json`처럼 원본 파일명 뒤에 `.form.json`을 붙인 설정 파일을 둘 수 있습니다. 예:

```json
{
  "title": "부전공 신청원",
  "source_url": "https://example.edu/forms/minor",
  "version": "2026-1",
  "fields": [
    {"key": "student_name", "label": "성명", "required": true},
    {"key": "student_id", "label": "학번", "required": true},
    {"key": "course_plan", "label": "신청 과목", "required": false, "multiline": true}
  ]
}
```

인적사항은 채팅창이 아닌 **초안 작성** 화면에 입력하세요. 초안 요청은 대화 이력과 LangSmith 추적에 기록하지 않고 서버 메모리에서만 처리합니다. 브라우저에서 다운로드가 시작되면 입력칸을 비웁니다. 양식 원본과 설정 파일은 Git에서 제외되어 있으므로 실제 파일은 사용할 서버에 직접 등록해야 합니다.

## 실행

```bash
./.venv/bin/python3 server.py
./.venv/bin/python3 web_demo_server.py
```

브라우저에서 `http://127.0.0.1:8600`을 엽니다. 8100 포트는 API, 8600 포트는 Vue 프록시입니다. 기존 `sh/run_agent_ollma.sh`, `sh/run_agent_openai.sh`, `sh/run_frontend.sh`도 사용할 수 있으며, 실행 시 세션 DB를 삭제하지 않습니다.

```bash
curl -s -X POST http://127.0.0.1:8100/chat \
  -H 'content-type: application/json' \
  -d '{"message":"부전공 포기 신청에 필요한 서류는 무엇인가요?","domain":"academic","thread_id":"demo-1"}'
```

응답에는 `reply`, 구조화된 `analysis`, 검색된 `sources`, 실행 `events`, 추적 및 비용 `metadata`가 포함됩니다. `GET /chat/stream`은 진행 이벤트와 검증 후 `FinalAnswer`, `[DONE]`을 보냅니다. 대화는 `session.db`에 저장됩니다.

인사말, 사용법 질문, 명백한 일상 대화는 규정 검색 없이 안내합니다. 그 밖의 질문은 선택한 분야의 규정을 검색하되, 질문의 핵심 주제가 포함된 조항을 우선 찾고 관련성이 낮은 결과는 답변 근거에서 제외합니다. 관련 규정이 없는 일반 대화에는 임의의 규정을 인용하거나 신청 가능 여부를 판단하지 않습니다. `출석인정`처럼 미리 나열되지 않은 주제도 규정 검색을 시도합니다. 새 주제가 명시된 질문은 현재 질문만 검색하고, `그 서류는?`처럼 앞선 내용에 의존하는 후속 질문에만 직전 대화 맥락을 사용합니다. 검색 결과는 벡터 유사도와 질문 핵심 단어의 일치도, 조항 제목을 함께 사용해 정렬합니다. 서류·절차·자격·일반 설명 질문은 화면에서 각각 알맞은 형식으로 표시하며, 검증된 근거로 인용된 조항만 출처에 표시합니다. 검색된 조항에 신청서 제출·작성 요건이 있거나 질문에서 양식·서류를 요청하면 관련 신청서도 찾습니다. 취소 서류 질문에서는 일반 신청서를 취소 신청서로 표시하지 않습니다.

## 테스트와 평가

```bash
./.venv/bin/python3 -m pytest -q
LANGCHAIN_TRACING_V2=false ./.venv/bin/python3 -m scripts.evaluate_real
```

실제 학칙 평가 정답표는 `docs/evaluation-real-cases.json`입니다. 먼저 정답표에 기록된 학칙·규정집 PDF를 로컬에서 색인해야 합니다. 평가 명령은 동일한 Chroma 색인과 Ollama 모델을 사용해 단일 검색 RAG와 현재 LangGraph를 차례로 실행하고, 결과를 `docs/evaluation-real-results.csv`와 `.md`에 기록합니다. `tests/fixtures/cases.json`은 가상 문서용 개발 예시입니다. 자세한 구조는 [architecture.md](docs/architecture.md), 측정 조건과 해석 범위는 [evaluation.md](docs/evaluation.md)를 참고하세요.
