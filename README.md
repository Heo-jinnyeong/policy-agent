# Policy Agent

규정 문서를 검색하고 신청 자격·서류·절차를 근거와 함께 안내하는 FastAPI + LangGraph + MCP + Vue 서비스입니다. 초기 범위는 학사 규정이며, 같은 인덱스 구조로 사내 규정도 검색할 수 있습니다.

**이 저장소에는 실제 규정 문서가 없습니다.** 문서를 직접 등록하기 전에는 신청 가능 여부를 판단하지 않습니다. `tests/fixtures/`의 가상 한빛대학교 및 가상 회사 문서는 자동 테스트 전용이며 실제 서비스 데이터가 아닙니다.

## 구성

- `rag/`: PDF/TXT/Markdown 로딩, 조항 중심 청킹, 로컬 다국어 임베딩, Chroma 영속 저장, 도메인 검색
- `mcp_server.py` / `agent/mcp_client.py`: stdio MCP Tool, Resource, Prompt와 연결
- `agent/graph.py`: 분류, 맥락, 계획, 검색, 제한적 재검색, 판단, 작성, 검증
- `server.py`: `/chat`, `/chat/stream`, SQLite 대화 이력, LangSmith 추적
- `web_chat.html` / `web_demo_server.py`: Vue 3 채팅과 SSE 프록시
- `scripts/evaluate.py`: 단일 검색 RAG와 Agentic RAG 평가

## 설치 및 환경변수

Python 3.11 이상이 필요합니다. 첫 설치에는 PyPI 패키지 다운로드가 필요합니다. 기본 임베딩은 로컬 Ollama의 `qwen3-embedding:0.6b`를 사용합니다. 대안으로 `./.venv/bin/python3 -m pip install -r requirements-huggingface.txt`를 실행하고 `POLICY_EMBEDDING_BACKEND=huggingface`, `POLICY_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`를 설정할 수 있으며 첫 실행에 모델 다운로드가 필요합니다. 임베딩 백엔드나 모델을 바꾸면 기존 인덱스와 차원이 다를 수 있으므로 별도 `POLICY_CHROMA_DIR`을 사용하세요.

```bash
python3 -m venv .venv
./.venv/bin/python3 -m pip install -r requirements.txt
cp -n .env.example .env
```

`.env`에서 `LLM_BACKEND=ollama` 또는 `openai`를 선택합니다. Ollama는 해당 모델을 로컬에 설치하고 서버를 띄워야 합니다. OpenAI는 본인의 `OPENAI_API_KEY`가 필요하며 과금될 수 있습니다. `LLM_BACKEND=fake`는 연결 시험용으로, 실제 규정 판단을 하지 않습니다. `.env`는 Git에서 제외됩니다.

## 문서 등록

실제 문서를 `data/policies/academic/` 또는 `data/policies/corporate/`에 넣습니다. PDF, UTF-8 TXT, Markdown을 지원합니다. 문서의 `시행일: YYYY-MM-DD` 줄은 메타데이터로 저장됩니다. 스캔 이미지 PDF는 OCR이 없어 텍스트를 추출할 수 없습니다.

```bash
./.venv/bin/python3 -m scripts.ingest_policies
```

기본 Chroma 경로는 `data/chroma/`이며 재시작 후 재사용합니다. 같은 파일은 SHA-256 해시로 건너뛰고 수정된 파일은 해당 문서 청크를 교체합니다. 검색 품질을 위해 실제 적용 규정의 버전과 시행일을 문서에 명시하세요. 오래된 문서가 함께 검색되면 적용 여부를 검토해야 합니다.

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

## 테스트와 평가

```bash
./.venv/bin/python3 -m pytest -q
./.venv/bin/python3 -m scripts.evaluate --cases path/to/real_reference_cases.json
```

평가에는 실제 등록 문서에서 검증한 질문·기대 조항·기대 판단이 10개 이상 필요합니다. `tests/fixtures/cases.json`은 가상 문서용 예시입니다. 평가 실행 시 API 서버와 로컬 평가 프로세스가 같은 Chroma DB 및 LLM 설정을 사용해야 합니다. 결과는 기본적으로 `docs/evaluation-results.csv`와 `.md`에 기록됩니다. 자세한 구조는 [architecture.md](docs/architecture.md), 평가 범위는 [evaluation.md](docs/evaluation.md)를 참고하세요.
