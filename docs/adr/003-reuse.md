# ADR 003: MCP와 Vue 재사용

상태: 채택

기존 stdio MCP 연결, FastAPI lifespan, SSE 프록시, Vue 3 단일 HTML 채팅, SQLite 세션 저장을 유지했다. MCP capability와 LangGraph 노드만 규정 도메인으로 바꿨다. 프런트엔드 빌드 과정이 없다는 장점이 있으나 Vue CDN 접근이 필요한 점은 배포 시 검토해야 한다.
