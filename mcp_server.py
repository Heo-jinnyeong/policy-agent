"""MCP capabilities for indexed policy documents."""
from fastmcp import FastMCP

from rag.retriever import get_document, search_policies
from application_forms.catalog import search_forms

mcp = FastMCP(name="policy-agent")


@mcp.tool()
def search_policy_documents(query: str, domain: str = "academic", top_k: int = 3) -> dict:
    """Search indexed regulations and return their text and provenance."""
    return {"results": search_policies(query, domain, top_k)}


@mcp.tool()
def search_application_forms(query: str, domain: str = "academic") -> dict:
    """Find uploaded blank application forms without applicant information."""
    return {"results": search_forms(query, domain)}


@mcp.resource("policy://documents/{document_id}")
def policy_document(document_id: str) -> str:
    """Read the indexed text for one policy document."""
    import json
    return json.dumps(get_document(document_id), ensure_ascii=False)


@mcp.prompt()
def policy_analysis_prompt(situation: str, domain: str = "academic") -> str:
    """Standard procedure for evidence based eligibility analysis."""
    return (f"분야: {domain}\n상황: {situation}\n"
            "관련 규정의 적용 조건과 예외를 확인하라. 신청 가능 여부, 서류, 절차, 기한, "
            "담당 부서는 인용한 본문에서 확인한 항목만 제시하라. 판단에 꼭 필요한 사용자 "
            "조건이 빠졌으면 추가 질문을 하라. 근거가 없으면 판단을 유보하라.")


if __name__ == "__main__":
    try:
        mcp.run(show_banner=False)
    except TypeError:
        mcp.run()
