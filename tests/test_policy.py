import asyncio
from pathlib import Path

from langchain_core.embeddings import Embeddings

from agent.deps import SupervisorDeps
from agent.graph import build_graph, sanitize_analysis, validate_analysis
from agent.policy_llm import PolicyLLM
from rag.loader import load_policy
from rag.retriever import get_document, search_policies
from rag.splitter import split_policy
from rag.vectorstore import ingest_directory, open_store

FIXTURE = Path(__file__).parent / "fixtures"


class KeywordEmbeddings(Embeddings):
    terms = ("부전공", "졸업", "휴학", "주차비", "식비", "장학금")

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(text.count(term)) for term in self.terms] + [0.01]


def test_loader_splitter_and_dedupe(tmp_path):
    pages = load_policy(FIXTURE / "academic" / "가상_학사규정.md", FIXTURE)
    chunks = split_policy(pages)
    assert any(c.metadata["article"].startswith("제1조") for c in chunks)
    assert all(c.metadata["effective_date"] == "2026-03-01" for c in chunks)
    store = open_store(KeywordEmbeddings(), tmp_path / "chroma")
    first = ingest_directory(FIXTURE, store)
    second = ingest_directory(FIXTURE, store)
    assert first["added"] == 4
    assert second["added"] == 0 and second["skipped"] == 4
    results = search_policies("부전공 포기", "academic", store=store)
    assert results and all(r["domain"] == "academic" for r in results)
    document = get_document(results[0]["document_id"], store)
    assert document["chunks"]
    assert all(r["domain"] == "corporate" for r in search_policies("주차비", "corporate", store=store))
    versions = [r for r in search_policies("부전공 포기", "academic", top_k=5, store=store)
                if r["article"].startswith("제1조")]
    assert versions[0]["effective_date"] >= versions[-1]["effective_date"]


def test_invalid_domain(tmp_path):
    store = open_store(KeywordEmbeddings(), tmp_path / "chroma")
    try:
        search_policies("test", "unknown", store=store)
        assert False
    except ValueError:
        pass


EVIDENCE = {"id": "doc:1:0", "document_id": "doc", "text": "제1조 부전공 포기 신청서는 학사팀에 제출한다.",
            "document_name": "가상_학사규정.md", "article": "제1조", "page": 0,
            "domain": "academic", "effective_date": "2026-03-01"}


class MockMCP:
    def __init__(self, evidence=None, fails=False):
        self.evidence = evidence or []
        self.fails = fails
        self.calls = 0

    async def call_tool(self, name, arguments, server="policy"):
        self.calls += 1
        if self.fails:
            raise RuntimeError("MCP unavailable")
        return {"results": self.evidence}

    async def find_prompt(self, name):
        return "policy"

    async def get_prompt(self, *args, **kwargs):
        return "근거만 사용"


class MockPolicyLLM:
    def __init__(self, valid=True):
        self.valid = valid

    def analyze(self, query, history, evidence, procedure):
        return {"eligibility": "eligible", "reason": "신청서 제출 가능",
                "reason_support": {"citation_id": "doc:1:0", "quote": "신청서는 학사팀에 제출한다"},
                "documents": [{"text": "부전공 포기 신청서", "citation_id": "doc:1:0",
                               "quote": "부전공 포기 신청서"}],
                "procedure": [{"text": "학사팀에 제출", "citation_id": "doc:1:0",
                               "quote": "학사팀에 제출한다"}]}, {"tokens": 10, "usd": 0.001}

    def judge(self, analysis, evidence):
        return self.valid


def run_graph(evidence=None, fails=False, llm=None, query="부전공 포기 신청서가 필요한가요?", history=None):
    mcp = MockMCP(evidence, fails)
    deps = SupervisorDeps(None, None, mcp, None, "policy-v2", llm or MockPolicyLLM())
    result = asyncio.run(build_graph(deps).ainvoke({"query": query, "domain": "academic",
        "history": history or [], "events": [], "tokens": 0, "cost_usd": 0}))
    return result, mcp


def test_grounded_answer_and_sources():
    result, mcp = run_graph([EVIDENCE])
    assert result["analysis"]["eligibility"] == "eligible"
    assert "가상_학사규정.md" in result["final"]
    assert mcp.calls == 1


def test_missing_evidence_research_once_then_abstains():
    result, mcp = run_graph([])
    assert result["analysis"]["eligibility"] == "need_more_information"
    assert mcp.calls == 2
    assert result["research_rounds"] == 1


def test_mcp_failure_abstains():
    result, mcp = run_graph(fails=True)
    assert result["analysis"]["eligibility"] == "need_more_information"
    assert any(e["type"] == "ResearchFailed" for e in result["events"])
    assert mcp.calls == 2


def test_validator_rejects_fabricated_citation():
    bad = {"eligibility": "eligible", "reason_support": {"citation_id": "other", "quote": "가짜"}}
    assert validate_analysis(bad, [EVIDENCE])


def test_validator_fallback_after_two_rejections():
    result, _ = run_graph([EVIDENCE], llm=MockPolicyLLM(valid=False))
    assert result["validation_attempts"] == 2
    assert result["analysis"]["eligibility"] == "need_more_information"


def test_followup_uses_history():
    result, _ = run_graph([EVIDENCE], query="그 서류는 어디에 내나요?",
        history=[{"role": "user", "content": "부전공 포기 신청을 하려 합니다"},
                 {"role": "assistant", "content": "신청서를 확인하세요"}])
    assert "부전공 포기" in result["context_query"]


def test_general_chat_does_not_search():
    result, mcp = run_graph(query="안녕하세요")
    assert result["route"] == "chat" and mcp.calls == 0


def test_multiple_regulations_can_be_retained():
    second = {**EVIDENCE, "id": "doc2:1:0", "document_id": "doc2",
              "text": "제2조 졸업 요건은 별도로 확인한다.", "article": "제2조"}
    result, _ = run_graph([EVIDENCE, second], query="부전공 포기 후 졸업 요건도 확인해야 하나요?")
    assert len(result["evidence"]) == 2


def test_insufficient_fields_trigger_one_extra_search():
    class EmptyPolicyLLM:
        def analyze(self, *args):
            return {"eligibility": "need_more_information", "reason": "정보 부족"}, {"tokens": 0, "usd": 0}

        def judge(self, *args):
            return True

    result, mcp = run_graph([EVIDENCE], llm=EmptyPolicyLLM())
    assert mcp.calls == 2
    assert result["research_rounds"] == 1


def test_llm_failure_abstains():
    class BrokenBackend:
        name = "broken"

        def _chat(self, *args, **kwargs):
            raise RuntimeError("offline")

    result, usage = PolicyLLM(BrokenBackend()).analyze("부전공 포기", [], [EVIDENCE])
    assert result["eligibility"] == "need_more_information" and usage["tokens"] == 0


def test_validator_rejects_unsupported_field_and_shape():
    bad = {"eligibility": "need_more_information", "documents": [
        {"text": "존재하지 않는 서류", "quote": "부전공 포기 신청서", "citation_id": "doc:1:0"}],
        "deadline": [{"text": "기한 없음"}]}
    assert len(validate_analysis(bad, [EVIDENCE])) >= 2


def test_uncited_conclusion_is_neutralized():
    raw = {"eligibility": "eligible", "reason": "무조건 신청 가능합니다"}
    safe = sanitize_analysis(raw, [EVIDENCE])
    assert safe["eligibility"] == "need_more_information"
    assert "신청 가능합니다" not in safe["reason"]
