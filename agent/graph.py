"""Evidence first LangGraph for policy guidance."""
import re
from functools import partial
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from .deps import SupervisorDeps
from .retry import with_retry

POLICY_TERMS = re.compile(r"규정|학칙|부전공|복수전공|졸업|휴학|복학|신청|서류|제출|등록|장학|출장|경비|식비|주차비|policy|regulation", re.I)
CORPORATE_TERMS = re.compile(r"사내|회사|출장|경비|식비|주차비|법인", re.I)
VALID_STATUSES = {"eligible", "conditional", "ineligible", "need_more_information"}


def sanitize_analysis(raw: dict, evidence: list[dict]) -> dict:
    """Discard unsupported model fields before they can reach a user response."""
    if not isinstance(raw, dict):
        raw = {}
    by_id = {e["id"]: e["text"] for e in evidence}

    def supported(item):
        if not isinstance(item, dict):
            return None
        citation = item.get("citation_id")
        quote = item.get("quote")
        source = by_id.get(citation, "")
        if not isinstance(quote, str) or not quote or quote not in source:
            return None
        text = item.get("text")
        if text is not None and (not isinstance(text, str) or text not in quote or len(text) > 100):
            return None
        return item

    status = raw.get("eligibility") if raw.get("eligibility") in VALID_STATUSES else "need_more_information"
    reason_support = supported(raw.get("reason_support"))
    unsupported_status = status != "need_more_information" and not reason_support
    if status != "need_more_information" and not reason_support:
        status = "need_more_information"
    clean = {"eligibility": status,
             "reason": ("신청 가능 여부의 근거를 확인하지 못했습니다." if unsupported_status
                        else ("규정 근거 또는 사용자 조건이 부족하여 신청 가능 여부를 판단할 수 없습니다."
                              if status == "need_more_information" and not reason_support and evidence
                              else raw.get("reason", ""))),
             "reason_support": reason_support or {},
             "follow_up_question": raw.get("follow_up_question", "") if status == "need_more_information" else ""}
    for key in ("conditions", "documents", "procedure"):
        items = raw.get(key, [])
        clean[key] = ([valid for item in items if (valid := supported(item))]
                      if isinstance(items, list) else [])
    for key in ("deadline", "department"):
        item = raw.get(key, {})
        if isinstance(item, list):
            item = item[0] if len(item) == 1 else {}
        clean[key] = supported(item) or {}
    if status == "need_more_information" and not clean["reason"]:
        clean["reason"] = "규정 근거 또는 사용자 조건이 부족하여 신청 가능 여부를 판단할 수 없습니다."
    return clean


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    history: list[dict]
    domain: str
    route: str
    context_query: str
    search_query: str
    evidence: list[dict]
    research_rounds: int
    validation_attempts: int
    analysis: dict
    final: str
    events: list[dict]
    cost_usd: float
    tokens: int
    validation_feedback: str
    retry_research: bool
    analysis_research_needed: bool


def _event(state, **event):
    return state.get("events", []) + [event]


@traceable(name="policy.route")
async def route_node(state: AgentState, deps: SupervisorDeps) -> dict:
    query = state["query"]
    prior_policy = any(POLICY_TERMS.search(m["content"]) for m in state.get("history", [])[-4:])
    route = "policy" if POLICY_TERMS.search(query) or prior_policy else "chat"
    domain = state.get("domain") or ("corporate" if CORPORATE_TERMS.search(query) else "academic")
    return {"route": route, "domain": domain,
            "events": _event(state, type="RouteDecided", route=route, domain=domain)}


@traceable(name="policy.chat")
async def chat_node(state: AgentState, deps: SupervisorDeps) -> dict:
    reply = "규정이나 신청 절차에 관한 질문을 입력해 주세요."
    return {"final": reply, "analysis": {"eligibility": "need_more_information"},
            "events": _event(state, type="Finalize")}


@traceable(name="policy.context")
async def context_node(state: AgentState, deps: SupervisorDeps) -> dict:
    query = state["query"]
    previous = [m["content"] for m in state.get("history", []) if m["role"] == "user"][-2:]
    context = "\n".join(previous + [query]) if previous and len(query) < 80 else query
    return {"context_query": context,
            "events": _event(state, type="ContextAnalyzed", domain=state["domain"])}


@traceable(name="policy.plan")
async def plan_node(state: AgentState, deps: SupervisorDeps) -> dict:
    return {"search_query": state["context_query"],
            "events": _event(state, type="PlanCreated", query=state["context_query"])}


@traceable(name="policy.research")
async def research_node(state: AgentState, deps: SupervisorDeps) -> dict:
    try:
        result, retries = await with_retry(deps.mcp.call_tool, "search_policy_documents",
            {"query": state["search_query"], "domain": state["domain"], "top_k": 5},
            server="policy", retries=2)
    except Exception as exc:
        return {"evidence": state.get("evidence", []),
                "events": _event(state, type="ResearchFailed", message=str(exc))}
    if "result" in result and isinstance(result["result"], dict):
        result = result["result"]
    new = result.get("results", [])
    evidence = list(state.get("evidence", []))
    seen = {(e["document_id"], e["text"]) for e in evidence}
    for entry in new:
        key = (entry["document_id"], entry["text"])
        if key not in seen:
            evidence.append(entry)
            seen.add(key)
    return {"evidence": evidence,
            "events": _event(state, type="ResearchCompleted", count=len(new), retries=retries)}


@traceable(name="policy.evidence_check")
async def evidence_node(state: AgentState, deps: SupervisorDeps) -> dict:
    sufficient = bool(state.get("evidence"))
    rounds = state.get("research_rounds", 0)
    if not sufficient and rounds < 1:
        query = state["context_query"] + " 관련 규정 적용 조건 예외 신청 서류 절차"
        return {"research_rounds": 1, "search_query": query, "retry_research": True,
                "events": _event(state, type="ReplanCheck", is_sufficient=False, round=1)}
    return {"retry_research": False,
            "events": _event(state, type="ReplanCheck", is_sufficient=sufficient, round=rounds)}


@traceable(name="policy.analysis")
async def analysis_node(state: AgentState, deps: SupervisorDeps) -> dict:
    if not state.get("evidence"):
        result = {"eligibility": "need_more_information",
                  "reason": "검색된 규정 근거가 없어 신청 가능 여부를 판단할 수 없습니다.",
                  "follow_up_question": "관련 규정 문서를 등록하거나 담당 부서에 확인해 주세요."}
        usage = {"tokens": 0, "usd": 0}
    else:
        procedure = ""
        try:
            server = await deps.mcp.find_prompt("policy_analysis_prompt")
            procedure = await deps.mcp.get_prompt("policy_analysis_prompt",
                {"situation": state["context_query"], "domain": state["domain"]}, server=server)
        except (ValueError, RuntimeError):
            pass
        if state.get("validation_feedback"):
            procedure += "\n이전 답변 검증 오류: " + state["validation_feedback"]
        result, usage = deps.policy_llm.analyze(state["context_query"],
            state.get("history", []), state["evidence"], procedure)
    result = sanitize_analysis(result, state.get("evidence", []))
    coverage = any(result.get(key) for key in ("conditions", "documents", "procedure", "deadline", "department"))
    replan = bool(state.get("evidence") and not coverage and
                  not result.get("follow_up_question") and state.get("research_rounds", 0) < 1)
    updates = {"analysis": result, "analysis_research_needed": replan,
            "tokens": state.get("tokens", 0) + usage.get("tokens", 0),
            "cost_usd": state.get("cost_usd", 0) + usage.get("usd", 0),
            "events": _event(state, type="PolicyAnalyzed", eligibility=result["eligibility"])}
    if replan:
        updates["search_query"] = state["context_query"] + " 적용 조건 예외 필요 서류 제출 절차"
        updates["research_rounds"] = 1
        updates["events"] = updates["events"] + [
            {"type": "ReplanCheck", "is_sufficient": False, "round": 1, "reason": "supported fields missing"}]
    return updates


@traceable(name="policy.missing_information")
async def missing_node(state: AgentState, deps: SupervisorDeps) -> dict:
    question = state["analysis"].get("follow_up_question", "")
    return {"events": _event(state, type="MissingInformation", question=question)}


def _normalise_items(items):
    return [item for item in items if isinstance(item, dict) and item.get("text")] if isinstance(items, list) else []


def _render(analysis: dict, evidence: list[dict]) -> str:
    labels = {"eligible": "신청 가능", "conditional": "조건부 가능",
              "ineligible": "신청 불가", "need_more_information": "추가 확인 필요"}
    lines = [f"판단: {labels[analysis['eligibility']]}", analysis.get("reason", "")]
    for key, label in (("conditions", "적용 조건"), ("documents", "필요 서류"),
                       ("procedure", "신청 절차")):
        items = _normalise_items(analysis.get(key, []))
        if items:
            lines.append(label + ": " + "; ".join(x["text"] for x in items))
    for key, label in (("deadline", "신청 기한"), ("department", "담당 부서")):
        value = analysis.get(key, {})
        if isinstance(value, dict) and value.get("text"):
            lines.append(f"{label}: {value['text']}")
    if analysis.get("follow_up_question"):
        lines.append("추가 확인: " + analysis["follow_up_question"])
    support = analysis.get("reason_support") or {}
    cited = {support.get("citation_id") if isinstance(support, dict) else None}
    for key in ("conditions", "documents", "procedure"):
        cited.update(item.get("citation_id") for item in _normalise_items(analysis.get(key, [])))
    for key in ("deadline", "department"):
        value = analysis.get(key, {})
        if isinstance(value, dict):
            cited.add(value.get("citation_id"))
    sources = [e for e in evidence if e["id"] in cited]
    if sources:
        lines.append("근거: " + "; ".join(
            f"{e['document_name']} {e.get('article') or ''}"
            + (f" {e['page']}쪽" if e.get("page") else "") for e in sources))
    return "\n".join(line for line in lines if line)


@traceable(name="policy.write")
async def write_node(state: AgentState, deps: SupervisorDeps) -> dict:
    return {"final": _render(state["analysis"], state.get("evidence", [])),
            "events": _event(state, type="DraftWritten")}


def validate_analysis(analysis: dict, evidence: list[dict]) -> list[str]:
    problems = []
    by_id = {e["id"]: e["text"] for e in evidence}
    if analysis.get("eligibility") not in VALID_STATUSES:
        problems.append("invalid eligibility")
    fields = []
    support = analysis.get("reason_support")
    if analysis.get("eligibility") != "need_more_information":
        if not isinstance(support, dict):
            problems.append("missing eligibility support")
        else:
            fields.append(support)
    for key in ("conditions", "documents", "procedure"):
        value = analysis.get(key, [])
        if not isinstance(value, list):
            problems.append(f"invalid {key}")
        else:
            fields.extend(_normalise_items(value))
    for key in ("deadline", "department"):
        value = analysis.get(key, {})
        if value and not isinstance(value, dict):
            problems.append(f"invalid {key}")
        if isinstance(value, dict) and value.get("text"):
            fields.append(value)
    for item in fields:
        quote = item.get("quote", "")
        if not quote or quote not in by_id.get(item.get("citation_id"), ""):
            problems.append("unverified citation or quote")
        if item.get("text") and (len(item["text"]) > 100 or item["text"] not in quote):
            problems.append("field text is not a concise source excerpt")
    if not evidence and analysis.get("eligibility") != "need_more_information":
        problems.append("conclusion without evidence")
    return problems


@traceable(name="policy.validate")
async def validate_node(state: AgentState, deps: SupervisorDeps) -> dict:
    problems = validate_analysis(state["analysis"], state.get("evidence", []))
    usage = {"tokens": 0, "usd": 0}
    if not problems and state.get("evidence") and state["analysis"]["eligibility"] != "need_more_information":
        judgment = deps.policy_llm.judge(state["analysis"], state["evidence"])
        valid, usage = judgment if isinstance(judgment, tuple) else (judgment, usage)
        if not valid:
            problems.append("semantic judge rejected analysis")
    attempts = state.get("validation_attempts", 0) + 1
    if problems and attempts >= 2:
        safe = {"eligibility": "need_more_information",
                "reason": "검색 근거와 답변의 일치 여부를 확인하지 못해 판단을 보류합니다.",
                "follow_up_question": "담당 부서에 원문 규정을 확인해 주세요."}
        return {"analysis": safe, "final": _render(safe, []), "validation_attempts": attempts,
                "validation_feedback": None,
                "tokens": state.get("tokens", 0) + usage.get("tokens", 0),
                "cost_usd": state.get("cost_usd", 0) + usage.get("usd", 0),
                "events": _event(state, type="Validation", valid=False, fallback=True, problems=problems)}
    return {"validation_attempts": attempts,
            "validation_feedback": ", ".join(problems) if problems else None,
            "tokens": state.get("tokens", 0) + usage.get("tokens", 0),
            "cost_usd": state.get("cost_usd", 0) + usage.get("usd", 0),
            "events": _event(state, type="Validation", valid=not problems, problems=problems)}


@traceable(name="policy.finalize")
async def finalize_node(state: AgentState, deps: SupervisorDeps) -> dict:
    return {"events": _event(state, type="Finalize")}


def build_graph(deps: SupervisorDeps):
    graph = StateGraph(AgentState)
    for name, fn in (("route", route_node), ("chat", chat_node), ("context", context_node),
                     ("plan", plan_node), ("research", research_node), ("evidence", evidence_node),
                     ("analysis", analysis_node), ("missing", missing_node), ("write", write_node),
                     ("validate", validate_node), ("finalize", finalize_node)):
        graph.add_node(name, partial(fn, deps=deps))
    graph.set_entry_point("route")
    graph.add_conditional_edges("route", lambda s: "chat" if s["route"] == "chat" else "context")
    graph.add_edge("chat", END)
    graph.add_edge("context", "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "evidence")
    graph.add_conditional_edges("evidence", lambda s: "research" if s.get("retry_research") else "analysis")
    graph.add_conditional_edges("analysis", lambda s: "research" if s.get("analysis_research_needed") else "missing")
    graph.add_edge("missing", "write")
    graph.add_edge("write", "validate")
    graph.add_conditional_edges("validate", lambda s: "analysis" if s.get("validation_feedback") else "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
