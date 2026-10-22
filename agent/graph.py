"""Evidence first LangGraph for policy guidance."""
import re
from functools import partial
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from .deps import SupervisorDeps
from .intent import FOLLOW_UP_START, POLICY_CUES, classify
from .retry import with_retry
from rag.retriever import MIN_RELEVANCE

FORM_TERMS = re.compile(r"신청|지원|양식|서식|작성|제출|서류|증빙", re.I)
DOCUMENT_QUESTION = re.compile(r"서류|증빙|신청서|양식|취소원|제출.*(?:무엇|뭐|어떤)", re.I)
PROCEDURE_QUESTION = re.compile(r"절차|어떻게\s*(?:신청|제출|취소)|어디에\s*(?:신청|제출|내)", re.I)
ELIGIBILITY_QUESTION = re.compile(r"가능|할\s*수\s*있|자격|대상(?:인가|이야|인지|자)", re.I)
FORM_REQUIREMENT = re.compile(
    r"(?:신청서|신청원|신고서|서식|양식).{0,50}(?:제출|작성|신청)"
    r"|(?:제출|작성).{0,50}(?:신청서|신청원|신고서|서식|양식)", re.S)
CORPORATE_TERMS = re.compile(r"사내|회사|출장|경비|식비|주차비|법인", re.I)
VALID_STATUSES = {"eligible", "conditional", "ineligible", "need_more_information"}
PROHIBITION_TERMS = re.compile(r"불가|금지|허용하지|할 수 없|할 수없|제한|대상에서 제외|인정하지 않")


def sanitize_analysis(raw: dict, evidence: list[dict]) -> dict:
    """Discard unsupported model fields before they can reach a user response."""
    if not isinstance(raw, dict):
        raw = {}
    by_id = {e["id"]: e["text"] for e in evidence}

    def source_span(source: str, excerpt: str) -> str | None:
        """Anchor a PDF excerpt despite line wraps and spaces inside words."""
        if not excerpt:
            return None
        if excerpt in source:
            return excerpt
        positions = [index for index, char in enumerate(source) if not char.isspace()]
        compact_source = "".join(source[index] for index in positions)
        compact_excerpt = "".join(char for char in excerpt if not char.isspace())
        if not compact_excerpt:
            return None
        start = compact_source.find(compact_excerpt)
        if start < 0:
            return None
        return source[positions[start]:positions[start + len(compact_excerpt) - 1] + 1]

    def supported(item):
        if not isinstance(item, dict):
            return None
        citation = item.get("citation_id")
        quote = item.get("quote")
        source = by_id.get(citation, "")
        if not isinstance(quote, str) or not quote:
            return None
        exact_quote = source_span(source, quote)
        if exact_quote is None:
            return None
        text = item.get("text")
        anchored = {**item, "quote": exact_quote}
        if text is not None:
            if not isinstance(text, str):
                return None
            exact_text = source_span(exact_quote, text)
            if exact_text is None or len(exact_text) > 100:
                return None
            anchored["text"] = exact_text
        return anchored

    status = raw.get("eligibility") if raw.get("eligibility") in VALID_STATUSES else "need_more_information"
    reason_support = supported(raw.get("reason_support"))
    unsupported_status = status != "need_more_information" and not reason_support
    if status != "need_more_information" and not reason_support:
        status = "need_more_information"
    unsupported_prohibition = (status == "ineligible" and reason_support and
                               not PROHIBITION_TERMS.search(reason_support["quote"]))
    if unsupported_prohibition:
        status = "need_more_information"
    clean = {"eligibility": status,
             "reason": ("신청 불가를 뒷받침하는 금지 규정을 확인하지 못했습니다." if unsupported_prohibition
                        else "신청 가능 여부의 근거를 확인하지 못했습니다." if unsupported_status
                        else ("규정 근거 또는 사용자 조건이 부족하여 신청 가능 여부를 판단할 수 없습니다."
                              if status == "need_more_information" and not reason_support and evidence
                              else raw.get("reason", ""))),
             "reason_support": reason_support or {},
             "follow_up_question": raw.get("follow_up_question", "") if status == "need_more_information" else ""}
    if reason_support and re.search(r"명시되지 않|규정에 없|확인되지 않", clean["reason"]):
        # A retrieved excerpt cannot prove the absence of a rule elsewhere.
        clean["reason"] = re.sub(r"\s+", " ", reason_support["quote"]).strip()
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


def _related_cancellation_rule(query: str, evidence: list[dict]) -> dict:
    """Provide a cited context clause without presenting it as a required form."""
    subject = next((name for name in ("부전공", "복수전공") if name in query), "")
    action = next((word for word in ("취소", "포기") if word in query), "")
    if not subject or not action:
        return {}
    for entry in evidence:
        for clause in re.split(r"(?=[①-⑳])", entry.get("text", "")):
            if subject in clause and action in clause and len(clause) <= 350:
                quote = clause.strip()
                return {"text": re.sub(r"\s+", " ", quote),
                        "quote": quote, "citation_id": entry["id"]}
    return {}


class AgentState(TypedDict, total=False):
    query: str
    actual_question: str
    thread_id: str
    history: list[dict]
    domain: str
    route: str
    question_type: str
    context_query: str
    relevant_history: list[dict]
    is_follow_up: bool
    policy_cue: bool
    out_of_scope: bool
    research_failed: bool
    search_query: str
    evidence: list[dict]
    forms: list[dict]
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
    query = state["query"].strip()
    route, actual = classify(query, state.get("history", []), deps.fast_llm)
    policy_cue = bool(POLICY_CUES.search(actual))
    domain = state.get("domain") or ("corporate" if CORPORATE_TERMS.search(actual) else "academic")
    question_type = ("documents" if DOCUMENT_QUESTION.search(actual) else
                     "procedure" if PROCEDURE_QUESTION.search(actual) else
                     "eligibility" if ELIGIBILITY_QUESTION.search(actual) else "explanation")
    return {"route": route, "domain": domain, "question_type": question_type,
            "policy_cue": policy_cue, "actual_question": actual,
            "events": _event(state, type="RouteDecided", route=route, domain=domain)}


@traceable(name="policy.chat")
async def chat_node(state: AgentState, deps: SupervisorDeps) -> dict:
    reply = "규정이나 신청 절차에 관한 질문을 입력해 주세요."
    return {"final": reply, "analysis": {"eligibility": "need_more_information"},
            "events": _event(state, type="Finalize")}


@traceable(name="policy.context")
async def context_node(state: AgentState, deps: SupervisorDeps) -> dict:
    query = state["actual_question"].strip()
    history = state.get("history", [])
    previous_index = next((i for i in range(len(history) - 1, -1, -1)
                           if history[i].get("role") == "user"), None)
    is_follow_up = bool(previous_index is not None and FOLLOW_UP_START.match(query))
    relevant_history = history[previous_index:] if is_follow_up else []
    context = (history[previous_index]["content"] + "\n후속 질문: " + query
               if is_follow_up else query)
    return {"context_query": context, "relevant_history": relevant_history,
            "is_follow_up": is_follow_up,
            "events": _event(state, type="ContextAnalyzed", domain=state["domain"],
                             is_follow_up=is_follow_up)}


@traceable(name="policy.plan")
async def plan_node(state: AgentState, deps: SupervisorDeps) -> dict:
    return {"search_query": state["context_query"],
            "events": _event(state, type="PlanCreated", query=state["context_query"])}


@traceable(name="policy.research")
async def research_node(state: AgentState, deps: SupervisorDeps) -> dict:
    try:
        result, retries = await with_retry(deps.mcp.call_tool, "search_policy_documents",
            {"query": state["search_query"], "domain": state["domain"], "top_k": 3},
            server="policy", retries=2)
    except Exception as exc:
        return {"evidence": state.get("evidence", []),
                "research_failed": True,
                "events": _event(state, type="ResearchFailed", message=str(exc))}
    if "result" in result and isinstance(result["result"], dict):
        result = result["result"]
    new = result.get("results", [])
    new = [entry for entry in new if entry.get("relevance", MIN_RELEVANCE) >= MIN_RELEVANCE]
    evidence = list(state.get("evidence", []))
    seen = {(e["document_id"], e["text"]) for e in evidence}
    for entry in new:
        key = (entry["document_id"], entry["text"])
        if key not in seen:
            evidence.append(entry)
            seen.add(key)
    return {"evidence": evidence, "research_failed": False,
            "events": _event(state, type="ResearchCompleted", count=len(new), retries=retries)}


@traceable(name="policy.evidence_check")
async def evidence_node(state: AgentState, deps: SupervisorDeps) -> dict:
    sufficient = bool(state.get("evidence"))
    rounds = state.get("research_rounds", 0)
    if (not sufficient and not state.get("research_failed") and
            not state.get("policy_cue") and not state.get("is_follow_up")):
        return {"out_of_scope": True, "retry_research": False,
                "events": _event(state, type="ReplanCheck", is_sufficient=False,
                                 round=rounds, reason="no relevant policy evidence")}
    if not sufficient and rounds < 1:
        query = state["context_query"] + " 관련 규정 적용 조건 예외 신청 서류 절차"
        return {"research_rounds": 1, "search_query": query, "retry_research": True,
                "events": _event(state, type="ReplanCheck", is_sufficient=False, round=1)}
    return {"out_of_scope": False, "retry_research": False,
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
        procedure += ("\n질문 유형: " + state.get("question_type", "explanation") +
                      ". 서류와 절차 질문에서는 신청 자격을 판정하려 하지 말고 질문한 항목만 추출하라.")
        result, usage = deps.policy_llm.analyze(state["context_query"],
            state.get("relevant_history", []), state["evidence"], procedure)
    result = sanitize_analysis(result, state.get("evidence", []))
    if DOCUMENT_QUESTION.search(state["actual_question"]):
        action = re.search(r"취소|포기", state["actual_question"])
        if action:
            result["documents"] = [item for item in result.get("documents", [])
                                   if action.group() in item.get("quote", "")]
    if DOCUMENT_QUESTION.search(state["actual_question"]) and not result.get("documents"):
        result["eligibility"] = "need_more_information"
        result["reason"] = "검색된 규정에서 질문하신 절차에 필요한 서류를 확인하지 못했습니다."
        result["reason_support"] = {}
        result["conditions"] = []
        result["procedure"] = []
        result["deadline"] = {}
        result["department"] = {}
        result["related_context"] = _related_cancellation_rule(state["actual_question"], state.get("evidence", []))
        quote = result["related_context"].get("quote", "")
        if "졸업신청자" in quote and re.search(r"요건을\s*충족하지\s*못", quote):
            subject = "부전공" if "부전공" in state["actual_question"] else "복수전공"
            result["reason"] += (f" 검색된 관련 조항은 졸업신청자가 {subject} 이수 요건을 "
                                 "충족하지 못한 경우를 설명합니다.")
        result["follow_up_question"] = ("담당 부서에 해당 절차의 제출 서류를 확인해 주세요."
                                        if state.get("research_rounds", 0) else "")
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


def _render(analysis: dict, evidence: list[dict], question_type: str = "eligibility") -> str:
    labels = {"eligible": "신청 가능", "conditional": "조건부 가능",
              "ineligible": "신청 불가", "need_more_information": "추가 확인 필요"}
    if question_type == "eligibility":
        lines = [f"판단: {labels[analysis['eligibility']]}", analysis.get("reason", "")]
        fields = (("conditions", "적용 조건"), ("documents", "필요 서류"),
                  ("procedure", "신청 절차"))
    elif question_type == "documents":
        lines = [] if analysis.get("documents") else [analysis.get("reason", "필요 서류를 확인하지 못했습니다.")]
        fields = (("documents", "필요 서류"), ("procedure", "신청 절차"))
    elif question_type == "procedure":
        lines = [] if analysis.get("procedure") else [analysis.get("reason", "절차를 확인하지 못했습니다.")]
        fields = (("procedure", "신청 절차"), ("documents", "필요 서류"))
    else:
        lines = [analysis.get("reason", "")] if analysis.get("reason_support") else []
        fields = (("conditions", "규정 내용"), ("procedure", "관련 절차"),
                  ("documents", "필요 서류"))
    for key, label in fields:
        items = _normalise_items(analysis.get(key, []))
        if items:
            lines.append(label + ": " + "; ".join(re.sub(r"\s+", " ", x["text"]).strip() for x in items))
    for key, label in (("deadline", "신청 기한"), ("department", "담당 부서")):
        value = analysis.get(key, {})
        if isinstance(value, dict) and value.get("text"):
            clean_text = re.sub(r"\s+", " ", value["text"]).strip()
            lines.append(f"{label}: {clean_text}")
    related = analysis.get("related_context", {})
    if isinstance(related, dict) and related.get("text"):
        lines.append("참고 규정: " + related["text"])
    if analysis.get("follow_up_question"):
        lines.append("추가 확인: " + analysis["follow_up_question"])
    support = analysis.get("reason_support") or {}
    cited = {support.get("citation_id") if isinstance(support, dict) else None}
    for key in ("conditions", "documents", "procedure"):
        cited.update(item.get("citation_id") for item in _normalise_items(analysis.get(key, [])))
    for key in ("deadline", "department", "related_context"):
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
    return {"final": _render(state["analysis"], state.get("evidence", []), state.get("question_type", "eligibility")),
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
        return {"analysis": safe, "final": _render(safe, [], state.get("question_type", "eligibility")), "validation_attempts": attempts,
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


async def form_lookup_node(state: AgentState, deps: SupervisorDeps) -> dict:
    asks_for_form = bool(FORM_TERMS.search(state["actual_question"]))
    evidence_requires_form = any(FORM_REQUIREMENT.search(e.get("text", ""))
                                 for e in state.get("evidence", []))
    if not (asks_for_form or evidence_requires_form):
        return {"forms": []}
    try:
        result, _ = await with_retry(deps.mcp.call_tool, "search_application_forms",
            {"query": state["context_query"], "domain": state["domain"]},
            server="policy", retries=1)
        if "result" in result and isinstance(result["result"], dict):
            result = result["result"]
        forms = result.get("results", [])
        action = re.search(r"취소|포기", state["actual_question"])
        if action:
            forms = [form for form in forms if action.group() in form.get("title", "")]
    except Exception:
        forms = []
    return {"forms": forms,
            "events": _event(state, type="FormsSearched", count=len(forms))}


def build_graph(deps: SupervisorDeps):
    graph = StateGraph(AgentState)
    for name, fn in (("route", route_node), ("chat", chat_node), ("context", context_node),
                     ("plan", plan_node), ("research", research_node), ("evidence", evidence_node),
                     ("analysis", analysis_node), ("missing", missing_node), ("write", write_node),
                     ("validate", validate_node), ("forms", form_lookup_node),
                     ("finalize", finalize_node)):
        graph.add_node(name, partial(fn, deps=deps))
    graph.set_entry_point("route")
    graph.add_conditional_edges("route", lambda s: "chat" if s["route"] == "chat" else "context")
    graph.add_edge("chat", END)
    graph.add_edge("context", "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "evidence")
    graph.add_conditional_edges("evidence", lambda s: "chat" if s.get("out_of_scope") else
                                "research" if s.get("retry_research") else "analysis")
    graph.add_conditional_edges("analysis", lambda s: "research" if s.get("analysis_research_needed") else "missing")
    graph.add_edge("missing", "write")
    graph.add_edge("write", "validate")
    graph.add_conditional_edges("validate", lambda s: "analysis" if s.get("validation_feedback") else "forms")
    graph.add_edge("forms", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
