"""Structured policy analysis. Evidence quotes are checked again in graph.py."""
import json
import logging

from langsmith import traceable

SYSTEM = """당신은 한국어 규정 분석가다. 반드시 JSON 객체만 출력하라.
제공된 규정 발췌만 근거로 판단하라. 규정에 없는 서류명, 기한, 부서, 조항을 만들지 마라.
스키마: {"eligibility":"eligible|conditional|ineligible|need_more_information",
"reason":"간결한 판단 사유", "reason_support":{"citation_id":"검색 결과 ID","quote":"본문 그대로의 짧은 인용"},
"conditions":[{"text":"조건","citation_id":"ID","quote":"본문 그대로"}],
"documents":[{"text":"서류","citation_id":"ID","quote":"본문 그대로"}],
"procedure":[{"text":"절차","citation_id":"ID","quote":"본문 그대로"}],
"deadline":{"text":"기한","citation_id":"ID","quote":"본문 그대로"},
"department":{"text":"부서","citation_id":"ID","quote":"본문 그대로"},
"follow_up_question":"적용 판단에 필수적인 사용자 조건이 없을 때만 질문, 아니면 빈 문자열"}.
conditions/documents/procedure의 text는 해당 quote 안에 실제로 포함된 80자 이하의 짧은 구절로 적어라.
deadline/department는 배열이 아닌 단일 객체다. 해당 문구가 원문에 없으면 빈 객체로 적어라.
"기한 없음"처럼 원문이 말하지 않은 부정 사실을 만들지 마라.
근거를 찾지 못한 항목은 빈 배열 또는 빈 객체로 표시하라. 시행일이 다른 규정이 충돌하면 최신 적용 여부를 확인하고 불명확하면 유보하라. 사용자 정보 부족이나 규정 부족이면
eligibility=need_more_information으로 하라. 절차나 규정 내용을 묻는 일반 질문에는 특정인의 신청 자격을 추측하지 말고, 확인된 절차와 조건을 근거와 함께 설명하라.
명시적인 금지 조항이 없는 한 eligibility=ineligible로 판단하지 마라. 이유에는 규정에 실제로 적힌 내용을 쓰고, 근거가 있는 절차를 없다고 말하지 마라.
부전공과 복수전공처럼 이름이 비슷한 다른 제도의 규정을 질문 대상에 적용하지 마라.
필요 서류를 묻는 질문에는 해당 제도의 해당 절차에 명시된 서류만 답하라. 서류 근거를 찾지 못했다면 다른 조건이나 전공 규정으로 답을 채우지 말고 확인이 필요하다고 답하라.
서류·절차 질문에는 자격 판정보다 질문에 직접 답하는 서류·절차 필드를 우선 채워라. 관련 규정이 있어도 질문한 서류가 명시되지 않았으면 documents는 빈 배열이다.
"어떤 경우"를 묻는 질문에서는 해당 조항의 각 항과 각 호를 모두 확인하고, 서로 다른 인정 사유를 빠뜨리지 마라. procedure에는 실제 신청·제출·접수 행위만 넣고, 인정 결과나 조건은 넣지 마라.
필요 없는 추가 질문은 하지 마라."""


class PolicyLLM:
    def __init__(self, backend):
        self.backend = backend
        self.name = getattr(backend, "name", "unknown")

    @traceable(run_type="llm", name="policy.analyze")
    def analyze(self, query: str, history: list[dict], evidence: list[dict],
                procedure_prompt: str = "") -> tuple[dict, dict]:
        if not hasattr(self.backend, "_chat"):
            return {"eligibility": "need_more_information", "reason": "실제 LLM이 연결되지 않아 규정 판단을 보류합니다."}, {"tokens": 0, "usd": 0}
        sources = [{"id": e["id"], "text": e["text"], "document_name": e["document_name"],
                    "article": e.get("article", ""), "page": e.get("page", 0),
                    "effective_date": e.get("effective_date", "")} for e in evidence]
        previous = history[-6:]
        user = json.dumps({"question": query, "history": previous, "instructions": procedure_prompt,
                           "evidence": sources}, ensure_ascii=False)
        try:
            content, usage = self.backend._chat(SYSTEM, user, json_mode=True)
            return json.loads(content), usage
        except Exception as exc:
            logging.getLogger(__name__).warning("policy analysis failed: %s", type(exc).__name__)
            return {"eligibility": "need_more_information", "reason": "모델 호출 또는 응답 해석에 실패해 판단을 보류합니다."}, {"tokens": 0, "usd": 0}

    @traceable(run_type="llm", name="policy.judge")
    def judge(self, analysis: dict, evidence: list[dict]):
        if not hasattr(self.backend, "_chat"):
            return True, {"tokens": 0, "usd": 0}
        prompt = json.dumps({"analysis": analysis, "evidence": evidence}, ensure_ascii=False)
        system = "규정 분석의 근거 일치, 신청 자격, 서류 및 절차 정확성을 심사하라. JSON {\"valid\":boolean}만 답하라. 모호하면 false."
        try:
            content, usage = self.backend._chat(system, prompt, json_mode=True)
            return json.loads(content).get("valid") is True, usage
        except Exception:
            return False, {"tokens": 0, "usd": 0}
