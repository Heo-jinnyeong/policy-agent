"""Measure one-search baseline and LangGraph service on the same indexed corpus.

Cases must contain question, domain, expected_article and expected_status.
Optional expected_documents, expected_procedure and expected_follow_up improve coverage.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from agent.policy_llm import PolicyLLM
from rag.retriever import search_policies
from server import _build_llms


def metrics(rows):
    count = len(rows)
    if not count:
        return {}
    annotated = [r["completeness"] for r in rows if r["completeness"] is not None]
    followups = [r["follow_up_correct"] for r in rows if r["follow_up_correct"] is not None]
    return {
        "retrieval_hit_at_3": sum(r["hit_at_3"] for r in rows) / count,
        "eligibility_accuracy": sum(r["eligibility_correct"] for r in rows) / count,
        "citation_accuracy": sum(r["citation_correct"] for r in rows) / count,
        "document_procedure_completeness": sum(annotated) / len(annotated) if annotated else None,
        "follow_up_appropriateness": sum(followups) / len(followups) if followups else None,
        "average_latency_seconds": sum(r["latency_seconds"] for r in rows) / count,
        "total_tokens": sum(r["tokens"] for r in rows),
        "estimated_cost_usd": sum(r["estimated_cost_usd"] for r in rows),
    }


def score(case, mode, evidence, analysis, seconds, tokens, cost):
    article = case.get("expected_article")
    cited_items = [x for key in ("documents", "conditions", "procedure")
                   for x in analysis.get(key, []) if isinstance(x, dict)]
    support = analysis.get("reason_support") or {}
    if isinstance(support, dict):
        if support.get("citation_id"):
            cited_items.append(support)
    for key in ("deadline", "department"):
        value = analysis.get(key)
        if isinstance(value, dict) and value.get("text"):
            cited_items.append(value)
    by_id = {e["id"]: e["text"] for e in evidence}
    expected_fields = case.get("expected_documents", []) + case.get("expected_procedure", [])
    answer_fields = " ".join(x.get("text", "") for key in ("documents", "procedure")
                             for x in analysis.get(key, []) if isinstance(x, dict))
    return {"mode": mode, "question": case["question"],
            "hit_at_3": int(any(article in (e.get("article") or "") for e in evidence[:3]) if article else not evidence),
            "eligibility_correct": int(analysis.get("eligibility") == case.get("expected_status")),
            "citation_correct": int(bool(cited_items) and all(
                x.get("quote") and x["quote"] in by_id.get(x.get("citation_id"), "")
                for x in cited_items)) if evidence else int(not cited_items),
            "completeness": (sum(field in answer_fields for field in expected_fields) / len(expected_fields)
                             if expected_fields else None),
            "follow_up_correct": (int(bool(analysis.get("follow_up_question")) == bool(case["expected_follow_up"]))
                                  if "expected_follow_up" in case else None),
            "latency_seconds": round(seconds, 3), "tokens": tokens,
            "estimated_cost_usd": cost}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8100")
    parser.add_argument("--output", type=Path, default=Path("docs/evaluation-results.csv"))
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if len(cases) < 10:
        raise ValueError("At least 10 reference cases are required")
    _, _, quality = _build_llms()
    llm = PolicyLLM(quality)
    rows = []
    with httpx.Client(timeout=360) as client:
        for index, case in enumerate(cases):
            start = time.perf_counter()
            evidence = search_policies(case["question"], case["domain"], 3)
            analysis, usage = llm.analyze(case["question"], [], evidence)
            rows.append(score(case, "baseline", evidence, analysis,
                              time.perf_counter() - start, usage.get("tokens", 0), usage.get("usd", 0)))
            start = time.perf_counter()
            response = client.post(args.server + "/chat", json={"message": case["question"],
                "domain": case["domain"], "thread_id": f"eval-{index}-{int(start)}"})
            response.raise_for_status()
            result = response.json()
            metadata = result["metadata"]
            rows.append(score(case, "agentic", result["sources"], result["analysis"],
                time.perf_counter() - start, metadata["tokens"], metadata["estimated_cost_usd"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {mode: metrics([r for r in rows if r["mode"] == mode]) for mode in ("baseline", "agentic")}
    summary_path = args.output.with_suffix(".md")
    note = ("**가상 테스트 문서 기반 결과이며 실제 규정 서비스 정확도가 아닙니다.**\n\n"
            if "fixtures" in args.cases.parts else "")
    configuration = (f"LLM: {os.getenv('LLM_BACKEND', 'fake')} / "
                     f"{os.getenv('OLLAMA_QUALITY_MODEL') or os.getenv('OPENAI_QUALITY_MODEL') or 'default'}; "
                     f"임베딩: {os.getenv('POLICY_EMBEDDING_BACKEND', 'default')} / "
                     f"{os.getenv('POLICY_EMBEDDING_MODEL', 'default')}\n\n")
    summary_path.write_text("# 실측 평가 결과\n\n" + note + "데이터: " + str(args.cases) +
                            "\n\n" + configuration + "```json\n" +
                            json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n")
    print(args.output, summary_path)


if __name__ == "__main__":
    main()
