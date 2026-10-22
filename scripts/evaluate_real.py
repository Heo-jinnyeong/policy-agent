"""Run a reproducible local comparison against annotated, indexed policy text.

Both paths use the current Ollama models and Chroma index. The baseline performs
one Top-3 search and one analysis call. The agentic path invokes the actual
LangGraph with MCP search, validation and optional retries. It runs locally so
the evaluator can inspect all retrieved evidence, not only API-cited sources.
"""

import argparse
import asyncio
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from dotenv import load_dotenv

load_dotenv()

from agent.deps import SupervisorDeps
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.policy_llm import PolicyLLM
from agent.session_store import SqliteSessionStore
from rag.retriever import search_policies
from rag.vectorstore import open_store
from server import _build_llms


def compact(value):
    return re.sub(r"\s+", "", str(value or "")).lower()


def is_gold(entry, gold):
    return (entry.get("document_id") == gold["document_id"]
            and entry.get("page") == gold["page"]
            and compact(entry.get("article")) == compact(gold["article"]))


def citation_items(analysis):
    items = []
    for field in ("reason_support", "deadline", "department", "related_context"):
        item = analysis.get(field)
        if isinstance(item, dict) and item.get("citation_id"):
            items.append(item)
    for field in ("conditions", "documents", "procedure"):
        items.extend(item for item in analysis.get(field, [])
                     if isinstance(item, dict) and item.get("citation_id"))
    return items


def answer_text(analysis):
    parts = [analysis.get("reason", "")]
    for field in ("conditions", "documents", "procedure"):
        parts.extend(item.get("text", "") for item in analysis.get(field, [])
                     if isinstance(item, dict))
    for field in ("deadline", "department", "related_context"):
        item = analysis.get(field)
        if isinstance(item, dict):
            parts.append(item.get("text", ""))
    return " ".join(part for part in parts if isinstance(part, str) and part)


def fact_coverage(case, answer):
    checks = case["answer_checks"]
    passed = sum(any(compact(option) in compact(answer) for option in alternatives)
                 for alternatives in checks)
    return round(passed / len(checks), 3)


def score(case, mode, evidence, analysis, seconds, tokens, rounds=0):
    gold = case["gold"]
    by_id = {item["id"]: item for item in evidence}
    citations = citation_items(analysis)
    answer = answer_text(analysis)
    return {
        "mode": mode,
        "question": case["question"],
        "gold_page": gold["page"],
        "gold_article": gold["article"],
        "retrieval_hit_at_3": int(any(is_gold(item, gold) for item in evidence[:3])),
        "gold_article_cited": int(any(is_gold(by_id.get(item["citation_id"], {}), gold)
                                      for item in citations)),
        "citation_quote_valid": int(bool(citations) and all(
            item.get("quote") and item["quote"] in by_id.get(item["citation_id"], {}).get("text", "")
            for item in citations)),
        "answer_fact_coverage": fact_coverage(case, answer),
        "latency_seconds": round(seconds, 3),
        "tokens": tokens,
        "research_rounds": rounds,
        "retrieved_articles": " | ".join(f"{e.get('page')}쪽 {e.get('article')}" for e in evidence[:3]),
        "answer_text": answer,
    }


def validate_cases(cases, store):
    if len(cases) < 10:
        raise ValueError("At least 10 real-document cases are required")
    data = store.get(where={"document_id": cases[0]["gold"]["document_id"]},
                     include=["documents", "metadatas"])
    rows = [dict(text=text, **meta) for text, meta in zip(data["documents"], data["metadatas"])]
    for case in cases:
        matches = [entry for entry in rows if is_gold(entry, case["gold"])]
        if not matches:
            raise ValueError(f"Gold source absent from active index: {case['question']}")
        original = compact(" ".join(entry["text"] for entry in matches))
        for alternatives in case["answer_checks"]:
            if not any(compact(option) in original for option in alternatives):
                raise ValueError(f"Expected fact absent from gold source: {case['question']} {alternatives}")
    return rows[0].get("content_hash", ""), len(data["ids"])


def average(rows, field):
    return sum(row[field] for row in rows) / len(rows)


def save_summary(path, cases_path, rows, model, embedding, digest, chunk_count):
    modes = {mode: [row for row in rows if row["mode"] == mode]
             for mode in ("baseline", "agentic")}
    lines = ["# 실제 학칙 기반 평가", "",
             f"집계 시각: {datetime.now().astimezone().isoformat(timespec='seconds')}",
             f"정답표: `{cases_path}` · 문항 {len(modes['baseline'])}개",
             f"규정 색인: 학사 규정집 PDF 1개, {chunk_count}개 조각 · SHA-256 {digest}",
             f"답변 모델: {model} · 임베딩 모델: {embedding}", "",
             "단일 검색 RAG는 Top-3 검색 후 같은 답변 모델을 한 번 호출했다. Agentic RAG는 실제 LangGraph와 MCP 검색, 검증, 필요 시 재검색을 실행했다.",
             "두 방식 모두 같은 로컬 색인과 모델을 사용했다. 10개 문항은 색인된 규정 원문에서 수작업으로 선정했고, 실행 전 정답 조항과 핵심 사실이 색인에 있는지 검사했다.", "",
             "| 측정 항목 | 단일 검색 RAG | Agentic RAG |", "|---|---:|---:|"]
    fields = [("정답 조항 Top-3 포함", "retrieval_hit_at_3", "%"),
              ("정답 조항 인용", "gold_article_cited", "%"),
              ("인용 문구 원문 일치", "citation_quote_valid", "%"),
              ("핵심 사실 포함", "answer_fact_coverage", "%"),
              ("평균 응답시간", "latency_seconds", "초"),
              ("기록된 모델 토큰", "tokens", "개")]
    for label, field, unit in fields:
        values = []
        for mode in ("baseline", "agentic"):
            value = sum(row[field] for row in modes[mode]) if field == "tokens" else average(modes[mode], field)
            values.append(f"{value:.0%}" if unit == "%" else f"{value:.2f}초" if unit == "초" else f"{value:,}개")
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += ["", "## 문항별 확인", "",
              "| 질문 | 정답 위치 | 단일 검색 Top-3 / 핵심 사실 | Agentic Top-3 / 핵심 사실 |",
              "|---|---|---:|---:|"]
    for case in json.loads(cases_path.read_text()):
        pair = [next(row for row in modes[mode] if row["question"] == case["question"])
                for mode in ("baseline", "agentic")]
        formatted = [f"{row['retrieval_hit_at_3']} / {row['answer_fact_coverage']:.0%}" for row in pair]
        lines.append(f"| {case['question']} | {case['gold']['page']}쪽 {case['gold']['article']} | {formatted[0]} | {formatted[1]} |")
    lines += ["", "## 해석 범위", "",
              "- Top-3 포함은 문서 ID·PDF 쪽수·조항 제목이 모두 같은 경우만 정답으로 셌다.",
              "- 정답 조항 인용은 구조화 답변의 citation_id가 정답 조항을 가리킬 때만 셌다. 인용 문구 원문 일치는 제시된 모든 인용이 실제 검색 텍스트에 그대로 있을 때만 인정했다.",
              "- 핵심 사실 포함은 정답표의 짧은 필수 표현이 구조화 답변 텍스트에 나타나는 비율이다. 동의어를 모두 포괄하지 못하므로 의미적 정답률과 같지 않다.",
              "- 평가 범위는 현재 색인된 규정집 1개와 선정한 10개 질문이다. 전체 대학요람이나 모든 질문에 대한 정확도로 일반화할 수 없다.",
              "- 토큰은 현재 그래프에 기록된 품질 모델의 분석·심사 사용량 합계다. 재입학 질문의 빠른 분류 모델 사용량은 그래프 집계에서 빠져 있어 에이전트의 실제 총 토큰은 표보다 많다. 비용 청구액이 아니다.",
              "- 같은 로컬 컴퓨터에서 순차 실행한 시간으로 외부 환경의 지연시간을 예측할 수 없다.", ""]
    path.write_text("\n".join(lines))


async def run(args):
    cases = json.loads(args.cases.read_text())
    store = open_store()
    digest, chunk_count = validate_cases(cases, store)
    backend, fast, quality = _build_llms("ollama")
    policy_llm = PolicyLLM(quality)
    mcp = MCPClient()
    session_store = SqliteSessionStore(Path(":memory:"))
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        await mcp.start()
        deps = SupervisorDeps(fast_llm=fast, quality_llm=quality, mcp=mcp,
                              session_store=session_store, version="policy-v2",
                              policy_llm=policy_llm)
        graph = build_graph(deps)
        with args.output.open("w", newline="") as handle:
            writer = None
            for index, case in enumerate(cases, 1):
                start = time.perf_counter()
                evidence = search_policies(case["question"], case["domain"], 3, store=store)
                analysis, usage = policy_llm.analyze(case["question"], [], evidence)
                row = score(case, "baseline", evidence, analysis,
                            time.perf_counter() - start, usage.get("tokens", 0))
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
                handle.flush()
                print(f"{index}/{len(cases)} baseline: hit={row['retrieval_hit_at_3']} fact={row['answer_fact_coverage']}", flush=True)

                start = time.perf_counter()
                result = await graph.ainvoke({"query": case["question"],
                    "thread_id": f"real-eval-{index}", "domain": case["domain"],
                    "history": [], "events": [], "cost_usd": 0, "tokens": 0})
                row = score(case, "agentic", result.get("evidence", []),
                            result.get("analysis", {}), time.perf_counter() - start,
                            result.get("tokens", 0), result.get("research_rounds", 0))
                writer.writerow(row)
                rows.append(row)
                handle.flush()
                print(f"{index}/{len(cases)} agentic: hit={row['retrieval_hit_at_3']} fact={row['answer_fact_coverage']}", flush=True)
    finally:
        await mcp.stop()
        session_store.close()
    save_summary(args.output.with_suffix(".md"), args.cases, rows, quality.name,
                 os.getenv("POLICY_EMBEDDING_MODEL", "qwen3-embedding:0.6b"), digest, chunk_count)
    print(args.output, args.output.with_suffix(".md"), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("docs/evaluation-real-cases.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/evaluation-real-results.csv"))
    parser.add_argument("--rescore", action="store_true", help="저장된 답변을 현재 정답 표현으로 다시 채점")
    args = parser.parse_args()
    if args.rescore:
        cases = json.loads(args.cases.read_text())
        digest, chunk_count = validate_cases(cases, open_store())
        by_question = {case["question"]: case for case in cases}
        with args.output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(cases) * 2:
            raise ValueError("Saved result does not contain both modes for every case")
        for row in rows:
            row["answer_fact_coverage"] = fact_coverage(by_question[row["question"]], row["answer_text"])
            for key in ("gold_page", "retrieval_hit_at_3", "gold_article_cited",
                        "citation_quote_valid", "tokens", "research_rounds"):
                row[key] = int(row[key])
            row["latency_seconds"] = float(row["latency_seconds"])
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _, _, quality = _build_llms("ollama")
        save_summary(args.output.with_suffix(".md"), args.cases, rows, quality.name,
                     os.getenv("POLICY_EMBEDDING_MODEL", "qwen3-embedding:0.6b"), digest, chunk_count)
        print(args.output, args.output.with_suffix(".md"))
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
